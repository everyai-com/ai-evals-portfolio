"""Drift monitor engine: sampling, offline scoring, per-day metrics and drift detection.

Statistics (all formulas live here so the README claims are checkable)
----------------------------------------------------------------------
Sampling     Deterministic weighted reservoir (Efraimidis & Spirakis 2006, "A-Res"): every row gets
             u = sha256(seed:date:task_id) -> (0, 1] and key = u ** (1 / w). Within each stratum
             the k rows with the largest keys are kept. With w = 1 this is a uniform k-of-n sample;
             w > 1 (e.g. thumbs-down feedback) makes a row proportionally more likely to be kept.
             k per stratum is allocated by largest remainder so the total is round(rate * n).
Wilson CI    evalkit.metrics.wilson_ci.
CUSUM        Page (1954), one-sided downward, standardised: with mu0 / sigma0 from the baseline,
             z_t = (x_t - mu0) / sigma0 and S_t = max(0, S_{t-1} - z_t - k). Alert when S_t > h.
z-score      z = (p_base - p_today) / sqrt(p_base (1 - p_base) / n_today); plus the Wilson
             non-overlap check today_hi < base_lo.
PSI          Population Stability Index sum_i (q_i - p_i) * ln(q_i / p_i) over shared bins. Bin
             shares use Jeffreys smoothing (count + 0.5) / (n + 0.5 k) so empty bins stay finite
             without exploding. Under no drift PSI is not zero: PSI * n_b n_c / (n_b + n_c) is
             roughly chi-square(k-1), so E[PSI] ~ (k-1)(1/n_b + 1/n_c); the detector subtracts
             that expected noise before comparing to the usual 0.1 / 0.25 thresholds.
             Significance gate: PSI * n_b n_c / (n_b + n_c) ~ chi-square(k-1) under H0, so the
             alert also needs chi2_sf(stat, k-1) <= max_p.
Hallucination spike  today_rate >= ratio * base_rate, today_rate - base_rate >= min_abs, at least
             min_count flags, and binomial tail P(X >= k | n_today, base_rate) <= max_p.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from evalkit.llm import LLMClient
from evalkit.metrics import wilson_ci
from evalkit.trajectory import Trajectory

# --------------------------------------------------------------------------- sampling

def _unit(seed: int, date: str, key: str) -> float:
    """Deterministic uniform in (0, 1] keyed on (seed, date, task_id)."""
    h = hashlib.sha256(f"{seed}:{date}:{key}".encode()).hexdigest()
    return (int(h[:12], 16) + 1) / (0x1_0000_0000_0000 + 1)


def _largest_remainder(sizes: dict[Any, int], total: int) -> dict[Any, int]:
    n = sum(sizes.values())
    if n == 0 or total <= 0:
        return {k: 0 for k in sizes}
    raw = {k: total * v / n for k, v in sizes.items()}
    alloc = {k: int(math.floor(r)) for k, r in raw.items()}
    short = total - sum(alloc.values())
    for k in sorted(sizes, key=lambda k: (-(raw[k] - alloc[k]), str(k)))[:short]:
        alloc[k] += 1
    return {k: min(v, sizes[k]) for k, v in alloc.items()}


class TrafficSampler:
    """Stratified (by hour or tenant) deterministic sample of a day's trajectory log."""

    def __init__(self, rate: float = 0.05, seed: int = 0, stratify_by: str = "hour",
                 feedback_weight: float = 1.0, weight_fn: Callable[[dict[str, Any]], float] | None = None,
                 min_total: int = 1) -> None:
        if not 0 < rate <= 1:
            raise ValueError("rate must be in (0, 1]")
        self.rate, self.seed, self.stratify_by = rate, seed, stratify_by
        self.feedback_weight = feedback_weight
        self.weight_fn = weight_fn
        self.min_total = min_total

    def stratum(self, row: dict[str, Any]) -> Any:
        meta = row.get("meta", {})
        if self.stratify_by == "hour":
            if "hour" in meta:
                return int(meta["hour"])
            ts = str(meta.get("ts", ""))
            m = re.search(r"T(\d{2})", ts)
            return int(m.group(1)) if m else 0
        return meta.get(self.stratify_by, "unknown")

    def weight(self, row: dict[str, Any]) -> float:
        if self.weight_fn is not None:
            return max(1e-9, float(self.weight_fn(row)))
        fb = (row.get("meta") or {}).get("feedback")
        return self.feedback_weight if fb in ("down", "thumbs_down", -1, "negative") else 1.0

    def sample(self, rows: Sequence[dict[str, Any]], date: str) -> list[dict[str, Any]]:
        groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            groups[self.stratum(r)].append(r)
        total = max(self.min_total, int(round(self.rate * len(rows)))) if rows else 0
        quotas = _largest_remainder({k: len(v) for k, v in groups.items()}, total)
        out: list[dict[str, Any]] = []
        for k in sorted(groups, key=str):
            keyed = [(_unit(self.seed, date, str(r.get("task_id"))) ** (1.0 / self.weight(r)), r)
                     for r in groups[k]]
            keyed.sort(key=lambda kr: (-kr[0], str(kr[1].get("task_id"))))
            out.extend(r for _, r in keyed[:quotas[k]])
        return out


# --------------------------------------------------------------------------- scorers

class Scorer(Protocol):
    name: str

    def score(self, traj: Trajectory) -> dict[str, Any]: ...


DEFAULT_TOOL_SCHEMA: dict[str, dict[str, list[str]]] = {
    "lookup_order": {"required": ["order_id"]},
    "search_menu": {"required": ["query"]},
    "issue_refund": {"required": ["order_id", "amount"]},
    "get_store_hours": {"required": ["store_id"]},
}


class RuleScorer:
    """Deterministic checks: tool calls valid against a schema, final answer non-empty, no error steps."""
    name = "rule"

    def __init__(self, schema: dict[str, dict[str, list[str]]] | None = None) -> None:
        self.schema = schema if schema is not None else DEFAULT_TOOL_SCHEMA

    def score(self, traj: Trajectory) -> dict[str, Any]:
        invalid: list[str] = []
        for tc in traj.tool_calls:
            spec = self.schema.get(tc.name)
            if spec is None:
                invalid.append(f"unknown tool {tc.name}")
                continue
            missing = [a for a in spec.get("required", []) if a not in (tc.args or {})]
            if missing:
                invalid.append(f"{tc.name} missing {','.join(missing)}")
        errors = [s for s in traj.steps if (s.tool_call and s.tool_call.error) or s.meta.get("error")]
        nonempty = bool(traj.final_answer and traj.final_answer.strip())
        return {"tools_valid": not invalid, "invalid_tool_calls": invalid, "answer_nonempty": nonempty,
                "no_error_steps": not errors, "error_steps": len(errors),
                "success": (not invalid) and nonempty and (not errors)}


HALLUCINATION_MARKERS = ["i personally guarantee", "our records confirm a 100% refund",
                         "store #999", "as the ceo told me", "double loyalty points for life"]

JUDGE_SYSTEM = ("You are a strict grader. Given a customer-support agent's final answer and the tool "
                "results it saw, reply HALLUCINATION if the answer asserts facts not supported by the "
                "tool results, otherwise reply OK.")


def default_judge_handler(system: str, prompt: str) -> str:
    """Scripted judge used by the MockLLM: flags the synthetic hallucination markers."""
    p = prompt.lower()
    return "HALLUCINATION" if any(m in p for m in HALLUCINATION_MARKERS) else "OK"


class JudgeScorer:
    """LLM-as-judge hallucination flag. Offline the client is a MockLLM scripted with `default_judge_handler`."""
    name = "judge"

    def __init__(self, client: LLMClient, system: str = JUDGE_SYSTEM) -> None:
        self.client = client
        self.system = system

    def score(self, traj: Trajectory) -> dict[str, Any]:
        if not traj.final_answer.strip():
            return {"hallucination": False, "judge_raw": "EMPTY"}
        evidence = "\n".join(f"- {tc.name}({json.dumps(tc.args, sort_keys=True)}) -> {json.dumps(tc.result, default=str)[:200]}"
                             for tc in traj.tool_calls)
        prompt = f"TOOL RESULTS:\n{evidence or '(none)'}\n\nFINAL ANSWER:\n{traj.final_answer}\n\nVerdict:"
        try:
            text = self.client.complete(prompt, system=self.system, max_tokens=8).text
        except Exception as e:  # noqa: BLE001 - a flaky judge must not take the nightly job down
            return {"hallucination": False, "judge_raw": f"ERROR {type(e).__name__}"}
        return {"hallucination": "HALLUCINATION" in text.upper(), "judge_raw": text.strip()[:40]}


class LengthScorer:
    name = "length"

    def score(self, traj: Trajectory) -> dict[str, Any]:
        ans = traj.final_answer or ""
        return {"answer_chars": len(ans), "answer_words": len(ans.split()), "n_steps": len(traj.steps),
                "n_tool_calls": len(traj.tool_calls)}


# --------------------------------------------------------------------------- per-day metrics

@dataclass
class DayMetrics:
    date: str
    n: int
    success_rate: float
    hallucination_rate: float
    avg_steps: float
    p95_latency_ms: float
    cost_per_request: float
    tools_valid_rate: float
    error_step_rate: float
    empty_answer_rate: float
    avg_answer_words: float
    tool_call_counts: list[int] = field(default_factory=list)
    answer_lengths: list[int] = field(default_factory=list)
    per_tenant: dict[str, dict[str, float]] = field(default_factory=dict)
    failed_ids: list[str] = field(default_factory=list)
    hallucinated_ids: list[str] = field(default_factory=list)
    feedback_down_rate: float = 0.0

    @property
    def successes(self) -> int:
        return int(round(self.success_rate * self.n))

    @property
    def hallucinations(self) -> int:
        return int(round(self.hallucination_rate * self.n))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DayMetrics:
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


class OfflineEvaluator:
    def __init__(self, scorers: Sequence[Scorer]) -> None:
        self.scorers = list(scorers)

    def score_one(self, traj: Trajectory) -> dict[str, Any]:
        out: dict[str, Any] = {"task_id": traj.task_id, "tenant": traj.meta.get("tenant", "unknown"),
                               "feedback": traj.meta.get("feedback"), "latency_ms": traj.total_latency_ms,
                               "cost_usd": traj.total_cost, "n_steps": len(traj.steps),
                               "n_tool_calls": len(traj.tool_calls),
                               "answer_words": len((traj.final_answer or "").split())}
        for s in self.scorers:
            out.update(s.score(traj))
        out.setdefault("success", bool(out.get("answer_nonempty", bool(traj.final_answer))))
        out.setdefault("hallucination", False)
        return out

    def evaluate(self, rows: Iterable[dict[str, Any] | Trajectory], date: str) -> DayMetrics:
        trajs = [r if isinstance(r, Trajectory) else Trajectory.from_dict(r) for r in rows]
        recs = [self.score_one(t) for t in trajs]
        n = len(recs)
        if n == 0:
            return DayMetrics(date, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        lat = np.array([r["latency_ms"] for r in recs], float)
        by_tenant: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in recs:
            by_tenant[str(r["tenant"])].append(r)
        per_tenant = {}
        for t, rs in sorted(by_tenant.items()):
            k = sum(r["success"] for r in rs)
            lo, hi = wilson_ci(k, len(rs))
            per_tenant[t] = {"n": len(rs), "success_rate": k / len(rs), "success_ci_lo": lo, "success_ci_hi": hi,
                             "hallucination_rate": sum(r["hallucination"] for r in rs) / len(rs),
                             "cost_per_request": float(np.mean([r["cost_usd"] for r in rs]))}
        return DayMetrics(
            date=date, n=n,
            success_rate=sum(r["success"] for r in recs) / n,
            hallucination_rate=sum(r["hallucination"] for r in recs) / n,
            avg_steps=float(np.mean([r["n_steps"] for r in recs])),
            p95_latency_ms=float(np.percentile(lat, 95)),
            cost_per_request=float(np.mean([r["cost_usd"] for r in recs])),
            tools_valid_rate=sum(bool(r.get("tools_valid", True)) for r in recs) / n,
            error_step_rate=sum(not r.get("no_error_steps", True) for r in recs) / n,
            empty_answer_rate=sum(not r.get("answer_nonempty", True) for r in recs) / n,
            avg_answer_words=float(np.mean([r["answer_words"] for r in recs])),
            tool_call_counts=[int(r["n_tool_calls"]) for r in recs],
            answer_lengths=[int(r["answer_words"]) for r in recs],
            per_tenant=per_tenant,
            failed_ids=[r["task_id"] for r in recs if not r["success"]],
            hallucinated_ids=[r["task_id"] for r in recs if r["hallucination"]],
            feedback_down_rate=sum(r["feedback"] == "down" for r in recs) / n,
        )


# --------------------------------------------------------------------------- drift statistics

def psi(baseline: Sequence[float], current: Sequence[float], bins: Sequence[float],
        alpha: float = 0.5, eps: float = 1e-9) -> float:
    """Population Stability Index between two samples over fixed bin edges (see module docstring).

    `alpha` is the additive smoothing per bin (0 = raw shares, 0.5 = Jeffreys). Values at or above
    the last edge fall into the last bin.
    """
    b = np.asarray(baseline, float)
    c = np.asarray(current, float)
    if b.size == 0 or c.size == 0:
        return 0.0
    edges = np.asarray(bins, float)
    pb, _ = np.histogram(np.minimum(b, edges[-1] - 1e-9), bins=edges)
    pc, _ = np.histogram(np.minimum(c, edges[-1] - 1e-9), bins=edges)
    k = len(edges) - 1
    p = np.maximum((pb + alpha) / (pb.sum() + alpha * k), eps)
    q = np.maximum((pc + alpha) / (pc.sum() + alpha * k), eps)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_expected_noise(n_baseline: int, n_current: int, n_bins: int) -> float:
    """E[PSI] under identical distributions ~ (k-1)(1/n_b + 1/n_c) (chi-square approximation)."""
    if n_baseline == 0 or n_current == 0:
        return 0.0
    return (n_bins - 1) * (1.0 / n_baseline + 1.0 / n_current)


def _gammainc_upper_reg(a: float, x: float) -> float:
    """Regularised upper incomplete gamma Q(a, x) (series / Lentz continued fraction, NR 6.2)."""
    if x <= 0:
        return 1.0
    if x < a + 1:
        term = total = 1.0 / a
        for n in range(1, 500):
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return max(0.0, 1.0 - total * math.exp(-x + a * math.log(x) - math.lgamma(a)))
    tiny = 1e-300
    b = x + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - math.lgamma(a)) * h)


def chi2_sf(x: float, df: int) -> float:
    """Chi-square survival function P(X > x) = Q(df/2, x/2)."""
    return _gammainc_upper_reg(df / 2.0, x / 2.0)


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exact sum (n is small: one day's sample)."""
    if k <= 0:
        return 1.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def cusum(values: Sequence[float], mu0: float, sigma0: float, k: float = 0.5, h: float = 4.0,
          direction: str = "down") -> tuple[list[float], int | None]:
    """Standardised one-sided CUSUM. Returns the S_t path and the first index where S_t > h."""
    sigma0 = max(sigma0, 1e-9)
    s, path, first = 0.0, [], None
    for i, x in enumerate(values):
        z = (x - mu0) / sigma0
        s = max(0.0, s + (-z if direction == "down" else z) - k)
        path.append(s)
        if first is None and s > h:
            first = i
    return path, first


# --------------------------------------------------------------------------- detector

@dataclass
class Alert:
    severity: str            # "warning" | "critical"
    metric: str
    baseline: float
    current: float
    evidence: str
    sample_trajectory_ids: list[str] = field(default_factory=list)
    date: str = ""
    detector: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Alert:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


DEFAULT_THRESHOLDS: dict[str, Any] = {
    "baseline_window_days": 7,
    "min_baseline_days": 3,
    "cusum": {"k": 0.5, "h": 3.0, "critical_h": 6.0, "min_sigma": 0.02},
    "zscore": {"warn": 3.0, "critical": 4.5},
    "psi": {"warn": 0.10, "critical": 0.25, "bias_correct": True, "max_p": 0.001,
            "tool_call_bins": [0, 1, 2, 3, 4, 100],
            "answer_length_bins": [0, 30, 60, 90, 130, 100000]},
    "hallucination": {"ratio": 2.0, "min_abs": 0.03, "critical_ratio": 3.0, "min_count": 3, "max_p": 0.005},
    "max_sample_ids": 5,
}


def load_thresholds(path: str | Path | None) -> dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_THRESHOLDS))
    if path:
        user = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


class DriftDetector:
    """Compare today's DayMetrics against a rolling baseline window of previous days."""

    def __init__(self, thresholds: dict[str, Any] | None = None) -> None:
        self.t = thresholds or load_thresholds(None)

    def baseline(self, history: Sequence[DayMetrics], today: DayMetrics) -> list[DayMetrics]:
        prev = sorted((m for m in history if m.date < today.date and m.n > 0), key=lambda m: m.date)
        return prev[-int(self.t["baseline_window_days"]):]

    def detect(self, history: Sequence[DayMetrics], today: DayMetrics) -> list[Alert]:
        base = self.baseline(history, today)
        if len(base) < int(self.t["min_baseline_days"]) or today.n == 0:
            return []
        alerts: list[Alert] = []
        ids = today.failed_ids[: self.t["max_sample_ids"]]
        rates = np.array([m.success_rate for m in base], float)
        mu0, sigma0 = float(rates.mean()), max(float(rates.std(ddof=1)) if len(rates) > 1 else 0.0,
                                               float(self.t["cusum"]["min_sigma"]))

        # (a) CUSUM on success rate (baseline days then today)
        c = self.t["cusum"]
        path, _ = cusum(list(rates) + [today.success_rate], mu0, sigma0, k=c["k"], h=c["h"])
        s_today = path[-1]
        if s_today > c["h"]:
            alerts.append(Alert("critical" if s_today > c["critical_h"] else "warning", "success_rate",
                                mu0, today.success_rate,
                                f"CUSUM S={s_today:.2f} > h={c['h']} (mu0={mu0:.3f}, sigma0={sigma0:.3f}, k={c['k']})",
                                ids, today.date, "cusum"))

        # (b) z-score + Wilson non-overlap on success rate
        zc = self.t["zscore"]
        n_base = sum(m.n for m in base)
        k_base = sum(m.successes for m in base)
        p_base = k_base / n_base if n_base else mu0
        se = math.sqrt(max(p_base * (1 - p_base), 1e-6) / today.n)
        z = (p_base - today.success_rate) / se
        b_lo, b_hi = wilson_ci(k_base, n_base)
        t_lo, t_hi = wilson_ci(today.successes, today.n)
        if z > zc["warn"] or t_hi < b_lo:
            sev = "critical" if z > zc["critical"] else "warning"
            alerts.append(Alert(sev, "success_rate", p_base, today.success_rate,
                                f"z={z:.2f} (warn>{zc['warn']}); Wilson today [{t_lo:.3f},{t_hi:.3f}] vs "
                                f"baseline [{b_lo:.3f},{b_hi:.3f}]" + (" non-overlapping" if t_hi < b_lo else ""),
                                ids, today.date, "zscore"))

        # (c) PSI on tool-call count and answer-length distributions
        pc = self.t["psi"]
        for metric, attr, bins in (("tool_call_count_dist", "tool_call_counts", pc["tool_call_bins"]),
                                   ("answer_length_dist", "answer_lengths", pc["answer_length_bins"])):
            base_vals = [v for m in base for v in getattr(m, attr)]
            cur_vals = getattr(today, attr)
            raw = psi(base_vals, cur_vals, bins)
            noise = psi_expected_noise(len(base_vals), len(cur_vals), len(bins) - 1) if pc.get("bias_correct", True) else 0.0
            val = max(0.0, raw - noise)
            n_b, n_c = len(base_vals), len(cur_vals)
            p_val = chi2_sf(raw * n_b * n_c / max(1, n_b + n_c), len(bins) - 2) if n_b and n_c else 1.0
            if val >= pc["warn"] and p_val <= pc.get("max_p", 1.0):
                bm, cm = (float(np.mean(base_vals)) if base_vals else 0.0), (float(np.mean(cur_vals)) if cur_vals else 0.0)
                alerts.append(Alert("critical" if val >= pc["critical"] else "warning", metric, bm, cm,
                                    f"PSI={val:.3f} (raw {raw:.3f} - expected noise {noise:.3f}, p={p_val:.2g}; "
                                    f"warn>={pc['warn']}, critical>={pc['critical']}); "
                                    f"baseline mean {bm:.1f} -> today {cm:.1f}",
                                    ids, today.date, "psi"))

        # (d) hallucination spike
        hc = self.t["hallucination"]
        n_h = sum(m.hallucinations for m in base)
        h_base = n_h / n_base if n_base else 0.0
        h_today = today.hallucination_rate
        p_tail = binom_sf(today.hallucinations, today.n, max(h_base, 1e-9))
        if (h_today - h_base >= hc["min_abs"] and h_today >= hc["ratio"] * max(h_base, 1e-9)
                and today.hallucinations >= hc.get("min_count", 1) and p_tail <= hc.get("max_p", 1.0)):
            sev = "critical" if h_today >= hc["critical_ratio"] * max(h_base, 1e-9) else "warning"
            alerts.append(Alert(sev, "hallucination_rate", h_base, h_today,
                                f"{h_today:.3f} vs baseline {h_base:.3f} (ratio {h_today / max(h_base, 1e-9):.1f}x, "
                                f"min_abs {hc['min_abs']}, binomial tail p={p_tail:.2g})",
                                today.hallucinated_ids[: self.t["max_sample_ids"]], today.date, "hallucination_spike"))
        return alerts
