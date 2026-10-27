"""Engine for the CI regression gate: EvalRunner (+ content-addressed cache, cost budget), Gate
(paired bootstrap, latency/cost thresholds, hysteresis, flaky-item exclusion) and the PR-comment
renderer. Pure functions / small classes; file I/O only where the object is explicitly a store.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evalkit.io import read_jsonl, stable_hash
from evalkit.report import md_table
from evalkit.stats import paired_bootstrap

COMMENT_MARKER = "<!-- eval-gate-comment -->"


# ----------------------------------------------------------------------------- config

@dataclass
class EvalConfig:
    name: str
    datasets: list[Path]
    model: str
    params: dict[str, Any]
    seeds: list[int]
    metrics: list[str]
    max_cost_usd: float
    est_per_call_usd: float
    gate: dict[str, Any]
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> EvalConfig:
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        base = path.parent
        gate = {"max_drop": 2.0, "max_p95_increase_pct": 25.0, "max_cost_increase_pct": 30.0,
                "hysteresis": {"task_success": 1, "latency_p95": 2, "cost_per_task": 2},
                "flake_threshold": 0.0, "top_n_failures": 10, "n_boot": 5000, "alpha": 0.05}
        gate.update(raw.get("gate", {}))
        cost = raw.get("cost", {})
        return cls(name=raw.get("name", path.stem),
                   datasets=[(base / d["path"]) if isinstance(d, dict) else (base / d)
                             for d in raw["datasets"]],
                   model=raw.get("model", "mock"), params=raw.get("params", {}),
                   seeds=[int(s) for s in raw.get("seeds", [0])],
                   metrics=raw.get("metrics", ["task_success"]),
                   max_cost_usd=float(cost.get("max_cost_usd", float("inf"))),
                   est_per_call_usd=float(cost.get("est_per_call_usd", 0.0)),
                   gate=gate, path=path)

    def load_examples(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for d in self.datasets:
            rows.extend(read_jsonl(d))
        return rows


# ----------------------------------------------------------------------------- cache

class EvalCache:
    """Content-addressed cache: key = sha256(prompt, model, params, dataset_row) -> result JSON.

    One file per key under `root` so `actions/cache` can restore it between CI runs. `params`
    must include everything that changes the SUT's behaviour (the variant, the seed) or a PR
    would silently reuse main's answers.
    """

    def __init__(self, root: str | Path | None):
        self.root = Path(root) if root else None
        self.hits = 0
        self.misses = 0
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(prompt: str, model: str, params: dict[str, Any], row: dict[str, Any]) -> str:
        return stable_hash({"prompt": prompt, "model": model, "params": params, "row": row}, 40)

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.root:
            self.misses += 1
            return None
        p = self.root / f"{key}.json"
        if p.exists():
            self.hits += 1
            return json.loads(p.read_text())
        self.misses += 1
        return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        if self.root:
            (self.root / f"{key}.json").write_text(json.dumps(value, sort_keys=True))

    def stats(self) -> dict[str, Any]:
        n = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses, "hit_rate": self.hits / n if n else 0.0,
                "dir": str(self.root) if self.root else None}


class BudgetExceeded(RuntimeError):
    """Raised when projected or actual spend exceeds the configured budget (CLI exit 2)."""


# ----------------------------------------------------------------------------- runner

def _default_grader(example: dict[str, Any], output: str) -> bool:
    return str(output).strip().lower() == str(example.get("expected", "")).strip().lower()


class EvalRunner:
    """Run every (example, seed) through the SUT, with caching and a cost budget.

    Produces the results dict written to `results.json`:
    ``{"meta": ..., "examples": [{id, seed, success, latency_ms, cost_usd, tags, input,
    expected, actual}], "aggregates": {...}, "cache": {...}}``.
    """

    def __init__(self, config: EvalConfig, sut: Callable[[dict[str, Any], int], Any],
                 cache: EvalCache | None = None, grader=None, variant: str = "main"):
        self.config = config
        self.sut = sut
        self.cache = cache or EvalCache(None)
        self.grader = grader or _default_grader
        self.variant = variant

    def _params(self, seed: int) -> dict[str, Any]:
        return {**self.config.params, "variant": self.variant, "seed": seed}

    def project_cost(self, examples: list[dict[str, Any]]) -> tuple[int, float]:
        """(cache misses, projected USD) before running anything."""
        misses = 0
        for ex in examples:
            for seed in self.config.seeds:
                key = EvalCache.key(ex["input"], self.config.model, self._params(seed), ex)
                if not (self.cache.root and (self.cache.root / f"{key}.json").exists()):
                    misses += 1
        return misses, misses * self.config.est_per_call_usd

    def run(self, examples: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        examples = self.config.load_examples() if examples is None else examples
        misses, projected = self.project_cost(examples)
        if projected > self.config.max_cost_usd:
            raise BudgetExceeded(f"projected spend ${projected:.4f} for {misses} uncached calls exceeds "
                                 f"max_cost_usd=${self.config.max_cost_usd:.4f}")
        rows: list[dict[str, Any]] = []
        spent = 0.0
        for ex in examples:
            for seed in self.config.seeds:
                params = self._params(seed)
                key = EvalCache.key(ex["input"], self.config.model, params, ex)
                rec = self.cache.get(key)
                if rec is None:
                    t0 = time.perf_counter()
                    out = self.sut(ex, seed)
                    wall = (time.perf_counter() - t0) * 1000
                    text = getattr(out, "output", out)
                    latency = getattr(out, "latency_ms", None)
                    rec = {"id": ex["id"], "seed": seed, "success": bool(self.grader(ex, text)),
                           "latency_ms": float(wall if latency is None else latency),
                           "cost_usd": float(getattr(out, "cost_usd", 0.0)),
                           "tags": list(ex.get("tags", [])), "input": ex["input"],
                           "expected": ex.get("expected"), "actual": str(text), "cached": False}
                    self.cache.put(key, rec)
                    spent += rec["cost_usd"]
                    if spent > self.config.max_cost_usd:
                        raise BudgetExceeded(f"actual spend ${spent:.4f} exceeded max_cost_usd="
                                             f"${self.config.max_cost_usd:.4f} after {len(rows) + 1} calls")
                else:
                    rec = {**rec, "cached": True}
                rows.append(rec)
        return {"meta": {"name": self.config.name, "variant": self.variant, "model": self.config.model,
                         "seeds": self.config.seeds, "n_examples": len(examples),
                         "config_hash": stable_hash({"model": self.config.model, "params": self.config.params,
                                                     "seeds": self.config.seeds})},
                "examples": rows, "aggregates": aggregate(rows), "cache": self.cache.stats()}


def per_item_success(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    by_id: dict[str, list[float]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["id"], r["seed"])):
        by_id[r["id"]].append(1.0 if r["success"] else 0.0)
    return dict(by_id)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """task_success = mean over items of mean-over-seeds success; p50/p95 latency over all calls;
    cost_per_task = total cost / number of distinct items."""
    if not rows:
        return {"task_success": 0.0, "n_items": 0, "n_calls": 0, "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0, "cost_total_usd": 0.0, "cost_per_task": 0.0, "by_tag": {}}
    items = per_item_success(rows)
    lat = np.array([r["latency_ms"] for r in rows])
    cost = float(sum(r["cost_usd"] for r in rows))
    by_tag: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        for t in r.get("tags") or ["untagged"]:
            by_tag[t].append(1.0 if r["success"] else 0.0)
    return {"task_success": float(np.mean([np.mean(v) for v in items.values()])),
            "n_items": len(items), "n_calls": len(rows),
            "latency_p50_ms": float(np.percentile(lat, 50)), "latency_p95_ms": float(np.percentile(lat, 95)),
            "cost_total_usd": cost, "cost_per_task": cost / len(items),
            "by_tag": {t: {"n": len(v), "success": float(np.mean(v))} for t, v in sorted(by_tag.items())}}


# ----------------------------------------------------------------------------- gate

@dataclass
class MetricVerdict:
    metric: str
    baseline: float
    current: float
    delta: float                  # current - baseline (points for success, % for latency/cost)
    threshold: str
    breach: bool                  # raw threshold breached on this run
    blocking: bool                # after hysteresis
    status: str                   # "pass" | "warn" | "block"
    note: str = ""
    ci: tuple[float, float] | None = None
    p_value: float | None = None
    consecutive: int = 0
    k: int = 1


@dataclass
class GateResult:
    blocked: bool
    verdicts: list[MetricVerdict]
    flaky: list[dict[str, Any]]
    failing: list[dict[str, Any]]   # top-N failing cases in the PR run
    n_paired: int
    summary: str
    state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"blocked": self.blocked, "verdicts": [asdict(v) for v in self.verdicts],
                "flaky": self.flaky, "failing": self.failing, "n_paired": self.n_paired,
                "summary": self.summary}


class GateState:
    """Hysteresis state persisted as JSON: per metric, consecutive breach/recovery counts and
    whether the metric is currently in the blocking state.

    Transition rule (k = required consecutive runs for that metric):
      breach   -> breaches += 1, recoveries = 0; blocking becomes True once breaches >= k
      recovery -> recoveries += 1, breaches = 0; blocking becomes False once recoveries >= k
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self.data: dict[str, Any] = {"metrics": {}, "runs": []}
        if self.path and self.path.exists():
            self.data = json.loads(self.path.read_text())

    def update(self, metric: str, breach: bool, k: int) -> dict[str, Any]:
        m = self.data["metrics"].setdefault(metric, {"consecutive_breaches": 0, "consecutive_recoveries": 0,
                                                     "blocking": False})
        if breach:
            m["consecutive_breaches"] += 1
            m["consecutive_recoveries"] = 0
            if m["consecutive_breaches"] >= k:
                m["blocking"] = True
        else:
            m["consecutive_recoveries"] += 1
            m["consecutive_breaches"] = 0
            if m["consecutive_recoveries"] >= k:
                m["blocking"] = False
        return m

    def record_run(self, info: dict[str, Any]) -> None:
        self.data["runs"] = (self.data["runs"] + [info])[-50:]

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True))


def detect_flaky(results: dict[str, Any], threshold: float = 0.0) -> dict[str, float]:
    """Items whose per-seed success has population std > threshold (binary outcomes: any
    disagreement across seeds gives std >= 0.47 for 2 seeds). Returns {id: std}."""
    out = {}
    for cid, vals in per_item_success(results["examples"]).items():
        if len(vals) > 1:
            s = float(np.std(vals))
            if s > threshold:
                out[cid] = s
    return out


class Gate:
    """Compare a PR results dict to a baseline results dict and decide block / warn / pass."""

    def __init__(self, config: EvalConfig, state: GateState | None = None):
        self.cfg = config.gate
        self.state = state or GateState(None)

    def evaluate(self, baseline: dict[str, Any], current: dict[str, Any]) -> GateResult:
        g = self.cfg
        hyst = g.get("hysteresis", {})
        thr = float(g["flake_threshold"])
        flaky_b, flaky_c = detect_flaky(baseline, thr), detect_flaky(current, thr)
        flaky_ids = set(flaky_b) | set(flaky_c)
        base_items, cur_items = per_item_success(baseline["examples"]), per_item_success(current["examples"])
        paired_ids = sorted((set(base_items) & set(cur_items)) - flaky_ids)
        a = [float(np.mean(base_items[i])) for i in paired_ids]
        b = [float(np.mean(cur_items[i])) for i in paired_ids]
        verdicts: list[MetricVerdict] = []

        # (a) task success: paired bootstrap on the same items, flaky excluded
        rep = paired_bootstrap(a, b, metric="task_success", n_boot=int(g["n_boot"]),
                               alpha=float(g["alpha"]), seed=0) if paired_ids else None
        drop_pts = -rep.delta * 100 if rep else 0.0
        max_drop = float(g["max_drop"])
        ci = (rep.ci_lo * 100, rep.ci_hi * 100) if rep else None
        excludes_zero = bool(rep and (rep.ci_hi < 0 or rep.ci_lo > 0))
        breach = bool(rep and drop_pts > max_drop and rep.ci_hi < 0)
        if breach:
            note = f"drop of {drop_pts:.1f} pts > {max_drop:g} and significant (95% CI excludes 0, p={rep.p_value:.3g})"
            status = "block"
        elif rep and drop_pts > max_drop:
            note = f"drop of {drop_pts:.1f} pts exceeds {max_drop:g} but is not significant at this n (CI includes 0)"
            status = "warn"
        elif rep and drop_pts > 0:
            note = (f"drop of {drop_pts:.1f} pts is within the {max_drop:g}-pt budget"
                    + (" (significant)" if excludes_zero else " (not significant)"))
            status = "warn" if drop_pts > max_drop / 2 else "pass"
        else:
            note, status = "no regression", "pass"
        k = int(hyst.get("task_success", 1))
        st = self.state.update("task_success", breach, k)
        verdicts.append(MetricVerdict("task_success", rep.a_mean * 100 if rep else 0.0,
                                      rep.b_mean * 100 if rep else 0.0, -drop_pts, f"drop <= {max_drop:g} pts",
                                      breach, st["blocking"], "block" if st["blocking"] else status, note,
                                      ci=ci, p_value=rep.p_value if rep else None,
                                      consecutive=st["consecutive_breaches"], k=k))

        # (b) latency p95 and (c) cost per task: relative increase thresholds with hysteresis
        for metric, key, thr_key in (("latency_p95", "latency_p95_ms", "max_p95_increase_pct"),
                                     ("cost_per_task", "cost_per_task", "max_cost_increase_pct")):
            bv, cv = float(baseline["aggregates"][key]), float(current["aggregates"][key])
            inc = (cv - bv) / bv * 100 if bv else 0.0
            limit = float(g[thr_key])
            breach = inc > limit
            k = int(hyst.get(metric, 1))
            st = self.state.update(metric, breach, k)
            if st["blocking"]:
                status, note = "block", f"+{inc:.1f}% > {limit:g}% on {st['consecutive_breaches']} consecutive run(s)"
            elif breach:
                status, note = "warn", f"+{inc:.1f}% > {limit:g}% (breach {st['consecutive_breaches']}/{k}; blocks at {k})"
            else:
                status, note = "pass", "within threshold"
            verdicts.append(MetricVerdict(metric, bv, cv, inc, f"increase <= {limit:g}%", breach,
                                          st["blocking"], status, note, consecutive=st["consecutive_breaches"], k=k))

        blocked = any(v.blocking for v in verdicts)
        failing = top_failures(baseline, current, int(g["top_n_failures"]), flaky_ids)
        flaky = [{"id": i, "std": round(max(flaky_b.get(i, 0), flaky_c.get(i, 0)), 3),
                  "in": "both" if i in flaky_b and i in flaky_c else ("baseline" if i in flaky_b else "pr")}
                 for i in sorted(flaky_ids)]
        headline = "BLOCKED" if blocked else ("PASSED with warnings" if any(v.status == "warn" for v in verdicts) else "PASSED")
        self.state.record_run({"variant": current["meta"].get("variant"), "blocked": blocked,
                               "task_success": current["aggregates"]["task_success"]})
        return GateResult(blocked, verdicts, flaky, failing, len(paired_ids),
                          f"{headline}: " + "; ".join(f"{v.metric} {v.status}" for v in verdicts),
                          state=self.state.data)


def top_failures(baseline: dict[str, Any], current: dict[str, Any], n: int,
                 exclude: set[str] | None = None) -> list[dict[str, Any]]:
    """Failing PR cases, regressions (passed on baseline) first, then pre-existing failures."""
    exclude = exclude or set()
    base_ok = {i: np.mean(v) == 1.0 for i, v in per_item_success(baseline["examples"]).items()}
    seen: set[str] = set()
    out = []
    for r in current["examples"]:
        if r["success"] or r["id"] in seen or r["id"] in exclude:
            continue
        seen.add(r["id"])
        out.append({"id": r["id"], "input": r["input"], "expected": r["expected"], "actual": r["actual"],
                    "tags": r.get("tags", []), "regression": bool(base_ok.get(r["id"], False))})
    out.sort(key=lambda d: (not d["regression"], d["id"]))
    return out[:n]


# ----------------------------------------------------------------------------- comment

def _badge(result: GateResult) -> str:
    if result.blocked:
        return "❌ **BLOCKED** — merge gate failed"
    if any(v.status == "warn" for v in result.verdicts):
        return "⚠️ **PASSED with warnings**"
    return "✅ **PASSED**"


def _fmt_delta(v: MetricVerdict) -> str:
    if v.metric == "task_success":
        return f"{v.delta:+.1f} pts"
    return f"{v.delta:+.1f}%"


def _trunc(s: Any, n: int = 60) -> str:
    s = str(s).replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def render_comment(result: GateResult, baseline: dict[str, Any], current: dict[str, Any],
                   marker: str = COMMENT_MARKER) -> str:
    """Markdown PR comment: badge, delta table with CIs, failing cases, flaky items, cache, cost."""
    cur_meta, agg = current["meta"], current["aggregates"]
    rows = []
    for v in result.verdicts:
        if v.metric == "task_success":
            base, cur = f"{v.baseline:.1f}%", f"{v.current:.1f}%"
            ci = f"[{v.ci[0]:+.1f}, {v.ci[1]:+.1f}] pts" if v.ci else "n/a"
        elif v.metric == "latency_p95":
            base, cur, ci = f"{v.baseline:.0f} ms", f"{v.current:.0f} ms", "—"
        else:
            base, cur, ci = f"${v.baseline:.5f}", f"${v.current:.5f}", "—"
        icon = {"pass": "✅", "warn": "⚠️", "block": "❌"}[v.status]
        rows.append([v.metric, base, cur, _fmt_delta(v), ci, v.threshold, f"{icon} {v.status}", v.note])
    lines = [marker, f"## Eval gate: {_badge(result)}", "",
             (f"`{cur_meta.get('variant', '?')}` vs baseline `{baseline['meta'].get('variant', 'main')}` · "
             f"model `{cur_meta.get('model')}` · {result.n_paired} paired items · seeds {cur_meta.get('seeds')}"), "",
             md_table(["metric", "baseline", "PR", "Δ", "95% CI", "threshold", "status", "note"], rows), ""]
    if result.failing:
        n_reg = sum(1 for f in result.failing if f["regression"])
        lines += [f"### Top failing cases ({len(result.failing)} shown, {n_reg} new vs baseline)", "",
                  md_table(["id", "new?", "input", "expected", "actual", "tags"],
                           [[f["id"], "🔴 regression" if f["regression"] else "pre-existing", _trunc(f["input"]),
                             _trunc(f["expected"], 30), _trunc(f["actual"], 30), ",".join(f["tags"])]
                            for f in result.failing]), ""]
    else:
        lines += ["### Failing cases", "", "None 🎉", ""]
    if result.flaky:
        lines += [f"### Flaky items excluded from the decision ({len(result.flaky)})", "",
                  md_table(["id", "std across seeds", "flaky in"], [[f["id"], f["std"], f["in"]] for f in result.flaky]),
                  ""]
    cache = current.get("cache", {})
    lines += ["### Run stats", "",
              (f"- cache: {cache.get('hits', 0)} hits / {cache.get('misses', 0)} misses "
              f"(hit rate {cache.get('hit_rate', 0.0) * 100:.0f}%)"),
              (f"- cost: ${agg['cost_total_usd']:.4f} total, ${agg['cost_per_task']:.6f} per task "
              f"over {agg['n_calls']} calls"),
              f"- latency: p50 {agg['latency_p50_ms']:.0f} ms, p95 {agg['latency_p95_ms']:.0f} ms", "",
              ("<sub>Decision rule: block when task-success drop > max_drop *and* the paired-bootstrap 95% CI "
              "excludes zero; latency/cost block after k consecutive breaches (hysteresis).</sub>")]
    return "\n".join(lines) + "\n"


def write_github_outputs(result: GateResult, current: dict[str, Any], comment_path: str | None,
                         env: dict[str, str] | None = None) -> dict[str, str]:
    """Append key=value lines to $GITHUB_OUTPUT and the comment to $GITHUB_STEP_SUMMARY when set."""
    env = os.environ if env is None else env
    v = {x.metric: x for x in result.verdicts}
    outputs = {"blocked": str(result.blocked).lower(), "status": result.summary.split(":")[0],
               "task_success": f"{current['aggregates']['task_success']:.4f}",
               "task_success_delta_pts": f"{v['task_success'].delta:.2f}",
               "latency_p95_ms": f"{current['aggregates']['latency_p95_ms']:.1f}",
               "comment_path": comment_path or ""}
    if env.get("GITHUB_OUTPUT"):
        with open(env["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.writelines(f"{k}={val}\n" for k, val in outputs.items())
    if env.get("GITHUB_STEP_SUMMARY") and comment_path and Path(comment_path).exists():
        with open(env["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(Path(comment_path).read_text())
    return outputs
