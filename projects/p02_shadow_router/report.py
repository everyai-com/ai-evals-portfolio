"""Cost/quality report over a shadow store.

Statistics
----------
* agreement rate      = judge-agree pairs / pairs with a shadow result, Wilson 95% CI
                        (evalkit.metrics.wilson_ci).
* cost / latency delta = paired bootstrap over per-request (shadow - primary) values
                        (evalkit.stats.paired_bootstrap; CI = 2.5/97.5 percentiles of resampled means).
* p50 / p95 latency   = numpy percentiles of the observed latencies per arm.
* error rate          = shadow errors (exceptions + timeouts) / shadowed requests, Wilson CI.
* projected monthly   = requests_per_day * 30 * mean cost per request, where requests_per_day is
                        the observed request count divided by the observed time span in days
                        (span is floored at one day so a short burst is not extrapolated upward).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evalkit.metrics import wilson_ci
from evalkit.report import md_table, pct
from evalkit.stats import paired_bootstrap
from projects.p02_shadow_router.core import DiffResult, Judge, TrajectoryDiff

DEFAULT_THRESHOLDS: dict[str, Any] = {
    "min_pairs": 30,
    "promote": {"min_agreement_lo": 0.90, "max_error_rate": 0.02, "max_shadow_worse_rate": 0.05,
                "max_cost_ratio": 1.0, "max_p95_latency_ratio": 1.5},
    "reject": {"max_agreement_hi": 0.80, "min_error_rate": 0.10, "min_shadow_worse_rate": 0.15,
               "min_cost_ratio": 10.0},
}


def load_thresholds(path: str | Path | None) -> dict[str, Any]:
    """Load YAML thresholds, layered over DEFAULT_THRESHOLDS."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_THRESHOLDS.items()}
    if path:
        user = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def recommend(summary: dict[str, Any], thresholds: dict[str, Any]) -> tuple[str, list[str]]:
    """promote / hold / reject from the summary numbers.

    reject  if any reject-rule fires (quality clearly worse, too many errors, cost out of range)
    promote if there are enough pairs and every promote-rule holds
    hold    otherwise (not enough evidence yet, or mixed signals)
    """
    rej, pro = thresholds["reject"], thresholds["promote"]
    reasons: list[str] = []
    n = summary["n_pairs"]
    agree_lo, agree_hi = summary["agreement_ci"]
    err = summary["error_rate"]
    worse = summary["shadow_worse_rate"]
    cost_ratio = summary["cost_ratio"]
    p95_ratio = summary["latency_p95_ratio"]

    if n >= 1 and agree_hi < rej["max_agreement_hi"]:
        reasons.append(f"agreement upper CI {agree_hi:.2f} < {rej['max_agreement_hi']}")
    if summary["n_shadowed"] >= 1 and err >= rej["min_error_rate"]:
        reasons.append(f"shadow error rate {err:.2%} >= {rej['min_error_rate']:.0%}")
    if n >= 1 and worse >= rej["min_shadow_worse_rate"]:
        reasons.append(f"shadow_worse rate {worse:.2%} >= {rej['min_shadow_worse_rate']:.0%}")
    if n >= 1 and cost_ratio >= rej["min_cost_ratio"]:
        reasons.append(f"cost ratio {cost_ratio:.1f}x >= {rej['min_cost_ratio']}x")
    if reasons:
        return "reject", reasons

    if n < thresholds["min_pairs"]:
        return "hold", [f"only {n} comparable pairs < min_pairs {thresholds['min_pairs']}"]
    checks = [
        (agree_lo >= pro["min_agreement_lo"], f"agreement lower CI {agree_lo:.2f} < {pro['min_agreement_lo']}"),
        (err <= pro["max_error_rate"], f"shadow error rate {err:.2%} > {pro['max_error_rate']:.0%}"),
        (worse <= pro["max_shadow_worse_rate"], f"shadow_worse rate {worse:.2%} > {pro['max_shadow_worse_rate']:.0%}"),
        (cost_ratio <= pro["max_cost_ratio"], f"cost ratio {cost_ratio:.2f}x > {pro['max_cost_ratio']}x"),
        (p95_ratio <= pro["max_p95_latency_ratio"], f"p95 latency ratio {p95_ratio:.2f}x > {pro['max_p95_latency_ratio']}x"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    if failed:
        return "hold", failed
    return "promote", ["all promote thresholds met"]


class ReportBuilder:
    def __init__(self, records: list[dict[str, Any]], thresholds: dict[str, Any] | None = None,
                 judge: Judge | None = None, n_boot: int = 2000, seed: int = 0) -> None:
        self.records = records
        self.thresholds = thresholds or load_thresholds(None)
        self.differ = TrajectoryDiff(judge)
        self.n_boot = n_boot
        self.seed = seed

    @classmethod
    def from_store(cls, store: Any, **kw: Any) -> "ReportBuilder":
        rows = store.read() if hasattr(store, "read") else __import__("evalkit.io", fromlist=["read_jsonl"]).read_jsonl(store)
        return cls(rows, **kw)

    # ------------------------------------------------------------------ numbers
    def build(self) -> dict[str, Any]:
        recs = self.records
        shadowed = [r for r in recs if r.get("shadowed")]
        diffs: list[DiffResult] = [self.differ.from_record(r) for r in shadowed]
        pairs = [(r, d) for r, d in zip(shadowed, diffs) if d.classification != "shadow_error"]
        n_total, n_shadowed, n_pairs = len(recs), len(shadowed), len(pairs)
        classes = Counter(d.classification for d in diffs)
        n_err = classes.get("shadow_error", 0)

        agree_k = sum(d.judge_agree for _, d in pairs)
        exact_k = sum(d.exact_match for _, d in pairs)
        norm_k = sum(d.normalized_match for _, d in pairs)
        agree_rate = agree_k / n_pairs if n_pairs else 0.0
        agree_ci = wilson_ci(agree_k, n_pairs)
        err_rate = n_err / n_shadowed if n_shadowed else 0.0
        err_ci = wilson_ci(n_err, n_shadowed)

        p_cost = np.array([r["primary"]["cost_usd"] for r, _ in pairs], float)
        s_cost = np.array([r["shadow"]["cost_usd"] for r, _ in pairs], float)
        p_lat = np.array([r["primary"]["latency_ms"] for r, _ in pairs], float)
        s_lat = np.array([r["shadow"]["latency_ms"] for r, _ in pairs], float)
        p_tok = np.array([r["primary"]["tokens"] for r, _ in pairs], float)
        s_tok = np.array([r["shadow"]["tokens"] for r, _ in pairs], float)

        def delta(a: np.ndarray, b: np.ndarray, name: str) -> dict[str, Any]:
            if a.size == 0:
                return {"metric": name, "n": 0, "a_mean": 0.0, "b_mean": 0.0, "delta": 0.0,
                        "ci_lo": 0.0, "ci_hi": 0.0, "p_value": 1.0, "significant": False}
            d = paired_bootstrap(a, b, metric=name, n_boot=self.n_boot, seed=self.seed).to_dict()
            return {k: d[k] for k in ("metric", "n", "a_mean", "b_mean", "delta", "ci_lo", "ci_hi",
                                      "p_value", "significant")}

        def pcts(x: np.ndarray) -> dict[str, float]:
            if x.size == 0:
                return {"p50": 0.0, "p95": 0.0, "mean": 0.0}
            return {"p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
                    "mean": float(x.mean())}

        all_primary_cost = np.array([r["primary"]["cost_usd"] for r in recs], float)
        ts = np.array([float(r.get("ts", 0.0)) for r in recs], float)
        span_days = max(1.0, (ts.max() - ts.min()) / 86400.0) if ts.size else 1.0
        req_per_day = n_total / span_days
        mean_p_cost_all = float(all_primary_cost.mean()) if all_primary_cost.size else 0.0
        mean_s_cost = float(s_cost.mean()) if s_cost.size else 0.0
        mean_p_cost_pairs = float(p_cost.mean()) if p_cost.size else 0.0
        cost_ratio = (mean_s_cost / mean_p_cost_pairs) if mean_p_cost_pairs > 0 else (float("inf") if mean_s_cost > 0 else 1.0)
        lat_p, lat_s = pcts(p_lat), pcts(s_lat)
        p95_ratio = lat_s["p95"] / lat_p["p95"] if lat_p["p95"] > 0 else 1.0

        summary: dict[str, Any] = {
            "n_requests": n_total, "n_shadowed": n_shadowed, "n_pairs": n_pairs,
            "observed_sample_rate": n_shadowed / n_total if n_total else 0.0,
            "primary_model": recs[0]["primary"]["model"] if recs else "",
            "shadow_model": next((r["shadow"]["model"] for r in shadowed if r.get("shadow")), ""),
            "agreement_rate": agree_rate, "agreement_ci": list(agree_ci),
            "exact_match_rate": exact_k / n_pairs if n_pairs else 0.0,
            "normalized_match_rate": norm_k / n_pairs if n_pairs else 0.0,
            "classes": {c: classes.get(c, 0) for c in ("same", "shadow_better", "shadow_worse",
                                                        "different_unknown", "shadow_error")},
            "shadow_better_rate": classes.get("shadow_better", 0) / n_pairs if n_pairs else 0.0,
            "shadow_worse_rate": classes.get("shadow_worse", 0) / n_pairs if n_pairs else 0.0,
            "error_rate": err_rate, "error_ci": list(err_ci),
            "cost": delta(p_cost, s_cost, "cost_usd"),
            "latency": delta(p_lat, s_lat, "latency_ms"),
            "tokens": delta(p_tok, s_tok, "tokens"),
            "latency_primary": lat_p, "latency_shadow": lat_s,
            "latency_p95_ratio": p95_ratio, "cost_ratio": cost_ratio,
            "mean_tool_edit_distance": float(np.mean([d.tool_edit_distance for _, d in pairs])) if pairs else 0.0,
            "traffic": {"span_days": span_days, "requests_per_day": req_per_day},
            "projected_monthly_cost": {
                "primary": req_per_day * 30 * mean_p_cost_all,
                "shadow": req_per_day * 30 * mean_s_cost,
            },
            "diffs": [d.to_dict() for d in diffs],
        }
        summary["projected_monthly_cost"]["delta"] = (summary["projected_monthly_cost"]["shadow"]
                                                      - summary["projected_monthly_cost"]["primary"])
        decision, reasons = recommend(summary, self.thresholds)
        summary["recommendation"] = decision
        summary["reasons"] = reasons
        return summary

    # ------------------------------------------------------------------ markdown
    def to_markdown(self, s: dict[str, Any] | None = None) -> str:
        s = s or self.build()
        c = s["classes"]
        lines = [
            f"# Shadow routing report: `{s['primary_model']}` (primary) vs `{s['shadow_model']}` (shadow)",
            "",
            f"**Recommendation: {s['recommendation'].upper()}** - " + "; ".join(s["reasons"]),
            "",
            "## Sample",
            md_table(["metric", "value"], [
                ["requests observed", s["n_requests"]],
                ["shadowed", f"{s['n_shadowed']} ({pct(s['observed_sample_rate'])})"],
                ["comparable pairs", s["n_pairs"]],
                ["shadow errors", f"{c['shadow_error']} ({pct(s['error_rate'])}, 95% CI {pct(s['error_ci'][0])}-{pct(s['error_ci'][1])})"],
                ["traffic span (days)", f"{s['traffic']['span_days']:.2f}"],
                ["requests / day", f"{s['traffic']['requests_per_day']:.1f}"],
            ]),
            "",
            "## Quality",
            md_table(["metric", "value"], [
                ["agreement (judge)", f"{pct(s['agreement_rate'])} (95% Wilson CI {pct(s['agreement_ci'][0])}-{pct(s['agreement_ci'][1])})"],
                ["exact match", pct(s["exact_match_rate"])],
                ["normalized match", pct(s["normalized_match_rate"])],
                ["same / better / worse / unknown / error",
                 f"{c['same']} / {c['shadow_better']} / {c['shadow_worse']} / {c['different_unknown']} / {c['shadow_error']}"],
                ["mean tool-sequence edit distance", f"{s['mean_tool_edit_distance']:.2f}"],
            ]),
            "",
            "## Cost and latency (paired, shadow - primary)",
            md_table(["metric", "primary mean", "shadow mean", "delta", "95% bootstrap CI", "p"], [
                ["cost / request (USD)", f"{s['cost']['a_mean']:.6f}", f"{s['cost']['b_mean']:.6f}",
                 f"{s['cost']['delta']:+.6f}", f"[{s['cost']['ci_lo']:+.6f}, {s['cost']['ci_hi']:+.6f}]", f"{s['cost']['p_value']:.3f}"],
                ["latency (ms)", f"{s['latency']['a_mean']:.1f}", f"{s['latency']['b_mean']:.1f}",
                 f"{s['latency']['delta']:+.1f}", f"[{s['latency']['ci_lo']:+.1f}, {s['latency']['ci_hi']:+.1f}]", f"{s['latency']['p_value']:.3f}"],
                ["tokens / request", f"{s['tokens']['a_mean']:.1f}", f"{s['tokens']['b_mean']:.1f}",
                 f"{s['tokens']['delta']:+.1f}", f"[{s['tokens']['ci_lo']:+.1f}, {s['tokens']['ci_hi']:+.1f}]", f"{s['tokens']['p_value']:.3f}"],
            ]),
            "",
            md_table(["latency", "p50 (ms)", "p95 (ms)"], [
                ["primary", f"{s['latency_primary']['p50']:.1f}", f"{s['latency_primary']['p95']:.1f}"],
                ["shadow", f"{s['latency_shadow']['p50']:.1f}", f"{s['latency_shadow']['p95']:.1f}"],
            ]),
            "",
            f"Cost ratio shadow/primary: **{s['cost_ratio']:.2f}x**; p95 latency ratio: **{s['latency_p95_ratio']:.2f}x**.",
            "",
            "## Projected monthly cost at 100% rollout",
            md_table(["arm", "USD / month"], [
                ["primary (today)", f"{s['projected_monthly_cost']['primary']:.2f}"],
                ["shadow (if promoted)", f"{s['projected_monthly_cost']['shadow']:.2f}"],
                ["delta", f"{s['projected_monthly_cost']['delta']:+.2f}"],
            ]),
        ]
        return "\n".join(lines) + "\n"
