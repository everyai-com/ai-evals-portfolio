"""Pareto engine: per tenant x policy aggregates, non-dominated frontier, savings, price sensitivity.

Definitions
-----------
success rate      successes / n with Wilson 95% CI (evalkit.metrics.wilson_ci).
cost / task       mean of `cost_usd`; with `price_scale` != 1 the System-Two component of each
                  row's `cost_breakdown` is multiplied by the scale before averaging.
cost / success    total cost / successes (inf when there are no successes).
Pareto frontier   policies not dominated on (min cost, max success): P dominates Q when
                  cost_P <= cost_Q and success_P >= success_Q with at least one strict. Computed
                  by sorting on (cost asc, success desc) and keeping points whose success beats
                  every cheaper point.
distance          Euclidean distance from a dominated point to its nearest frontier vertex in
                  (log10 cost, success) space; 0 on the frontier.
savings           For a dominated policy, the cheapest frontier policy with success >= its own:
                  savings = cost_P - cost_target per task (>= 0), plus the success gain.
marginal cost     Between adjacent frontier points (by cost): delta cost per +1 success point
                  (= delta cost / (100 * delta success)).
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from evalkit.metrics import wilson_ci

SYSTEM_TWO = "mock-system-two"
GLOBAL = "__global__"


@dataclass
class PolicyStats:
    tenant: str
    policy: str
    model: str
    n: int
    successes: int
    success_rate: float
    ci_lo: float
    ci_hi: float
    mean_cost: float
    mean_cost_one: float          # cost not affected by the price scale
    mean_cost_two: float          # System-Two cost at scale 1.0
    p95_latency_ms: float
    mean_latency_ms: float
    cost_per_success: float
    mean_tokens: float
    escalation_rate: float
    on_frontier: bool = False
    distance_to_frontier: float = 0.0
    target_policy: str | None = None
    savings_per_task: float = 0.0
    savings_pct: float = 0.0
    success_gain: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["cost_per_success"] == float("inf"):
            d["cost_per_success"] = None
        return d


def scaled_cost(row: dict[str, Any], price_scale: float, system_two: str = SYSTEM_TWO) -> float:
    """Row cost with the System-Two component multiplied by `price_scale`."""
    bd = row.get("cost_breakdown")
    if bd is not None:
        return sum(v * (price_scale if k == system_two else 1.0) for k, v in bd.items())
    full = system_two in str(row.get("model_or_cascade", "")) and "->" not in str(row.get("model_or_cascade", ""))
    return float(row["cost_usd"]) * (price_scale if full else 1.0)


def _split_cost(row: dict[str, Any], system_two: str) -> tuple[float, float]:
    bd = row.get("cost_breakdown")
    if bd is not None:
        two = float(bd.get(system_two, 0.0))
        return float(sum(bd.values())) - two, two
    c = float(row["cost_usd"])
    model = str(row.get("model_or_cascade", ""))
    return (0.0, c) if system_two in model and "->" not in model else (c, 0.0)


def _tokens(row: dict[str, Any]) -> float:
    t = row.get("tokens", 0)
    return float(sum(t.values())) if isinstance(t, dict) else float(t)


def _stats(tenant: str, policy: str, rows: Sequence[dict[str, Any]], price_scale: float, system_two: str) -> PolicyStats:
    n = len(rows)
    k = sum(bool(r["success"]) for r in rows)
    lo, hi = wilson_ci(k, n)
    split = [_split_cost(r, system_two) for r in rows]
    one = float(np.mean([a for a, _ in split]))
    two = float(np.mean([b for _, b in split]))
    mean_cost = one + two * price_scale
    lat = np.array([float(r["latency_ms"]) for r in rows])
    return PolicyStats(
        tenant=tenant, policy=policy, model=str(rows[0].get("model_or_cascade", "")), n=n, successes=k,
        success_rate=k / n, ci_lo=lo, ci_hi=hi, mean_cost=mean_cost, mean_cost_one=one, mean_cost_two=two,
        p95_latency_ms=float(np.percentile(lat, 95)), mean_latency_ms=float(lat.mean()),
        cost_per_success=(mean_cost * n / k) if k else float("inf"),
        mean_tokens=float(np.mean([_tokens(r) for r in rows])),
        escalation_rate=sum(bool(r.get("escalated", False)) for r in rows) / n,
    )


def pareto_frontier(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of non-dominated points on (min cost, max success), ordered by increasing cost."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], -points[i][1]))
    best = -float("inf")
    out: list[int] = []
    for i in order:
        if points[i][1] > best:
            out.append(i)
            best = points[i][1]
    return out


def annotate_frontier(stats: list[PolicyStats]) -> list[PolicyStats]:
    """Mark frontier membership and fill distance / savings fields in place."""
    pts = [(s.mean_cost, s.success_rate) for s in stats]
    front = pareto_frontier(pts)
    fset = set(front)
    for i, s in enumerate(stats):
        s.on_frontier = i in fset
        if s.on_frontier:
            s.distance_to_frontier, s.target_policy, s.savings_per_task, s.savings_pct, s.success_gain = 0.0, None, 0.0, 0.0, 0.0
            continue
        s.distance_to_frontier = min(
            float(np.hypot(np.log10(max(s.mean_cost, 1e-12)) - np.log10(max(stats[j].mean_cost, 1e-12)),
                           s.success_rate - stats[j].success_rate)) for j in front)
        cands = [stats[j] for j in front if stats[j].success_rate >= s.success_rate]
        target = min(cands, key=lambda t: t.mean_cost)  # always non-empty: a dominator is on the frontier
        s.target_policy = target.policy
        s.savings_per_task = s.mean_cost - target.mean_cost
        s.savings_pct = s.savings_per_task / s.mean_cost if s.mean_cost > 0 else 0.0
        s.success_gain = target.success_rate - s.success_rate
    return stats


def marginal_costs(stats: Sequence[PolicyStats]) -> list[dict[str, Any]]:
    """Marginal cost per +1 success point between adjacent frontier policies (cheapest first)."""
    front = sorted((s for s in stats if s.on_frontier), key=lambda s: s.mean_cost)
    out = []
    for a, b in itertools.pairwise(front):
        d_succ = (b.success_rate - a.success_rate) * 100
        out.append({"from": a.policy, "to": b.policy, "delta_cost": b.mean_cost - a.mean_cost,
                    "delta_success_pts": d_succ,
                    "cost_per_success_point": (b.mean_cost - a.mean_cost) / d_succ if d_succ > 0 else None})
    return out


def aggregate(rows: Iterable[dict[str, Any]], price_scale: float = 1.0, system_two: str = SYSTEM_TWO,
              include_global: bool = True) -> dict[str, list[PolicyStats]]:
    """{tenant: [PolicyStats...]} (+ GLOBAL pooled over tenants), frontier-annotated."""
    by: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[str(r["tenant"])][str(r["routing_policy"])].append(r)
        if include_global:
            by[GLOBAL][str(r["routing_policy"])].append(r)
    out: dict[str, list[PolicyStats]] = {}
    for tenant in sorted(by, key=lambda t: (t == GLOBAL, t)):
        stats = [_stats(tenant, pol, rs, price_scale, system_two) for pol, rs in sorted(by[tenant].items())]
        out[tenant] = annotate_frontier(stats)
    return out


def price_sensitivity(rows: Sequence[dict[str, Any]], scales: Sequence[float] = (1.0, 0.5, 0.25),
                      system_two: str = SYSTEM_TWO) -> dict[str, dict[str, list[str]]]:
    """{scale: {tenant: [frontier policies cheapest-first]}} as System-Two price is scaled."""
    out: dict[str, dict[str, list[str]]] = {}
    for sc in scales:
        agg = aggregate(rows, price_scale=sc, system_two=system_two)
        out[f"{sc:g}"] = {t: [s.policy for s in sorted(st, key=lambda s: s.mean_cost) if s.on_frontier]
                          for t, st in agg.items()}
    return out


def build_metrics(rows: Sequence[dict[str, Any]], price_scale: float = 1.0, system_two: str = SYSTEM_TWO,
                  sensitivity_scales: Sequence[float] = (1.0, 0.5, 0.25)) -> dict[str, Any]:
    agg = aggregate(rows, price_scale=price_scale, system_two=system_two)
    tenants: dict[str, Any] = {}
    for t, stats in agg.items():
        front = [s.policy for s in sorted(stats, key=lambda s: s.mean_cost) if s.on_frontier]
        tenants[t] = {"policies": [s.to_dict() for s in stats], "frontier": front,
                      "marginal": marginal_costs(stats), "n_tasks": sum(s.n for s in stats) // max(1, len(stats))}
    return {"price_scale": price_scale, "system_two": system_two, "n_rows": len(rows),
            "tenants": tenants, "tenant_order": [t for t in agg if t != GLOBAL] + [GLOBAL],
            "sensitivity": price_sensitivity(rows, sensitivity_scales, system_two)}
