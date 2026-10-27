import math

import pytest

from projects.p10_pareto_dashboard.core import (
    GLOBAL,
    PolicyStats,
    aggregate,
    annotate_frontier,
    build_metrics,
    marginal_costs,
    pareto_frontier,
    price_sensitivity,
    scaled_cost,
)


def test_pareto_frontier_hand_built_five_points():
    # (cost, success): A cheap/low, B dominated by C (same success, pricier), C, D dominated by C (pricier, worse), E best
    pts = [(1.0, 0.50), (3.0, 0.80), (2.0, 0.80), (4.0, 0.70), (8.0, 0.95)]
    assert pareto_frontier(pts) == [0, 2, 4]
    assert pareto_frontier([]) == []
    assert pareto_frontier([(1.0, 0.5)]) == [0]
    # equal cost, equal success: only the first survives; equal cost, different success: the better one
    assert pareto_frontier([(1.0, 0.5), (1.0, 0.5)]) == [0]
    assert pareto_frontier([(1.0, 0.5), (1.0, 0.9)]) == [1]


def ps(policy, cost, succ, n=100, two=0.0):
    k = int(round(succ * n))
    return PolicyStats("t", policy, policy, n, k, succ, succ - 0.05, succ + 0.05, cost, cost - two, two,
                       1000.0, 500.0, cost / succ if succ else float("inf"), 800.0, 0.0)


def test_savings_and_distance():
    stats = annotate_frontier([ps("A", 1.0, 0.50), ps("B", 3.0, 0.80), ps("C", 2.0, 0.80), ps("D", 4.0, 0.70), ps("E", 8.0, 0.95)])
    by = {s.policy: s for s in stats}
    assert [s.policy for s in stats if s.on_frontier] == ["A", "C", "E"]
    assert by["B"].target_policy == "C" and by["B"].savings_per_task == pytest.approx(1.0)
    assert by["B"].savings_pct == pytest.approx(1 / 3) and by["B"].success_gain == 0.0
    assert by["D"].target_policy == "C" and by["D"].savings_per_task == pytest.approx(2.0)
    assert by["D"].success_gain == pytest.approx(0.10)
    assert by["B"].distance_to_frontier == pytest.approx(math.log10(3.0) - math.log10(2.0))
    assert by["A"].distance_to_frontier == 0.0 and by["A"].target_policy is None
    m = marginal_costs(stats)
    assert [(x["from"], x["to"]) for x in m] == [("A", "C"), ("C", "E")]
    assert m[0]["delta_success_pts"] == pytest.approx(30.0) and m[0]["cost_per_success_point"] == pytest.approx(1.0 / 30)
    assert m[1]["delta_cost"] == pytest.approx(6.0)


def rows_for(policy, model, n_ok, n, one_cost, two_cost, tenant="t1"):
    return [{"tenant": tenant, "routing_policy": policy, "model_or_cascade": model, "task_id": f"{i}",
             "success": i < n_ok, "cost_usd": one_cost + two_cost, "latency_ms": 100.0,
             "tokens": {"input": 10, "output": 5},
             "cost_breakdown": {"mock-system-one": one_cost, "mock-system-two": two_cost}} for i in range(n)]


def test_price_scale_moves_policy_onto_frontier():
    rows = (rows_for("cheap", "mock-system-one", 60, 100, 0.001, 0.0)
            + rows_for("cascade", "mock-system-one->mock-system-two", 85, 100, 0.001, 0.004)
            + rows_for("expensive", "mock-system-two", 90, 100, 0.0, 0.010))
    at1 = {s.policy: s for s in aggregate(rows, 1.0)["t1"]}
    assert at1["expensive"].on_frontier  # best success, nothing dominates it even at list price
    # give the cascade higher success than expensive at list price -> expensive dominated
    rows2 = (rows_for("cheap", "mock-system-one", 60, 100, 0.001, 0.0)
             + rows_for("cascade", "mock-system-one->mock-system-two", 90, 100, 0.001, 0.004)
             + rows_for("expensive", "mock-system-two", 90, 100, 0.0, 0.010))
    a1 = {s.policy: s for s in aggregate(rows2, 1.0)["t1"]}
    assert not a1["expensive"].on_frontier and a1["expensive"].target_policy == "cascade"
    assert a1["expensive"].mean_cost == pytest.approx(0.010)
    # at 0.25x System-Two price: expensive = 0.0025 < cascade = 0.001 + 0.001 = 0.002? no -> still 0.0025 > 0.002
    a25 = {s.policy: s for s in aggregate(rows2, 0.25)["t1"]}
    assert a25["expensive"].mean_cost == pytest.approx(0.0025) and a25["cascade"].mean_cost == pytest.approx(0.002)
    assert not a25["expensive"].on_frontier
    # at 0.12x: expensive = 0.0012 < cascade 0.00148 -> expensive on the frontier, cascade falls off
    a12 = {s.policy: s for s in aggregate(rows2, 0.12)["t1"]}
    assert a12["expensive"].on_frontier and not a12["cascade"].on_frontier and a12["cheap"].on_frontier
    # at 0.1x: expensive costs exactly what cheap costs with higher success -> cheap is dominated too
    sens = price_sensitivity(rows2, scales=(1.0, 0.12, 0.1))
    assert sens["1"]["t1"] == ["cheap", "cascade"]
    assert sens["0.12"]["t1"] == ["cheap", "expensive"] and sens["0.1"]["t1"] == ["expensive"]
    assert scaled_cost(rows2[-1], 0.5) == pytest.approx(0.005)
    # rows without a breakdown: full System-Two rows scale, others do not
    assert scaled_cost({"cost_usd": 1.0, "model_or_cascade": "mock-system-two"}, 0.5) == 0.5
    assert scaled_cost({"cost_usd": 1.0, "model_or_cascade": "mock-system-one"}, 0.5) == 1.0


def test_aggregate_stats_and_global():
    rows = rows_for("p", "m", 30, 50, 0.001, 0.0, tenant="a") + rows_for("p", "m", 40, 50, 0.002, 0.0, tenant="b")
    agg = aggregate(rows)
    assert set(agg) == {"a", "b", GLOBAL}
    g = agg[GLOBAL][0]
    assert g.n == 100 and g.successes == 70 and g.success_rate == 0.7 and g.mean_cost == pytest.approx(0.0015)
    assert g.cost_per_success == pytest.approx(0.0015 * 100 / 70) and g.ci_lo < 0.7 < g.ci_hi
    assert g.mean_tokens == 15.0 and g.p95_latency_ms == 100.0
    m = build_metrics(rows, sensitivity_scales=(1.0,))
    assert m["tenant_order"] == ["a", "b", GLOBAL] and m["tenants"]["a"]["frontier"] == ["p"]
