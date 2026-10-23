"""Gate behaviour on the demo variants: pass / warn / block / hysteresis / flaky exclusion."""
from __future__ import annotations

import json

from projects.p04_ci_regression_gate.core import COMMENT_MARKER, Gate, GateState, detect_flaky, render_comment


def test_pr_good_passes(config, baseline, results):
    res = Gate(config).evaluate(baseline, results["pr_good"])
    assert not res.blocked
    assert res.n_paired == 300
    v = {x.metric: x for x in res.verdicts}
    assert v["task_success"].delta > 0 and v["task_success"].status == "pass"


def test_pr_regressed_blocks_and_lists_failing_ids(config, baseline, results):
    res = Gate(config).evaluate(baseline, results["pr_regressed"])
    assert res.blocked
    v = {x.metric: x for x in res.verdicts}
    assert v["task_success"].status == "block"
    assert v["task_success"].delta < -4.5            # -5 points on 300 items
    assert v["task_success"].ci[1] < 0                # CI excludes zero
    assert len(res.failing) == config.gate["top_n_failures"]
    assert all(f["regression"] for f in res.failing)  # regressions are listed first
    comment = render_comment(res, baseline, results["pr_regressed"])
    assert "BLOCKED" in comment and COMMENT_MARKER in comment
    for f in res.failing:
        assert f["id"] in comment and f["expected"] in comment


def test_one_point_drop_warns_but_passes(config, baseline, results):
    res = Gate(config).evaluate(baseline, results["pr_minor"])
    assert not res.blocked
    v = {x.metric: x for x in res.verdicts}["task_success"]
    assert -1.5 < v.delta < -0.5
    assert v.status in {"pass", "warn"} and not v.breach
    assert "not significant" in v.note or "within" in v.note


def test_drop_over_threshold_but_not_significant_warns(config, baseline, results):
    """Shrink the sample to 40 items: a 5-point drop is no longer significant -> warn, not block."""
    ids = {r["id"] for r in baseline["examples"]}
    keep = sorted(ids)[:40]
    small_b = {**baseline, "examples": [r for r in baseline["examples"] if r["id"] in keep]}
    cur = results["pr_regressed"]
    small_c = {**cur, "examples": [r for r in cur["examples"] if r["id"] in keep]}
    res = Gate(config).evaluate(small_b, small_c)
    v = {x.metric: x for x in res.verdicts}["task_success"]
    assert v.delta < 0
    if v.delta < -config.gate["max_drop"]:
        assert v.status == "warn" and "not significant" in v.note and not res.blocked


def test_pr_slow_blocks_only_after_two_consecutive_runs(config, baseline, results, tmp_path):
    state_path = tmp_path / "gate_state.json"
    state = GateState(state_path)
    r1 = Gate(config, state).evaluate(baseline, results["pr_slow"])
    state.save()
    v1 = {x.metric: x for x in r1.verdicts}["latency_p95"]
    assert v1.breach and not r1.blocked and v1.status == "warn" and v1.consecutive == 1
    state2 = GateState(state_path)
    r2 = Gate(config, state2).evaluate(baseline, results["pr_slow"])
    state2.save()
    v2 = {x.metric: x for x in r2.verdicts}["latency_p95"]
    assert r2.blocked and v2.status == "block" and v2.consecutive == 2
    # recovery: one good run does not clear a k=2 metric, two do
    state3 = GateState(state_path)
    r3 = Gate(config, state3).evaluate(baseline, results["pr_good"])
    state3.save()
    assert r3.blocked and json.loads(state_path.read_text())["metrics"]["latency_p95"]["consecutive_recoveries"] == 1
    state4 = GateState(state_path)
    r4 = Gate(config, state4).evaluate(baseline, results["pr_good"])
    assert not r4.blocked


def test_flaky_items_detected_and_excluded(config, baseline, results):
    flaky = detect_flaky(results["pr_flaky"], config.gate["flake_threshold"])
    assert len(flaky) == 15
    res = Gate(config).evaluate(baseline, results["pr_flaky"])
    assert not res.blocked
    assert res.n_paired == 300 - 15
    assert {f["id"] for f in res.flaky} == set(flaky)
    assert not any(f["id"] in flaky for f in res.failing)
    comment = render_comment(res, baseline, results["pr_flaky"])
    assert "Flaky items excluded" in comment


def test_state_file_round_trip(tmp_path):
    p = tmp_path / "s.json"
    st = GateState(p)
    assert st.update("m", True, 2) == {"consecutive_breaches": 1, "consecutive_recoveries": 0, "blocking": False}
    assert st.update("m", True, 2)["blocking"] is True
    st.save()
    st2 = GateState(p)
    assert st2.update("m", False, 2)["blocking"] is True
    assert st2.update("m", False, 2)["blocking"] is False
