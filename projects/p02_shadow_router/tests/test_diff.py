from evalkit.trajectory import Trajectory
from projects.p02_shadow_router.core import (
    TrajectoryDiff,
    containment_match,
    edit_distance,
    normalize,
)


def traj(answer, tools=(), success=None, cost=0.0, lat=0.0, tokens=0):
    t = Trajectory("t", model="m")
    for name in tools:
        t.add_tool(name, {}, cost_usd=cost / max(1, len(tools)), latency_ms=lat / max(1, len(tools)),
                   input_tokens=tokens // max(1, len(tools)))
    if not tools:
        t.add_llm("", cost_usd=cost, latency_ms=lat, input_tokens=tokens)
    return t.finish(answer, success)


def test_edit_distance():
    assert edit_distance([], []) == 0
    assert edit_distance(["a", "b"], ["a", "b"]) == 0
    assert edit_distance(["a", "b", "c"], ["a", "c"]) == 1
    assert edit_distance(["a"], ["b", "c"]) == 2
    assert edit_distance(["search", "calc", "answer"], ["calc", "search", "answer"]) == 2


def test_normalize_and_default_judge():
    assert normalize("  Paris. ") == "paris" and normalize("POSITIVE") == normalize("positive")
    d = TrajectoryDiff()
    r = d.compare(traj("Paris"), traj("paris."))
    assert r.classification == "same" and r.normalized_match and not r.exact_match
    assert containment_match("", "Paris", "The capital is Paris.")


def test_classifications():
    d = TrajectoryDiff()
    assert d.compare(traj("a", success=False), traj("b", success=True)).classification == "shadow_better"
    assert d.compare(traj("a", success=True), traj("b", success=False)).classification == "shadow_worse"
    assert d.compare(traj("a", success=True), traj("b", success=True)).classification == "same"
    assert d.compare(traj("a"), traj("b")).classification == "different_unknown"
    assert d.compare(traj("a", success=False), traj("b", success=False)).classification == "different_unknown"
    r = d.compare(traj("a"), None, shadow_error="RuntimeError: down")
    assert r.classification == "shadow_error" and r.shadow_error == "RuntimeError: down"
    r = d.compare(traj("a"), traj("a"), shadow_error="timeout")
    assert r.classification == "shadow_error"


def test_pluggable_judge_bool_and_label():
    lenient = TrajectoryDiff(judge=lambda p, a, b: a[:1] == b[:1])
    assert lenient.compare(traj("apple"), traj("avocado")).classification == "same"
    labeller = TrajectoryDiff(judge=lambda p, a, b: "shadow_better")
    assert labeller.compare(traj("x"), traj("y")).classification == "shadow_better"


def test_deltas_and_tools():
    d = TrajectoryDiff()
    p = traj("x", tools=("search", "calc"), cost=0.002, lat=100, tokens=200)
    s = traj("x", tools=("search",), cost=0.001, lat=40, tokens=100)
    r = d.compare(p, s)
    assert r.tool_edit_distance == 1 and r.primary_tools == ["search", "calc"]
    assert abs(r.cost_delta + 0.001) < 1e-12 and abs(r.latency_delta_ms + 60) < 1e-9 and r.token_delta == -100


def test_from_record_uses_expected_for_success_labels():
    d = TrajectoryDiff()
    rec = {"request_id": "r", "kind": "completion", "prompt": "sum?", "meta": {"expected": "42"},
           "primary": {"text": "41", "model": "p", "tokens": 2, "latency_ms": 1, "cost_usd": 0.0},
           "shadow": {"text": "42", "model": "s", "tokens": 2, "latency_ms": 1, "cost_usd": 0.0},
           "shadow_error": None}
    r = d.from_record(rec)
    assert r.classification == "shadow_better" and r.primary_success is False and r.shadow_success is True
