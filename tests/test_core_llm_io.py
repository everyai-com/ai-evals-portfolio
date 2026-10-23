import json
from evalkit import MockLLM, Trajectory, read_jsonl, write_jsonl, stable_hash
from evalkit.llm import estimate_cost


def test_mock_is_deterministic_and_scripted():
    m = MockLLM(handlers=[(r"capital of france", "Paris"), (r"json", lambda s, p: json.dumps({"ok": 1}))])
    assert m.complete("What is the capital of France?").text == "Paris"
    assert m.complete("return json please").json() == {"ok": 1}
    a = m.complete("anything else").text
    b = MockLLM().complete("anything else").text
    assert a == b


def test_mock_failure_is_reproducible():
    import pytest
    m = MockLLM(fail_rate=1.0)
    with pytest.raises(RuntimeError):
        m.complete("x")


def test_cost_estimate():
    assert estimate_cost("claude-haiku-5-5", 1_000_000, 0) == 0.80


def test_trajectory_roundtrip(tmp_path):
    t = Trajectory("t1", model="mock").add_tool("search", {"q": "x"}, result=["a"]).add_llm("thinking").finish("done", True)
    t2 = Trajectory.from_json(t.to_json())
    assert t2.tool_names == ["search"] and t2.success is True and len(t2.steps) == 3
    p = tmp_path / "x.jsonl"
    write_jsonl(p, [t.to_dict()])
    assert read_jsonl(p)[0]["task_id"] == "t1"
    assert stable_hash(t.to_dict()) == stable_hash(t2.to_dict())


def test_mock_latency_not_double_counted():
    m = MockLLM(latency_ms=20)
    r = m.complete("x")
    assert 15 <= r.latency_ms < 60


def test_append_jsonl(tmp_path):
    from evalkit.io import append_jsonl
    p = tmp_path / "a.jsonl"
    append_jsonl(p, [{"a": 1}])
    append_jsonl(p, [{"a": 2}])
    assert [r["a"] for r in read_jsonl(p)] == [1, 2]
