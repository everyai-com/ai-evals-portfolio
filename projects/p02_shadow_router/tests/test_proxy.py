import time

import pytest

from evalkit.llm import MockLLM
from evalkit.trajectory import Trajectory
from projects.p02_shadow_router.core import (
    JsonlStore,
    MemoryStore,
    ShadowAgentRunner,
    ShadowProxy,
    should_shadow,
)


class Exploding:
    model = "boom"

    def complete(self, prompt, **kw):
        raise RuntimeError("shadow backend down")


class Slow:
    model = "slow"

    def complete(self, prompt, **kw):
        time.sleep(0.03)
        return MockLLM("slow").complete(prompt, **kw)


def test_caller_always_gets_primary_even_when_shadow_raises():
    primary = MockLLM("mock-system-one", handlers=[(r".", "PRIMARY")])
    store = MemoryStore()
    with ShadowProxy(primary, Exploding(), sample_rate=1.0, store=store) as proxy:
        for i in range(20):
            assert proxy.complete(f"q{i}", request_id=f"r{i}").text == "PRIMARY"
    rows = store.read()
    assert len(rows) == 20 and all(r["shadowed"] for r in rows)
    assert all(r["shadow"] is None and "shadow backend down" in r["shadow_error"] for r in rows)
    assert proxy.stats["shadow_errors"] == 20


def test_shadow_failures_never_surface_and_are_recorded_with_fail_rate():
    primary = MockLLM("mock-system-one")
    shadow = MockLLM("mock-system-two", fail_rate=0.5, seed=3)
    store = MemoryStore()
    with ShadowProxy(primary, shadow, sample_rate=1.0, store=store) as proxy:
        for i in range(60):
            proxy.complete(f"prompt {i}", request_id=f"r{i}")  # must not raise
    rows = store.read()
    errs = [r for r in rows if r["shadow_error"]]
    oks = [r for r in rows if r["shadow"]]
    assert len(rows) == 60 and 10 < len(errs) < 50 and len(oks) + len(errs) == 60
    assert all("simulated backend failure" in r["shadow_error"] for r in errs)


def test_shadow_timeout_is_logged_not_raised():
    store = MemoryStore()
    with ShadowProxy(MockLLM("p"), Slow(), sample_rate=1.0, store=store, shadow_timeout_ms=5) as proxy:
        proxy.complete("hello", request_id="x")
    (row,) = store.read()
    assert row["shadow_error"].startswith("timeout") and proxy.stats["shadow_timeouts"] == 1


def test_primary_errors_do_propagate():
    with ShadowProxy(Exploding(), MockLLM("s"), sample_rate=1.0) as proxy, pytest.raises(RuntimeError):
        proxy.complete("x", request_id="r")


def test_sampling_rate_within_tolerance_and_deterministic():
    ids = [f"req-{i}" for i in range(20_000)]
    picked = [i for i in ids if should_shadow(i, 0.05, seed=0)]
    assert 0.04 < len(picked) / len(ids) < 0.06
    assert picked == [i for i in ids if should_shadow(i, 0.05, seed=0)]
    assert picked != [i for i in ids if should_shadow(i, 0.05, seed=1)]
    assert all(should_shadow(i, 1.0) for i in ids[:100]) and not any(should_shadow(i, 0.0) for i in ids[:100])
    # monotone: anything sampled at 5% is also sampled at 10%
    assert all(should_shadow(i, 0.10) for i in picked)


def test_proxy_sampling_is_deterministic_across_instances():
    def run(seed):
        store = MemoryStore()
        with ShadowProxy(MockLLM("p"), MockLLM("s"), sample_rate=0.2, seed=seed, store=store) as proxy:
            for i in range(300):
                proxy.complete(f"q{i}", request_id=f"id-{i}")
        return sorted(r["request_id"] for r in store.read() if r["shadowed"])
    a, b = run(0), run(0)
    assert a == b and 30 < len(a) < 90
    assert run(7) != a


def test_jsonl_store_roundtrip_and_fields(tmp_path):
    store = JsonlStore(tmp_path / "s.jsonl")
    primary = MockLLM("mock-system-one", handlers=[(r".", "A")])
    shadow = MockLLM("mock-system-two", handlers=[(r".", "B")])
    with ShadowProxy(primary, shadow, sample_rate=1.0, store=store) as proxy:
        proxy.complete("hi", request_id="r1", meta={"ts": 123.0, "expected": "A"})
    (row,) = store.read()
    assert row["request_id"] == "r1" and row["prompt"] == "hi" and row["ts"] == 123.0
    for arm in ("primary", "shadow"):
        assert {"text", "model", "tokens", "latency_ms", "cost_usd"} <= set(row[arm])
    assert row["primary"]["text"] == "A" and row["shadow"]["text"] == "B"
    assert row["shadow"]["cost_usd"] > row["primary"]["cost_usd"] > 0  # PRICES_PER_MTOK differ


def test_agent_runner_stores_both_trajectories():
    def agent_a(task_id, inp):
        return Trajectory(task_id, model="a").add_tool("search", {"q": inp}).finish("42", True)

    def agent_b(task_id, inp):
        return Trajectory(task_id, model="b").add_tool("search", {"q": inp}).add_tool("calc", {}).finish("41", False)

    store = MemoryStore()
    with ShadowAgentRunner(agent_a, agent_b, sample_rate=1.0, store=store) as runner:
        t = runner.run("t1", "what is 6*7")
    assert t.final_answer == "42"
    (row,) = store.read()
    assert row["kind"] == "trajectory"
    assert row["primary"]["tool_names"] == ["search"] and row["shadow"]["tool_names"] == ["search", "calc"]
    assert Trajectory.from_dict(row["shadow"]["trajectory"]).final_answer == "41"
