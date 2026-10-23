from pathlib import Path

import pytest

from evalkit.llm import MockLLM

from projects.p11_counterfactual_replay import cli, demo
from projects.p11_counterfactual_replay.core import (
    Recorder, Replayer, bisect, diff, hops, load_run, node_id_of, node_output,
)

DATA = Path(__file__).resolve().parents[1] / "data"


def record(corrupt=False, task_id="golden"):
    llm = demo.make_llm()
    rec = Recorder(llm, demo.make_tools(corrupt_search=corrupt), task_id=task_id)
    return rec.run(demo.demo_agent), llm


def test_recorder_captures_every_hop_with_hashes():
    traj, llm = record()
    assert [node_id_of(s) for s in hops(traj)] == demo.NODE_ORDER
    assert len(llm.calls) == 4
    assert traj.final_answer == demo.GOLDEN_ANSWER
    for s in hops(traj):
        assert len(s.meta["input_hash"]) == 16 and s.meta["source"] == "live"
    assert node_output(hops(traj)[1]) == ["acme-offices-2025", "acme-benefits"]


def test_corrupted_search_gives_wrong_answer():
    traj, _ = record(corrupt=True)
    assert traj.final_answer != demo.GOLDEN_ANSWER
    assert "550" in traj.final_answer


def test_cache_hit_replay_makes_zero_model_calls_before_override():
    traj, _ = record()
    fresh = MockLLM(model="mock-system-two", handlers=demo.demo_handlers())
    rp = Replayer(traj, {"compute": 999.0}, llm=fresh, tools=demo.make_tools())
    out = rp.run(demo.demo_agent)
    # plan / extract are before the override -> served from cache; verify / answer are live
    assert len(fresh.calls) == 2
    assert rp.cache_hits == 4 and rp.live_calls == 2
    sources = [s.meta["source"] for s in hops(out)]
    assert sources == ["cache", "cache", "cache", "cache", "override", "live", "live"]
    # downstream LLM hops ran on the new model; cached ones keep the recorded model
    assert [s.model for s in hops(out) if s.kind == "llm"] == ["mock-system-one", "mock-system-one",
                                                               "mock-system-two", "mock-system-two"]


def test_override_at_last_node_needs_no_backend():
    traj, _ = record()
    out = Replayer(traj, {"answer": "forty-two"}).run(demo.demo_agent)
    assert out.final_answer == "forty-two"
    assert all(s.meta["source"] in ("cache", "override") for s in hops(out))


def test_override_changes_downstream_and_diff_reports_it():
    traj, _ = record()
    rp = Replayer(traj, {"search": ["zenith-offices-2025", "acme-history"]},
                  llm=demo.make_llm(), tools=demo.make_tools())
    out = rp.run(demo.demo_agent)
    assert out.final_answer != traj.final_answer
    d = diff(traj, out)
    assert d.first_divergence == 1 and d.rows[1].node_b == "search"
    assert d.changed_indices == [1, 2, 3, 4, 6]  # verify says VERIFIED in both runs
    assert d.final_diverges
    md = d.to_markdown()
    assert "DIVERGE" in md and "**changed**" in md


def test_swapping_search_back_fixes_corrupted_run():
    bad, _ = record(corrupt=True)
    good, _ = record()
    golden_search = node_output(hops(good)[1])
    rp = Replayer(bad, {"search": golden_search}, llm=demo.make_llm(), tools=demo.make_tools())
    assert rp.run(demo.demo_agent).final_answer == demo.GOLDEN_ANSWER


def test_tool_override_reexecutes_node():
    bad, _ = record(corrupt=True)
    rp = Replayer(bad, tool_overrides={"search": demo.make_tools()["search"]},
                  llm=demo.make_llm(), tools=demo.make_tools())
    out = rp.run(demo.demo_agent)
    assert out.final_answer == demo.GOLDEN_ANSWER
    assert hops(out)[1].meta["source"] == "tool_override"


def test_memo_mode_reuses_unchanged_downstream_hops():
    traj, _ = record()
    fresh = MockLLM(model="mock-system-two", handlers=demo.demo_handlers())
    # overriding plan with its own recorded output changes nothing -> everything is a cache hit
    rp = Replayer(traj, {"plan": node_output(hops(traj)[0])}, llm=fresh, tools=demo.make_tools(), mode="memo")
    rp.run(demo.demo_agent)
    assert len(fresh.calls) == 0 and rp.live_calls == 0


def test_bisect_identifies_corrupted_node():
    bad, _ = record(corrupt=True)
    good, _ = record()
    res = bisect(bad, good, demo.demo_agent, llm=demo.make_llm(), tools=demo.make_tools())
    assert res.culprit == "search" and res.culprit_index == 1
    assert [t["node"] for t in res.tried] == ["plan", "search"]
    assert res.tried[0]["fixed"] is False and res.tried[1]["fixed"] is True


def test_unknown_override_target_raises():
    traj, _ = record()
    with pytest.raises(KeyError):
        Replayer(traj, {"nope": 1})


def test_committed_fixtures_match_fresh_recordings():
    assert load_run(DATA / "golden_run.jsonl").final_answer == demo.GOLDEN_ANSWER
    bad = load_run(DATA / "corrupted_run.jsonl")
    assert bad.final_answer != demo.GOLDEN_ANSWER
    fresh, _ = record(corrupt=True, task_id="corrupted")
    assert [node_output(s) for s in hops(bad)] == [node_output(s) for s in hops(fresh)]


def test_cli_end_to_end(tmp_path, capsys):
    assert cli.main(["record", "--out", str(tmp_path), "--task-id", "g"]) == 0
    assert cli.main(["record", "--out", str(tmp_path), "--task-id", "bad", "--corrupt", "search"]) == 0
    assert cli.main(["diff", "--a", str(tmp_path / "g.jsonl"), "--b", str(tmp_path / "bad.jsonl")]) == 0
    assert cli.main(["bisect", "--run", str(tmp_path / "bad.jsonl"), "--golden", str(tmp_path / "g.jsonl")]) == 0
    out = capsys.readouterr().out
    assert "culprit = `search`" in out
    rc = cli.main(["replay", "--run", str(tmp_path / "bad.jsonl"),
                   "--override", 'search=["acme-offices-2025","acme-benefits"]', "--out", str(tmp_path / "fixed.jsonl")])
    assert rc == 0
    assert load_run(tmp_path / "fixed.jsonl").final_answer == demo.GOLDEN_ANSWER
    assert cli.main(["replay", "--run", str(tmp_path / "bad.jsonl"), "--override", "compute=1"]) == 1
    assert cli.main(["replay", "--run", str(tmp_path / "bad.jsonl"), "--override", "garbage"]) == 2
