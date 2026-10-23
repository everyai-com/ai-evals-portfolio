import json

import pytest

from evalkit.llm import MockLLM, approx_tokens

from projects.p13_context_eviction import cli
from projects.p13_context_eviction.core import (
    FactPlanter, LLMAnswerer, LookupAnswerer, Message, NoiseGenerator, RetrievalContext,
    SummarizeOldest, TruncateOldest, build_conversation, default_managers, run_matrix,
    summary_handler, total_tokens,
)

BUDGET = 4000


def matrix(noise, managers=("truncate", "summarize", "retrieval"), facts=12, seed=0):
    llm = cli.make_llm()
    all_m = default_managers(llm)
    res = run_matrix({m: all_m[m] for m in managers}, noise, BUDGET, facts, seed)
    return {(r.manager, r.noise_tokens): r for r in res}


def test_truncate_loses_early_facts_but_retrieval_keeps_them():
    res = matrix([32000], ("truncate", "retrieval"))
    trunc, retr = res[("truncate", 32000)], res[("retrieval", 32000)]
    assert trunc.by_position("found")["early"] == 0.0
    assert trunc.by_position("found")["middle"] == 0.0
    assert trunc.recall < 0.5
    assert retr.by_position("found")["early"] == 1.0
    assert retr.recall == 1.0 and retr.latest_correct == 1.0
    assert retr.false_retention < trunc.false_retention


def test_truncate_keeps_everything_when_it_fits():
    res = matrix([2000], ("truncate",))
    r = res[("truncate", 2000)]
    assert r.recall == 1.0 and r.latest_correct == 1.0 and r.compression_ratio == pytest.approx(1.0, abs=0.01)


def test_contradicting_updates_resolve_to_latest():
    res = matrix([8000], ("summarize", "retrieval"))
    for name in ("summarize", "retrieval"):
        r = res[(name, 8000)]
        assert any(p.has_update for p in r.probes)
        assert r.recall == 1.0
        assert r.update_correct == 1.0 and r.latest_correct == 1.0
    # summarisation actually happened and compressed the context
    s = res[("summarize", 8000)]
    assert s.compression_ratio < 0.6 and s.mean_rendered_tokens <= BUDGET


def test_summary_keeps_only_key_lines_in_order():
    text = "[user] hello\n[user] Note: KEY: budget=1\n[tool] log\n[user] Correction: KEY: budget changed to 2\n[user] KEY: budget=1"
    assert summary_handler("", "SUMMARIZE\n" + text) == "KEY: budget=1\nKEY: budget changed to 2"


def test_summarize_oldest_calls_llm_and_fits_budget():
    llm = MockLLM(handlers=[(r"^SUMMARIZE", summary_handler)])
    mgr = SummarizeOldest(llm)
    mgr.add(Message("user", "Note: KEY: owner=own_1"))
    for i in range(200):
        mgr.add(Message("tool", f"log line {i} " * 20))
    mgr.add(Message("user", "Correction: KEY: owner changed to own_2"))
    out = mgr.render(500)
    assert mgr.llm_calls >= 1 and len(llm.calls) == mgr.llm_calls
    assert approx_tokens(out) <= 500
    assert LookupAnswerer().answer(out, "", "owner") == "own_2"
    calls_before = mgr.llm_calls
    mgr.render(500)  # state persists; no extra compression
    assert mgr.llm_calls == calls_before


def test_retrieval_ranks_by_query_and_fills_with_recency():
    r = RetrievalContext()
    r.add(Message("user", "KEY: region=reg_1"))
    for i in range(50):
        r.add(Message("tool", f"batch {i} processed"))
    r.add(Message("user", "KEY: region changed to reg_2"))
    sel = r.select(60, query="What is the current value of region?")
    assert sel[0] == 0 and sel[-1] == 51 and sel == sorted(sel)
    assert LookupAnswerer().answer(r.render(60, "What is the current value of region?"), "", "region") == "reg_2"
    assert r.select(60, query="") == list(range(52))[-len(r.select(60, "")):]  # pure recency


def test_lookup_answerer_takes_last_value():
    ctx = "KEY: budget=bud_1\nnoise\nKEY: budget changed to bud_2\nKEY: tier=tie_9"
    a = LookupAnswerer()
    assert a.answer(ctx, "", "budget") == "bud_2"
    assert a.answer(ctx, "", "tier") == "tie_9"
    assert a.answer(ctx, "", "owner") is None


def test_llm_answerer_is_pluggable():
    llm = cli.make_llm()
    ctx = "[user] KEY: budget=bud_1\n[user] Correction: KEY: budget changed to bud_2"
    a = LLMAnswerer(llm)
    assert a.answer(ctx, "What is the current value of budget?", "budget") == "bud_2"
    assert a.answer(ctx, "What is the current value of owner?", "owner") is None
    assert len(llm.calls) == 2


def test_noise_generator_is_seeded_and_sized():
    a = NoiseGenerator(seed=3).generate(5000)
    b = NoiseGenerator(seed=3).generate(5000)
    c = NoiseGenerator(seed=4).generate(5000)
    assert [m.content for m in a] == [m.content for m in b]
    assert [m.content for m in a] != [m.content for m in c]
    assert 5000 <= total_tokens(a) <= 5600
    keys = set(FactPlanter.KEYS)
    assert not any(k in m.content for m in a for k in keys)


def test_fact_planter_positions_and_updates():
    facts = FactPlanter(seed=0).plant(12)
    assert [f.position for f in facts[:3]] == ["early", "middle", "late"]
    assert all(f.update_frac > f.frac for f in facts if f.update_value)
    msgs, noise = build_conversation(NoiseGenerator(0).generate(2000), facts)
    assert len(msgs) == len(noise) + len(facts) + sum(1 for f in facts if f.update_value)


def test_determinism_across_runs_with_same_seed():
    a = [r.to_dict() for r in run_matrix(default_managers(cli.make_llm()), [2000, 8000], BUDGET, 9, seed=5)]
    b = [r.to_dict() for r in run_matrix(default_managers(cli.make_llm()), [2000, 8000], BUDGET, 9, seed=5)]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c = [r.to_dict() for r in run_matrix(default_managers(cli.make_llm()), [2000, 8000], BUDGET, 9, seed=6)]
    assert json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True)


def test_rendered_context_respects_budget():
    for r in matrix([8000]).values():
        assert all(p.rendered_tokens <= BUDGET for p in r.probes)


def test_cli_end_to_end(tmp_path, capsys):
    out = tmp_path / "m.json"
    rc = cli.main(["run", "--budget", "4000", "--noise", "2000,8000", "--facts", "6", "--seed", "0", "--json", str(out)])
    assert rc == 0
    text = capsys.readouterr().out
    assert "| truncate | 2000 |" in text and "| retrieval | 8000 |" in text
    assert len(json.loads(out.read_text())) == 6
    assert cli.main(["run", "--managers", "bogus"]) == 2


def test_committed_fixture_matches_fresh_run():
    from pathlib import Path
    fixture = json.loads((Path(__file__).resolve().parents[1] / "data" / "default_run.json").read_text())
    fresh = [r.to_dict() for r in run_matrix(default_managers(cli.make_llm()), [2000, 8000, 32000], 4000, 12, seed=0)]
    assert json.dumps(fresh, sort_keys=True) == json.dumps(fixture, sort_keys=True)
