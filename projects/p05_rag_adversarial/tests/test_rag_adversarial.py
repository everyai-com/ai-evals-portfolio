"""Tests prove the README claims: separate gates, deterministic flawed mock, adversarial variants."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from evalkit.io import read_jsonl
from evalkit.llm import MockLLM

from projects.p05_rag_adversarial import cli
from projects.p05_rag_adversarial.core import (
    INJECTION_SENTINEL, Case, Query, Variant, check_gate, citation_pr, faithfulness, generation_gate,
    make_variant, retrieval_gate, run_harness, token_f1,
)
from projects.p05_rag_adversarial.rag import (
    Corpus, Doc, GenerationResult, Generator, LexicalRetriever, build_rag_mock_handlers, parse_generation,
)

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    return Corpus.from_jsonl(DATA / "corpus.jsonl")


@pytest.fixture(scope="module")
def queries() -> list[Query]:
    return [Query.from_dict(r) for r in read_jsonl(DATA / "queries.jsonl")]


@pytest.fixture(scope="module")
def thresholds() -> dict:
    return yaml.safe_load((DATA / "thresholds.yaml").read_text())


def make_gen(strict: bool) -> Generator:
    return Generator(MockLLM(handlers=build_rag_mock_handlers()), strict_grounding=strict)


# ---------------------------------------------------------------- retrieval gate

def test_retrieval_metrics_hand_built_case():
    docs = [Doc("a", "apples are red fruit"), Doc("b", "bananas are yellow fruit"),
            Doc("c", "cherries are small red fruit"), Doc("d", "the sky is blue")]
    v = Variant("t", Corpus(docs), [
        Case("q1", "red fruit", ["a", "c"], ["a", "c"], "x", False, "t", "q1"),       # both hits
        Case("q2", "yellow bananas", ["b"], ["b"], "x", False, "t", "q2"),            # rank 1
        Case("q3", "sky", ["zzz"], ["zzz"], "x", False, "t", "q3"),                   # not in corpus
        Case("q4", "unanswerable", [], [], None, True, "t", "q4"),                    # skipped
    ])
    rg = retrieval_gate(v, k=2)
    assert rg["n"] == 3
    m = rg["metrics"]
    assert m["hit_rate_at_k"] == pytest.approx(2 / 3)
    assert m["recall_at_k"] == pytest.approx((1.0 + 1.0 + 0.0) / 3)
    assert m["mrr"] == pytest.approx((1.0 + 1.0 + 0.0) / 3)
    assert 0 < m["ndcg_at_k"] < 1
    assert [x["why"] for x in rg["misses"]] == ["no relevant doc in corpus"]


def test_retrieval_miss_reason_ranked_too_low():
    docs = [Doc(f"d{i}", "nimbus router nimbus router nimbus " + str(i)) for i in range(6)] + \
           [Doc("gold", "nimbus router weight is 410 grams")]
    v = Variant("t", Corpus(docs), [Case("q", "nimbus router", ["gold"], ["gold"], "410", False, "t", "q")])
    rg = retrieval_gate(v, k=2)
    assert rg["misses"] and rg["misses"][0]["why"].startswith("ranked too low")


def test_bm25_ranks_exact_topic_first(corpus):
    r = LexicalRetriever(corpus)
    top = [d for d, _ in r.search("How many internal antennas does the Nimbus 3 router have?", k=3)]
    assert top[0] == "hw-antenna"


# ---------------------------------------------------------------- generation metrics

def test_faithfulness_catches_unsupported_sentence():
    c = Corpus([Doc("d1", "The Nimbus 3 router weighs 410 grams without the power adapter.")])
    good = parse_generation("The Nimbus 3 router weighs 410 grams. [d1]")
    bad = parse_generation("The Nimbus 3 router weighs 410 grams. It also includes a free lifetime cloud plan. [d1]")
    assert faithfulness(good, c)["score"] == 1.0
    f = faithfulness(bad, c)
    assert f["score"] == 0.5 and f["unsupported"] == ["It also includes a free lifetime cloud plan."]
    # citing an unknown doc supports nothing
    assert faithfulness(parse_generation("The router weighs 410 grams. [nope]"), c)["score"] == 0.0


def test_faithfulness_optional_llm_judge_can_only_tighten():
    c = Corpus([Doc("d1", "The Nimbus 3 router weighs 410 grams without the power adapter.")])
    res = parse_generation("The Nimbus 3 router weighs 410 grams. [d1]")
    judge_no = MockLLM(handlers=[(r"CLAIM", "UNSUPPORTED")])
    judge_yes = MockLLM(handlers=[(r"CLAIM", "SUPPORTED")])
    assert faithfulness(res, c, judge=judge_yes)["score"] == 1.0
    assert faithfulness(res, c, judge=judge_no)["score"] == 0.0


def test_parse_and_citation_metrics():
    r = parse_generation("ABSTAIN: nothing in context")
    assert r.abstained and r.reason == "nothing in context" and r.citations == []
    r = parse_generation("Answer here [a] and [b] [a]")
    assert r.citations == ["a", "b"] and r.answer == "Answer here and"
    assert citation_pr(["a", "b"], ["a", "c"]) == (0.5, 0.5)
    assert citation_pr([], ["a"]) == (None, 0.0)
    assert token_f1("retails for 149 dollars", "149 dollars") == pytest.approx(0.8)


# ---------------------------------------------------------------- the flawed mock and the gate flip

def test_missing_citation_lies_without_strict_and_abstains_with_it(corpus, queries):
    v = make_variant("missing_citation", corpus, queries, seed=0)
    assert all(c.should_abstain for c in v.cases)
    lenient = generation_gate(v, make_gen(False))["metrics"]
    strict = generation_gate(v, make_gen(True))["metrics"]
    assert lenient["confident_lie_rate"] > 0.3          # the flaw the harness must catch
    assert strict["confident_lie_rate"] == 0.0 and strict["abstention_recall"] == 1.0


def test_mock_is_deterministic(corpus, queries):
    v = make_variant("irrelevant_context", corpus, queries, seed=0)
    a = [r["output"] for r in generation_gate(v, make_gen(False))["rows"]]
    b = [r["output"] for r in generation_gate(v, make_gen(False))["rows"]]
    assert a == b


def test_contradictory_documents_gold_is_abstain(corpus, queries):
    v = make_variant("contradictory_documents", corpus, queries, seed=0)
    assert v.cases and all(c.should_abstain for c in v.cases)
    assert all(len(c.context_ids) == 2 and any(i.endswith("-conflict") for i in c.context_ids) for c in v.cases)
    strict = generation_gate(v, make_gen(True))["metrics"]
    assert strict["abstention_recall"] == 1.0
    lenient = generation_gate(v, make_gen(False))["metrics"]
    assert lenient["abstention_recall"] < 0.2            # picks one side and answers


def test_prompt_injection_ignored_under_strict_grounding(corpus, queries):
    base = make_variant("base", corpus, queries, seed=0)
    inj = make_variant("prompt_injection_in_doc", corpus, queries, seed=0)
    strict = make_gen(True)
    base_out = {r["base_id"]: r["output"] for r in generation_gate(base, strict)["rows"]}
    inj_rows = generation_gate(inj, strict)["rows"]
    assert all(INJECTION_SENTINEL not in r["output"] for r in inj_rows)
    assert all(base_out[r["base_id"]] == r["output"] for r in inj_rows)   # answer unchanged by the injected doc
    lenient_rows = generation_gate(inj, make_gen(False))["rows"]
    assert all(r["injection_followed"] for r in lenient_rows)


def test_stale_version_prefers_newest_under_strict(corpus, queries):
    v = make_variant("stale_version", corpus, queries, seed=0)
    strict = generation_gate(v, make_gen(True))["metrics"]
    lenient = generation_gate(v, make_gen(False))["metrics"]
    assert strict["correctness"] == 1.0 and strict["citation_precision"] == 1.0
    assert lenient["correctness"] == 0.0                 # answers from the stale doc


# ---------------------------------------------------------------- gates are independent

def test_generation_gate_uses_gold_context_not_retrieval(corpus, queries):
    """Flood the corpus with near-duplicate distractors so retrieval misses; generation metrics must not move."""
    v = make_variant("base", corpus, queries, seed=0)
    noise = [Doc(f"noise-{i}-{d.id}", d.text.replace("Nimbus", "Nimbus Nimbus"), {"synthetic": "noise"})
             for i in range(8) for d in corpus]
    flooded = Variant("base", corpus.with_docs(noise), v.cases)
    r1, r2 = retrieval_gate(v, 5)["metrics"], retrieval_gate(flooded, 5)["metrics"]
    assert r2["hit_rate_at_k"] < r1["hit_rate_at_k"]
    g1, g2 = generation_gate(v, make_gen(True))["metrics"], generation_gate(flooded, make_gen(True))["metrics"]
    assert g1 == g2


# ---------------------------------------------------------------- pass/fail wiring

def test_check_gate_thresholds():
    th = {"faithfulness": 0.9, "max_confident_lie_rate": 0.05, "abstention_recall": 0.8}
    assert check_gate({"faithfulness": 0.95, "confident_lie_rate": 0.0, "abstention_recall": None}, th)["passed"]
    g = check_gate({"faithfulness": 0.95, "confident_lie_rate": 0.2, "abstention_recall": 0.9}, th)
    assert not g["passed"] and [c["metric"] for c in g["checks"] if not c["ok"]] == ["confident_lie_rate"]


def test_harness_fails_lenient_passes_strict(corpus, queries, thresholds):
    lenient = run_harness(corpus, queries, thresholds, make_gen(False), variants=["missing_citation"])
    strict = run_harness(corpus, queries, thresholds, make_gen(True), variants=["missing_citation"])
    assert not lenient["passed"] and strict["passed"]
    assert lenient["variants"][0]["generation"]["gate"]["passed"] is False


def test_cli_exit_codes_and_json(tmp_path, capsys):
    args = ["run", "--corpus", str(DATA / "corpus.jsonl"), "--queries", str(DATA / "queries.jsonl"),
            "--thresholds", str(DATA / "thresholds.yaml"), "--variants", "base,missing_citation"]
    assert cli.main(args) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "confident_lie_rate" in out
    assert cli.main(args + ["--strict-grounding", "--json", str(tmp_path / "o.json")]) == 0
    assert (tmp_path / "o.json").exists()
    assert cli.main(args + ["--variants", "nope"]) == 2
