"""Operators, metamorphic relations, LLM generator, dedup, baseline blind spots."""
from __future__ import annotations

import json
from pathlib import Path

from evalkit.io import read_jsonl
from evalkit.llm import MockLLM
from projects.p12_synthetic_edgecases.core import Deduper, make_case
from projects.p12_synthetic_edgecases.llm_gen import DEMO_BATCH, LLMGenerator, demo_llm_handler
from projects.p12_synthetic_edgecases.metamorphic import RELATIONS, generate_pairs, relation_holds
from projects.p12_synthetic_edgecases.operators import OPERATORS, apply_operators
from projects.p12_synthetic_edgecases.spec import TaskSpec, nfkc, validate_case
from projects.p12_synthetic_edgecases.sut import baseline_sut

DATA = Path(__file__).resolve().parent.parent / "data"
SPEC = TaskSpec.load(DATA / "task_spec.yaml")
GOLDEN = read_jsonl(DATA / "golden_seed.jsonl")


def test_every_operator_yields_valid_deduped_mutations():
    for name in OPERATORS:
        total = 0
        for g in GOLDEN:
            muts = apply_operators(g, SPEC, seed=0, names=[name])
            seen = set()
            for m in muts:
                assert m.operator_name == name and m.parent_id == g["id"]
                assert m.mutated_input != g["input"], (name, m.rationale)
                assert m.mutated_input not in seen, (name, m.rationale)       # no duplicates per parent
                seen.add(m.mutated_input)
                assert m.rationale
                case = make_case(m.mutated_input, m.expected, name, g, "rule", 0, m.rationale, None, SPEC)
                assert validate_case(case, SPEC) == [], (name, m.rationale, validate_case(case, SPEC))
                assert case["category"] in SPEC.failure_taxonomy
            total += len(muts)
        assert total > 0, name
    assert len(OPERATORS) == 14


def test_operators_are_deterministic_and_seed_sensitive():
    a = [m.to_dict() for g in GOLDEN for m in apply_operators(g, SPEC, seed=3)]
    b = [m.to_dict() for g in GOLDEN for m in apply_operators(g, SPEC, seed=3)]
    c = [m.to_dict() for g in GOLDEN for m in apply_operators(g, SPEC, seed=4)]
    assert a == b and a != c


def test_unicode_digit_mutations_keep_the_decimal_value():
    import unicodedata
    def ascii_digits(s):
        return "".join(str(unicodedata.decimal(ch)) if ch.isdecimal() else ch for ch in s)
    for g in GOLDEN[:5]:
        muts = apply_operators(g, SPEC, names=["unicode_digits"])
        assert len(muts) == 2
        for m in muts:
            assert m.mutated_input != g["input"]
            assert ascii_digits(m.mutated_input) == g["input"] and m.expected == g["expected"]
        assert nfkc(muts[0].mutated_input) == g["input"]          # full-width digits are NFKC-compatible


def test_metamorphic_pairs_have_the_stated_relation():
    n = 0
    for g in GOLDEN:
        for p in generate_pairs(g, seed=0):
            assert p.relation in RELATIONS and p.follow_up_input != g["input"]
            assert relation_holds(p, g["expected"])
            if p.relation in {"paraphrase", "add_irrelevant", "reorder_clauses"}:
                assert p.expected == g["expected"]
            elif p.relation == "scale_qty":
                assert p.expected["qty"] == 2 * g["expected"]["qty"] and p.expected["sku"] == g["expected"]["sku"]
            elif p.relation == "shift_date":
                assert p.expected["date"] > g["expected"]["date"] and p.expected["qty"] == g["expected"]["qty"]
            n += 1
    assert n >= 150
    assert generate_pairs({"id": "e", "input": "x", "expected": {"error": "parse_failed"}}) == []


def test_llm_generator_parses_mock_json_and_rejects_malformed_rows():
    gen = LLMGenerator(MockLLM(handlers=[(r".", demo_llm_handler)]))
    rows = gen.generate(SPEC, GOLDEN, list(OPERATORS), n=10)
    assert len(rows) == len(DEMO_BATCH) - 3 and all(r["operator_name"] == "llm" for r in rows)
    assert {r["category"] for r in rows} <= set(SPEC.failure_taxonomy)
    reasons = [r["reason"] for r in gen.rejected]
    assert len(reasons) == 3
    assert any("not an int" in r for r in reasons) and any("empty input" in r for r in reasons)
    assert any("not in taxonomy" in r for r in reasons)
    garbage = LLMGenerator(MockLLM(handlers=[(r".", "sorry, no cases today")]))
    assert garbage.generate(SPEC, GOLDEN, [], 5) == [] and "unparseable" in garbage.rejected[0]["reason"]
    obj = LLMGenerator(MockLLM(handlers=[(r".", json.dumps({"input": "x"}))]))
    assert obj.generate(SPEC, GOLDEN, [], 5) == [] and obj.rejected[0]["reason"] == "not a JSON array"


def test_deduper_flags_exact_and_near_duplicates():
    d = Deduper(GOLDEN, threshold=0.85)
    g = GOLDEN[0]
    assert d.check({"input": g["input"].upper() + "  ", "parent_id": g["id"]}) == "exact_duplicate"
    assert d.check({"input": g["input"] + "!", "parent_id": g["id"]}) == "near_duplicate_of_golden"
    novel = {"input": "Ship a dozen of SKU-9999 on 2026-03-14 — actually, cancel that.", "parent_id": g["id"]}
    assert d.check(novel) is None
    assert d.check({"input": novel["input"] + ".", "parent_id": g["id"]}) == "near_duplicate_of_sibling"
    assert d.check({"input": "Totally different request text with no overlap at all 42", "parent_id": None}) is None


def test_baseline_sut_has_the_documented_blind_spots():
    assert baseline_sut("Ship 12 units of SKU-4471 on 2026-03-14") == {"qty": 12, "sku": "SKU-4471", "date": "2026-03-14"}
    assert baseline_sut("Ship １２ units of SKU-4471 on 2026-03-14") == {"error": "parse_failed"}     # unicode digits
    assert baseline_sut("Ship 12 units of SKU-4471 on 03/14/26") == {"error": "parse_failed"}          # two-digit year
    assert baseline_sut("Ship a dozen units of SKU-4471 on 2026-03-14") == {"error": "parse_failed"}   # qty words
    assert baseline_sut("") == {"error": "empty_input"}
    assert all(baseline_sut(g["input"]) == g["expected"] for g in GOLDEN)
