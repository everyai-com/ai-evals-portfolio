"""Difficulty filter, coverage gaps, promote (semver + changelog), determinism, CLI."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from evalkit.io import read_jsonl, write_jsonl
from evalkit.llm import MockLLM
from projects.p12_synthetic_edgecases import cli
from projects.p12_synthetic_edgecases.core import (
    DifficultyEstimator,
    coverage_matrix,
    generate_cases,
    promote,
    render_coverage,
    select_round_robin,
)
from projects.p12_synthetic_edgecases.llm_gen import demo_llm_handler
from projects.p12_synthetic_edgecases.spec import TaskSpec
from projects.p12_synthetic_edgecases.sut import baseline_sut

DATA = Path(__file__).resolve().parent.parent / "data"
SPEC = TaskSpec.load(DATA / "task_spec.yaml")
GOLDEN = read_jsonl(DATA / "golden_seed.jsonl")


def _client():
    return MockLLM(handlers=[(r".", demo_llm_handler)])


def test_difficulty_filter_keeps_failures_and_drops_always_pass():
    est = DifficultyEstimator(baseline_sut, n_runs=3)
    hard = {"input": "Ship １２ units of SKU-4471 on 2026-03-14", "expected": {"qty": 12, "sku": "SKU-4471", "date": "2026-03-14"}}
    easy = {"input": "Ship 12 units of SKU-4471 on 2026-03-14 Thanks!", "expected": {"qty": 12, "sku": "SKU-4471", "date": "2026-03-14"}}
    assert est.assess(hard)["hard"] and est.assess(hard)["failures"] == 3
    assert not est.assess(easy)["hard"] and est.assess(easy)["status"] == "easy"
    assert est.assess({"input": "Do not ship anything", "expected": None})["status"] == "unlabelled"
    calls = {"n": 0}

    def flaky(text):
        calls["n"] += 1
        return {"qty": calls["n"], "sku": "SKU-0000", "date": "2026-01-01"}
    d = DifficultyEstimator(flaky, n_runs=3).assess({"input": "x", "expected": {"qty": 1, "sku": "SKU-0000", "date": "2026-01-01"}})
    assert d["disagreement"] and d["hard"]
    res = generate_cases(GOLDEN, SPEC, n=500, seed=0, client=None)
    assert res.stats["dropped"]["baseline_always_passes"] > 500
    assert all(c["difficulty"]["hard"] for c in res.kept)
    assert not any(c["operator"] == "injection" for c in res.kept)        # baseline shrugs off injections
    easy_kept = generate_cases(GOLDEN, SPEC, n=5000, seed=0, client=None, keep_easy=True)
    assert any(c["operator"] == "injection" for c in easy_kept.kept)


def test_coverage_matrix_lists_uncovered_cells():
    res = generate_cases(GOLDEN, SPEC, n=200, seed=0, client=_client())
    cov = res.coverage
    tags = {t for g in GOLDEN for t in g["tags"]}
    assert set(cov.tag_by_category) == tags
    assert ("iso_date", "injection") in cov.uncovered and ("iso_date", "length") in cov.uncovered
    assert {"injection", "length"} <= set(cov.operator_category_uncovered)
    assert ("iso_date", "boundary") not in cov.uncovered
    only_boundary = [c for c in res.kept if c["operator"] == "boundary"]
    sub = coverage_matrix(only_boundary, SPEC, GOLDEN)
    assert sub.by_operator == {"boundary": len(only_boundary)}
    assert all(k != "boundary" for _, k in sub.uncovered if _ == "iso_date") and len(sub.uncovered) > len(cov.uncovered)
    md = render_coverage(cov, SPEC, res.stats)
    assert "## Uncovered cells" in md and "- iso_date x injection" in md


def test_generation_is_deterministic_with_same_seed():
    a = generate_cases(GOLDEN, SPEC, n=120, seed=7, client=_client())
    b = generate_cases(GOLDEN, SPEC, n=120, seed=7, client=_client())
    assert [c["id"] for c in a.kept] == [c["id"] for c in b.kept] and len(a.kept) == 120
    assert [c["input"] for c in a.kept] == [c["input"] for c in b.kept]
    c = generate_cases(GOLDEN, SPEC, n=120, seed=8, client=_client())
    assert [x["id"] for x in c.kept] != [x["id"] for x in a.kept]
    assert len({x["id"] for x in a.kept}) == 120                       # ids unique
    assert all({"operator", "parent_id", "generator", "seed"} <= set(x) for x in a.kept)   # provenance
    assert {x["generator"] for x in a.kept} == {"rule", "metamorphic", "llm"}


def test_round_robin_selection_balances_operators():
    cases = [{"id": f"a{i}", "operator": "a"} for i in range(50)] + [{"id": f"b{i}", "operator": "b"} for i in range(5)]
    sel = select_round_robin(cases, 20, seed=0)
    assert len(sel) == 20 and sum(c["operator"] == "b" for c in sel) == 5


def test_promote_bumps_version_and_appends_changelog(tmp_path):
    golden, version, changelog = tmp_path / "golden.jsonl", tmp_path / "VERSION", tmp_path / "CHANGELOG.md"
    shutil.copy(DATA / "golden_seed.jsonl", golden)
    version.write_text("0.1.0\n")
    res = generate_cases(GOLDEN, SPEC, n=60, seed=0, client=None)
    review = res.review_queue[:3]
    labelled = [c for c in res.kept if c["expected"] is not None][:2]
    reviewed = [{"id": review[0]["id"], "approved": True, "expected": {"error": "no_shipment"}},
                {"id": review[1]["id"], "approved": False, "expected": {"error": "no_shipment"}},
                {"id": review[2]["id"], "approved": True, "expected": {"qty": "12", "sku": "SKU-0000", "date": "2026-01-01"}},
                {"id": labelled[0]["id"], "approved": True}]
    out = promote(res.kept, reviewed, golden, version, changelog, SPEC, today="2026-10-24")
    assert out["added"] == 2 and out["version"] == "0.2.0" and out["previous_version"] == "0.1.0"
    assert out["skipped"] == {"not_approved_or_unknown": 1, "invalid_label": 1}
    rows = read_jsonl(golden)
    assert len(rows) == 42 and rows[-2]["expected"] == {"error": "no_shipment"} and "synthetic" in rows[-2]["tags"]
    assert version.read_text().strip() == "0.2.0"
    assert "- 0.2.0 (2026-10-24): +2 synthetic cases promoted" in changelog.read_text()
    again = promote(res.kept, reviewed, golden, version, changelog, SPEC, today="2026-10-25")
    assert again["added"] == 0 and version.read_text().strip() == "0.2.0"
    everything = promote(res.kept, [], golden, version, changelog, SPEC, include_labelled=True)
    assert everything["version"] == "0.3.0" and everything["added"] == res.stats["labelled"] - 1


def test_cli_generate_coverage_promote(tmp_path):
    out = tmp_path / "out"
    assert cli.main(["generate", "--golden", str(DATA / "golden_seed.jsonl"), "--spec", str(DATA / "task_spec.yaml"),
                     "--n", "50", "--seed", "0", "--out", str(out)]) == 0
    gen = read_jsonl(out / "generated.jsonl")
    assert len(gen) == 50 and (out / "coverage.md").exists()
    rq = read_jsonl(out / "review_queue.jsonl")
    assert rq and all(r["expected"] is None for r in rq)
    assert cli.main(["coverage", "--generated", str(out / "generated.jsonl"), "--spec", str(DATA / "task_spec.yaml"),
                     "--golden", str(DATA / "golden_seed.jsonl"), "--json", str(tmp_path / "cov.json")]) == 0
    assert "uncovered" in json.loads((tmp_path / "cov.json").read_text())
    golden, version, changelog = tmp_path / "g.jsonl", tmp_path / "VERSION", tmp_path / "C.md"
    shutil.copy(DATA / "golden_seed.jsonl", golden)
    version.write_text("1.4.0")
    reviewed = tmp_path / "reviewed.jsonl"
    write_jsonl(reviewed, [{"id": rq[0]["id"], "approved": True, "expected": {"error": "no_shipment"}}])
    assert cli.main(["promote", "--generated", str(out / "generated.jsonl"), "--reviewed", str(reviewed), "--golden", str(golden),
                     "--spec", str(DATA / "task_spec.yaml"), "--version", str(version), "--changelog", str(changelog)]) == 0
    assert version.read_text().strip() == "1.5.0" and len(read_jsonl(golden)) == 41
    assert cli.main(["promote", "--generated", str(out / "generated.jsonl"), "--golden", str(golden), "--spec",
                     str(DATA / "task_spec.yaml"), "--version", str(version), "--changelog", str(changelog)]) == 1
    assert cli.main(["bogus"]) == 2
