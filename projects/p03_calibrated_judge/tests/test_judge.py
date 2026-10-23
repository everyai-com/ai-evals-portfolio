"""Tests prove the README claims: injected biases are detected, debiasing works, data is versioned."""

import re
from collections import Counter
from pathlib import Path

import pytest

from evalkit.io import dataset_fingerprint, read_jsonl
from evalkit.llm import MockLLM
from evalkit.metrics import cohen_kappa
from projects.p03_calibrated_judge import core
from projects.p03_calibrated_judge.cli import main
from projects.p03_calibrated_judge.data.make_anchors import STRATA, make_anchors

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def anchors():
    return read_jsonl(DATA / "anchors.jsonl")


@pytest.fixture(scope="module")
def biased(anchors):
    return core.calibrate(
        core.PairwiseJudge(core.make_mock_judge(anchors, "biased")), anchors, "biased"
    )


@pytest.fixture(scope="module")
def unbiased(anchors):
    return core.calibrate(
        core.PairwiseJudge(core.make_mock_judge(anchors, "unbiased")), anchors, "unbiased"
    )


# ---------------------------------------------------------------- dataset


def test_dataset_shape_and_strata(anchors):
    assert len(anchors) == 500
    counts = Counter(r["strata"] for r in anchors)
    assert set(counts) == set(STRATA) and all(v == 100 for v in counts.values())
    required = {
        "id",
        "prompt",
        "response_a",
        "response_b",
        "source_model_a",
        "source_model_b",
        "length_a",
        "length_b",
        "annotator_1",
        "annotator_2",
        "gold",
        "strata",
        "difficulty",
    }
    assert all(required <= set(r) for r in anchors)
    assert {r["gold"] for r in anchors} == {"A", "B", "tie"}
    assert len({r["id"] for r in anchors}) == 500


def test_generator_is_seeded_and_annotators_agree_about_85_percent():
    a, b = make_anchors(100, seed=3), make_anchors(100, seed=3)
    assert a == b
    rows = make_anchors(500, seed=0)
    agree = sum(r["annotator_1"] == r["annotator_2"] for r in rows) / 500
    assert 0.78 <= agree <= 0.92
    k = cohen_kappa([r["annotator_1"] for r in rows], [r["annotator_2"] for r in rows])
    assert 0.6 <= k <= 0.9


def test_fingerprint_matches_changelog(anchors):
    fp = dataset_fingerprint(DATA / "anchors.jsonl")
    changelog = (DATA / "CHANGELOG.md").read_text()
    sha = re.search(r"sha256: `([0-9a-f]{64})`", changelog).group(1)
    rows = int(re.search(r"rows: (\d+)", changelog).group(1))
    assert fp["sha256"] == sha and fp["n_rows"] == rows == 500


def test_stratified_sample(anchors):
    s = core.stratified_sample(anchors, by="strata", n=25, seed=1)
    assert len(s) == 25 and Counter(r["strata"] for r in s) == {k: 5 for k in STRATA}
    s2 = core.stratified_sample(anchors, by="difficulty", n=7, seed=1)
    assert len(s2) == 7 and max(Counter(r["difficulty"] for r in s2).values()) <= 3
    assert core.stratified_sample(anchors, n=25, seed=1) == s


# ---------------------------------------------------------------- kappa


def test_kappa_computations(anchors):
    h = core.human_agreement(anchors)
    assert h["kappa"] == pytest.approx(
        cohen_kappa([r["annotator_1"] for r in anchors], [r["annotator_2"] for r in anchors])
    )
    assert h["band"] in {"moderate", "substantial", "almost perfect"}
    gold = [r["gold"] for r in anchors]
    assert cohen_kappa(gold, gold) == 1.0
    ag = core.agreement(anchors, gold)
    assert (
        ag["accuracy"] == 1.0
        and ag["kappa_vs_gold"] == 1.0
        and ag["kappa_vs_gold_band"] == "almost perfect"
    )
    assert ag["human_kappa"] == pytest.approx(h["kappa"])
    assert set(ag["per_strata"]) == set(STRATA)


# ---------------------------------------------------------------- bias detection


def test_position_bias_detected_only_when_injected(biased, unbiased):
    assert biased.position["value"] > 0.2
    assert abs(unbiased.position["value"]) < 0.05
    assert biased.position["ci"][0] > 0.2 and unbiased.position["ci"][1] < 0.1
    assert (
        biased.position["verdict"] == "needs debiasing"
        and unbiased.position["verdict"] == "acceptable"
    )
    assert unbiased.position["consistency_rate"] > biased.position["consistency_rate"]


def test_verbosity_bias_detected(biased, unbiased):
    assert biased.verbosity["value"] > 0.1 and biased.verbosity["ci"][0] > 0.05
    assert abs(unbiased.verbosity["value"]) < 0.05
    assert (
        biased.verbosity["conditional_rate"] > 0.25 and unbiased.verbosity["conditional_rate"] < 0.1
    )
    assert (
        biased.verbosity["verdict"] == "needs debiasing"
        and unbiased.verbosity["verdict"] == "acceptable"
    )
    # the biased judge's longer-win-rate exceeds gold's in the bucketed view
    b = [x for x in biased.verbosity["by_length_ratio"] if x["n"] >= 20]
    assert all(x["judge_longer_win_rate"] > x["gold_longer_win_rate"] for x in b)


def test_self_preference_detected(biased, unbiased):
    assert biased.self_preference["value"] > 0.1
    assert abs(unbiased.self_preference["value"]) < 0.05
    assert biased.self_preference["verdict"] == "needs debiasing"
    assert unbiased.self_preference["verdict"] == "acceptable"


def test_each_bias_is_independently_detectable(anchors):
    sample = core.stratified_sample(anchors, n=250, seed=0)
    for kw, key in [
        ({"position_bias": 0.3}, "position"),
        ({"verbosity_bias": 0.3}, "verbosity"),
        ({"self_preference": 0.3}, "self_preference"),
    ]:
        res = core.calibrate(core.PairwiseJudge(core.BiasedMockJudge(sample, **kw)), sample)
        assert res.flagged == [key], (kw, res.flagged)


def test_swap_debiasing_reduces_position_bias(anchors):
    raw = core.PairwiseJudge(core.make_mock_judge(anchors, "biased"))
    r_raw = core.calibrate(raw, anchors)
    r_deb = core.calibrate(core.SwapDebiasedJudge(raw), anchors)
    assert r_deb.position["value"] < r_raw.position["value"]
    assert abs(r_deb.position["value"]) < 0.05 and r_deb.position["consistency_rate"] == 1.0
    r = anchors[0]
    v = core.debias_by_swapping(raw, r["prompt"], r["response_a"], r["response_b"])
    assert v.label in {"A", "B", "tie"}


def test_unbiased_judge_tracks_gold_and_conclusion(unbiased, biased):
    assert unbiased.agreement["accuracy"] > 0.85
    assert unbiased.agreement["kappa_vs_gold"] > 0.75
    assert unbiased.flagged == [] and unbiased.usable_for().startswith("USABLE")
    assert biased.usable_for().startswith("USABLE ONLY") or biased.usable_for().startswith(
        "NOT usable"
    )
    md = biased.to_markdown()
    assert "needs debiasing" in md and "human ceiling" in md
    d = biased.to_dict()
    assert set(d["flagged"]) >= {"position", "verbosity", "self_preference"}


# ---------------------------------------------------------------- judges & parsing


def test_pairwise_and_pointwise_parsing():
    client = MockLLM(
        handlers=[(r"### Response B", "B is more complete.\nVERDICT: B"), (r"SCORE:", "SCORE: 4")]
    )
    assert core.PairwiseJudge(client).judge("p", "a", "b").label == "B"
    assert core.PointwiseJudge(client).score("p", "a") == 4
    assert core.parse_pairwise("I think it's a tie").label == "tie"
    assert core.parse_pairwise("garbage").parsed is False
    assert core.parse_score("I'd give it 2/5") == (2, True)


def test_cli_end_to_end(tmp_path):
    out = tmp_path / "a.jsonl"
    assert main(["make-anchors", "--n", "50", "--seed", "1", "--out", str(out)]) == 0
    assert len(read_jsonl(out)) == 50 and (tmp_path / "CHANGELOG.md").exists()
    assert (
        main(
            [
                "calibrate",
                "--anchors",
                str(out),
                "--judge",
                "unbiased",
                "--json",
                str(tmp_path / "r.json"),
            ]
        )
        == 0
    )
    assert main(["calibrate", "--anchors", str(out), "--judge", "biased"]) == 1
    assert main(["kappa", "--anchors", str(out)]) == 0
    assert main(["bogus"]) == 2
