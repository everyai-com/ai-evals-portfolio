"""Tests prove the README claims: headline format, power warnings, Holm, pairing, SVG, seeds."""

import re
from pathlib import Path

import numpy as np
import pytest

from evalkit.io import read_jsonl, write_jsonl
from projects.p07_significance_engine import core
from projects.p07_significance_engine.cli import main

DATA = Path(__file__).resolve().parents[1] / "data"
HEADLINE = re.compile(r"Model B wins by \d+\.\d% ± \d+\.\d% \(p<0\.05, n=400\)")


@pytest.fixture(scope="module")
def big():
    return core.load_pair(DATA / "model_a.jsonl", DATA / "model_b.jsonl", "success")


@pytest.fixture(scope="module")
def big_result(big):
    return core.compare_paired(big, n_boot=4000, keep_boots=True)


def test_n400_is_significant_and_headline_matches(big_result):
    r = big_result
    assert r.report.significant and r.report.winner == "B"
    assert HEADLINE.fullmatch(r.headline()), r.headline()
    assert r.power_warning is None
    assert r.binary and r.mcnemar["p_value"] < 0.05 and r.permutation_p < 0.05
    assert r.effect_band in {"negligible", "small"}
    assert r.required_n_paired <= 400 < r.required_n_unpaired
    assert r.strata and set(r.strata["per_strata"]) == {"qa", "code", "math", "summarize"}
    assert "Model B wins by" in r.to_markdown()


def test_n50_same_effect_is_noise_with_power_warning():
    data = core.load_pair(DATA / "small_a.jsonl", DATA / "small_b.jsonl", "success")
    r = core.compare_paired(data, n_boot=4000)
    assert not r.report.significant and r.report.n == 50
    assert r.power_warning and "noise" in r.power_warning
    assert abs(r.report.delta) < r.mde and r.power < 0.8
    assert r.required_n_paired > 50
    assert r.headline().startswith("No significant difference") and "underpowered" in r.headline()


def test_power_warning_on_significant_but_underpowered():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 60)
    b = a + 0.3 + rng.normal(0, 0.9, 60)
    r = core.compare(a, b, "score", n_boot=2000)
    # delta is small relative to sd(diff)/sqrt(n): either not significant+noise, or significant+underpowered
    assert r.power_warning is not None
    assert ("underpowered" in r.power_warning) == r.report.significant


def test_holm_reduces_false_positives_on_null_candidates():
    rng = np.random.default_rng(7)
    n = 200
    base = [{"id": f"e{i}", "success": int(rng.random() < 0.7)} for i in range(n)]
    cands = {}
    for k in range(10):
        rows = []
        for r in base:
            u = rng.random()
            succ = r["success"] if u < 0.8 else int(rng.random() < 0.7)  # same 0.7 rate: null
            rows.append({"id": r["id"], "success": succ})
        cands[f"c{k}"] = rows
    res = core.multi_compare(base, cands, "success", n_boot=2000)
    d = res.to_dict()
    assert d["holm_rejections"] <= d["raw_rejections"] and d["holm_rejections"] == 0
    # pure p-value check: 5 raw rejections, 0 after Holm
    adj = core.holm_bonferroni([0.01, 0.02, 0.03, 0.04, 0.05, 0.3, 0.4, 0.5, 0.6, 0.7])
    assert sum(p["reject"] for p in adj) == 0 and adj[0]["p_adj"] == pytest.approx(0.1)
    adj2 = core.holm_bonferroni([0.001, 0.2, 0.9])
    assert [p["reject"] for p in adj2] == [True, False, False]
    assert adj2[1]["p_adj"] == pytest.approx(0.4) and adj2[2]["p_adj"] == pytest.approx(0.9)
    assert "Holm" in res.to_markdown()


def test_unpaired_ids_warn_or_refuse():
    a = [{"id": f"x{i}", "score": 0.5} for i in range(10)]
    b = [{"id": f"x{i}", "score": 0.6} for i in range(3, 13)]
    d = core.pair_results(a, b, "score")
    assert d.n == 7 and len(d.unpaired_a) == 3 and len(d.unpaired_b) == 3 and d.warnings
    with pytest.raises(ValueError, match="unpaired"):
        core.pair_results(a, b, "score", strict=True)
    with pytest.raises(ValueError, match="duplicate"):
        core.pair_results(a + a[:1], b, "score")
    with pytest.raises(ValueError, match="no ids in common"):
        core.pair_results(a, [{"id": "zz", "score": 1}], "score")


def test_formats_and_metric_detection():
    assert core.detect_format([{"id": 1, "success": 1}]) == "per_example"
    assert core.detect_format([{"id": 1, "a": 0.1, "b": 0.2}]) == "paired"
    assert core.detect_format([{"id": 1, "model": "m", "seed": 0, "score": 1}]) == "runs"
    with pytest.raises(ValueError):
        core.detect_format([{"foo": 1}])
    d = core.pair_from_single([{"id": "p", "a": 0.1, "b": 0.3, "tags": ["t"]}] * 3)
    assert d.n == 3 and d.strata == ["t", "t", "t"]
    assert core.pick_metric([{"id": 1, "success": 1}], None) == "success"


def test_svg_output(big_result, tmp_path):
    svg = core.svg_histogram(
        big_result.boots, big_result.report.ci_lo, big_result.report.ci_hi, big_result.report.delta
    )
    assert svg.startswith("<svg") and "</svg>" in svg and svg.count("<rect") > 20
    out = tmp_path / "o.svg"
    rc = main(
        [
            "compare",
            "--a",
            str(DATA / "model_a.jsonl"),
            "--b",
            str(DATA / "model_b.jsonl"),
            "--metric",
            "success",
            "--n-boot",
            "1000",
            "--plot",
            str(out),
            "--json",
            str(tmp_path / "r.json"),
        ]
    )
    assert rc == 0 and out.read_text().startswith("<svg") and (tmp_path / "r.json").exists()


def test_bootstrap_distribution_matches_evalkit(big):
    boots = core.bootstrap_distribution(big.a, big.b, n_boot=500, seed=3)
    rep = core.paired_bootstrap(big.a, big.b, n_boot=500, seed=3)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    assert lo == pytest.approx(rep.ci_lo) and hi == pytest.approx(rep.ci_hi)


def test_seed_variance_warning_triggers_and_clears():
    rng = np.random.default_rng(0)
    rows = []
    for model, mean in (("m_a", 0.70), ("m_b", 0.72)):
        for seed in range(3):
            off = rng.normal(0, 0.06)  # huge seed noise vs a 2-point delta
            rows += [
                {
                    "id": f"e{i}",
                    "model": model,
                    "seed": seed,
                    "score": float(np.clip(mean + off + rng.normal(0, 0.02), 0, 1)),
                }
                for i in range(100)
            ]
    s = core.seed_analysis(rows, "score")
    assert s["any_warning"] and s["comparisons"][0]["warning"]
    quiet = [{**r, "score": 0.70 if r["model"] == "m_a" else 0.80} for r in rows]
    s2 = core.seed_analysis(quiet, "score")
    assert not s2["any_warning"] and s2["models"]["m_a"]["std"] == 0.0
    # demo fixture: 3 seeds x 2 models, warning on success
    demo = core.seed_analysis(read_jsonl(DATA / "runs.jsonl"), "success")
    assert demo["models"]["model_a"]["n_runs"] == 3 and "WARNING" in core.seeds_markdown(demo)
    assert main(["seeds", "--runs", str(DATA / "runs.jsonl"), "--metric", "success"]) == 1


def test_strata_simpson_check():
    # micro winner B (big stratum carries it) but macro winner A: paradox in paired form
    a = np.array([0.0] * 90 + [1.0] * 10 + [1.0] * 10)
    b = np.array([0.0] * 80 + [1.0] * 20 + [1.0] * 5 + [0.0] * 5)
    strata = ["big"] * 100 + ["small"] * 10
    s = core.strata_breakdown(a, b, strata, "success", n_boot=500)
    assert s["overall_direction"] == "B" and s["conflicting_strata"] == ["small"]
    assert s["simpson_paradox"] and s["macro_direction"] == "A"
    assert s["note"] and "Simpson" in s["note"]
    a2 = np.array([0.0] * 50 + [1.0] * 50 + [0.0] * 50 + [1.0] * 50)
    b2 = np.array([0.0] * 45 + [1.0] * 55 + [0.0] * 40 + [1.0] * 60)
    s2 = core.strata_breakdown(a2, b2, ["x"] * 100 + ["y"] * 100, "success", n_boot=500)
    assert not s2["conflicting_strata"] and not s2["simpson_paradox"]


def test_effect_bands_and_mde():
    assert core.effect_band(0.1) == "negligible" and core.effect_band(0.3) == "small"
    assert core.effect_band(0.6) == "medium" and core.effect_band(-1.0) == "large"
    assert core.paired_mde(0.2, 400) == pytest.approx((1.96 + 0.8416) * 0.2 / 20, rel=1e-3)
    assert core.paired_required_n(0.2, 0.03) > core.paired_required_n(0.2, 0.05)
    assert 0 < core.post_hoc_power(0.2, 0.03, 400) < 1


def test_cli_power_multi_and_errors(tmp_path):
    assert main(["power", "--baseline", "0.7", "--delta", "0.02"]) == 0
    c2 = tmp_path / "model_c.jsonl"
    write_jsonl(
        c2,
        [
            {**r, "success": r["success"] if i % 7 else 1 - r["success"]}
            for i, r in enumerate(read_jsonl(DATA / "model_a.jsonl"))
        ],
    )
    assert (
        main(
            [
                "multi",
                "--baseline",
                str(DATA / "model_a.jsonl"),
                "--candidates",
                str(DATA / "model_b.jsonl"),
                str(c2),
                "--metric",
                "success",
                "--n-boot",
                "500",
                "--json",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "multi",
                "--baseline",
                str(DATA / "model_a.jsonl"),
                "--candidates",
                str(DATA / "model_b.jsonl"),
                "--strict",
                "--n-boot",
                "500",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "compare",
                "--a",
                str(DATA / "model_a.jsonl"),
                "--b",
                str(DATA / "small_b.jsonl"),
                "--strict",
            ]
        )
        == 2
    )
    assert main(["compare", "--a", str(DATA / "runs.jsonl")]) == 2
    p = tmp_path / "paired.jsonl"
    write_jsonl(
        p, [{"id": i, "a": 0.4 + 0.01 * (i % 3), "b": 0.5 + 0.01 * (i % 5)} for i in range(60)]
    )
    assert main(["compare", "--a", str(p), "--n-boot", "500"]) in (0, 1)
    assert main(["nope"]) == 2
