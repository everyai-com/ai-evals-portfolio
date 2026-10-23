import numpy as np
import pytest

from evalkit.metrics import (cohen_kappa, fleiss_kappa, precision_recall_f1, wilson_ci, bootstrap_ci,
                             mrr, ndcg_at_k, hit_rate, recall_at_k, interpret_kappa)
from evalkit.stats import paired_bootstrap, permutation_test, mcnemar_exact, required_sample_size, _z


def test_prf1_basic():
    m = precision_recall_f1([1, 1, 0, 0], [1, 0, 1, 0])
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5 and m["accuracy"] == 0.5


def test_kappa_perfect_and_chance():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    # two raters who agree exactly at chance -> kappa ~ 0
    a = ["y", "y", "n", "n"]
    b = ["y", "n", "y", "n"]
    assert abs(cohen_kappa(a, b)) < 1e-9
    assert interpret_kappa(0.75) == "substantial"


def test_fleiss_matches_known_value():
    # 3 raters, 4 items, full agreement on all -> 1.0
    assert fleiss_kappa([["a", "a", "a"], ["b", "b", "b"], ["a", "a", "a"], ["b", "b", "b"]]) == 1.0


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(90, 100)
    assert 0.82 < lo < 0.90 < hi < 0.95


def test_bootstrap_ci_contains_mean():
    pt, lo, hi = bootstrap_ci([0, 1, 1, 1, 0, 1, 1, 0, 1, 1], seed=1)
    assert lo <= pt <= hi


def test_ir_metrics():
    retrieved = [["d1", "d2", "d3"], ["d4", "d5"]]
    relevant = [{"d2"}, {"d9"}]
    assert mrr(retrieved, relevant) == pytest.approx(0.25)
    assert hit_rate(retrieved, relevant, k=2) == 0.5
    assert recall_at_k(retrieved, relevant, 3) == 0.5
    graded = [{"d2": 3, "d1": 1}, {"d9": 2}]
    assert 0 < ndcg_at_k(retrieved, graded, 3) < 1


def test_paired_bootstrap_detects_real_delta():
    rng = np.random.default_rng(0)
    a = (rng.random(400) < 0.70).astype(float)
    b = np.clip(a + (rng.random(400) < 0.12), 0, 1)  # B fixes ~12% of cases
    rep = paired_bootstrap(a, b, n_boot=2000, seed=0)
    assert rep.winner == "B" and rep.significant and rep.ci_lo > 0
    assert "Model B wins" in rep.headline()


def test_paired_bootstrap_noise_is_tie():
    rng = np.random.default_rng(1)
    a = (rng.random(50) < 0.7).astype(float)
    b = (rng.random(50) < 0.7).astype(float)
    rep = paired_bootstrap(a, b, n_boot=2000, seed=0)
    assert rep.winner == "tie" or rep.half_width > 0.05


def test_permutation_and_mcnemar():
    a = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    b = [1, 1, 1, 1, 1, 0, 1, 1, 0, 1]
    assert permutation_test(a, b, n_perm=2000) < 0.1
    m = mcnemar_exact(a, b)
    assert m["b_fixes"] == 5 and m["b_breaks"] == 0


def test_sample_size_and_z():
    assert abs(_z(0.975) - 1.95996) < 1e-3
    n = required_sample_size(0.70, 0.02)
    assert 6000 < n < 10000  # 2 points on a 70% baseline needs thousands of samples
