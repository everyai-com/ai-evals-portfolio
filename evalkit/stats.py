"""Statistical rigor for eval deltas: bootstrap, paired tests, effect size, sample size."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class DeltaReport:
    metric: str
    n: int
    a_mean: float
    b_mean: float
    delta: float            # b - a
    ci_lo: float
    ci_hi: float
    half_width: float
    p_value: float
    effect_size: float      # Cohen's h for proportions, Cohen's d otherwise
    winner: str             # "A", "B" or "tie"
    significant: bool
    seed: int
    n_boot: int

    def headline(self) -> str:
        sign = "+" if self.delta >= 0 else "-"
        pct = abs(self.delta) * 100
        hw = self.half_width * 100
        p = "p<0.05" if self.p_value < 0.05 else f"p={self.p_value:.2f}"
        if self.winner == "tie":
            return (f"No significant difference on {self.metric}: Δ={sign}{pct:.1f}% ± {hw:.1f}% "
                    f"({p}, n={self.n})")
        return (f"Model {self.winner} wins on {self.metric} by {pct:.1f}% ± {hw:.1f}% "
                f"({p}, n={self.n})")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["headline"] = self.headline()
        return d


def cohens_h(p1: float, p2: float) -> float:
    return 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((b.mean() - a.mean()) / pooled) if pooled else 0.0


def paired_bootstrap(a: Sequence[float], b: Sequence[float], metric: str = "task_success",
                     n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0) -> DeltaReport:
    """Paired bootstrap over per-example scores from two systems on the SAME examples.

    p-value is the two-sided bootstrap p for H0: delta == 0 (fraction of resamples where the
    sign flips, doubled). Pairing removes example-difficulty variance, which is why a paired
    test on 200 examples can detect what an unpaired one needs 2000 for.
    """
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired comparison needs equal-length score arrays")
    n = a_arr.size
    if n == 0:
        raise ValueError("no samples")
    diff = b_arr - a_arr
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    delta = float(diff.mean())
    # two-sided p: proportion of bootstrap deltas on the other side of zero
    if delta >= 0:
        p = 2 * float((boots <= 0).mean())
    else:
        p = 2 * float((boots >= 0).mean())
    p = min(1.0, max(p, 1.0 / n_boot))
    binary = set(np.unique(a_arr)) | set(np.unique(b_arr)) <= {0.0, 1.0}
    es = cohens_h(a_arr.mean(), b_arr.mean()) if binary else cohens_d(a_arr, b_arr)
    significant = p < alpha
    winner = "tie" if not significant else ("B" if delta > 0 else "A")
    return DeltaReport(metric=metric, n=n, a_mean=float(a_arr.mean()), b_mean=float(b_arr.mean()),
                       delta=delta, ci_lo=float(lo), ci_hi=float(hi),
                       half_width=float((hi - lo) / 2), p_value=p, effect_size=float(es),
                       winner=winner, significant=significant, seed=seed, n_boot=n_boot)


def permutation_test(a: Sequence[float], b: Sequence[float], n_perm: int = 10_000,
                     seed: int = 0) -> float:
    """Paired sign-flip permutation test p-value (two-sided) for mean(b - a) == 0."""
    diff = np.asarray(b, float) - np.asarray(a, float)
    n = diff.size
    if n == 0:
        return 1.0
    observed = abs(diff.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    perm = np.abs((diff * signs).mean(axis=1))
    return float((np.sum(perm >= observed) + 1) / (n_perm + 1))


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> dict:
    """Exact McNemar test for paired binary outcomes. Returns discordant counts and p."""
    a_arr, b_arr = np.asarray(a, int), np.asarray(b, int)
    b01 = int(((a_arr == 0) & (b_arr == 1)).sum())  # B fixed what A missed
    b10 = int(((a_arr == 1) & (b_arr == 0)).sum())  # B broke what A had
    n_disc = b01 + b10
    if n_disc == 0:
        return {"b_fixes": b01, "b_breaks": b10, "p_value": 1.0}
    k = min(b01, b10)
    p = sum(math.comb(n_disc, i) for i in range(k + 1)) / 2 ** n_disc
    return {"b_fixes": b01, "b_breaks": b10, "p_value": min(1.0, 2 * p)}


def required_sample_size(baseline: float, min_detectable_delta: float, power: float = 0.8,
                         alpha: float = 0.05) -> int:
    """Samples per arm to detect a delta in a proportion (two-sided, normal approx)."""
    z_a = _z(1 - alpha / 2)
    z_b = _z(power)
    p1, p2 = baseline, min(1.0, max(0.0, baseline + min_detectable_delta))
    pbar = (p1 + p2) / 2
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar)) + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(math.ceil(num / (p1 - p2) ** 2)) if p1 != p2 else 0


def seed_variance(runs: Sequence[Sequence[float]]) -> dict:
    """Variance across repeated runs (different seeds) of the same system."""
    means = np.array([np.mean(r) for r in runs])
    return {"n_runs": len(runs), "mean": float(means.mean()), "std": float(means.std(ddof=1)) if len(runs) > 1 else 0.0,
            "min": float(means.min()), "max": float(means.max()), "range": float(means.max() - means.min())}


def _z(q: float) -> float:
    """Inverse standard normal CDF (Acklam approximation)."""
    if not 0 < q < 1:
        raise ValueError("q must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if q < plow:
        ql = math.sqrt(-2 * math.log(q))
        return (((((c[0] * ql + c[1]) * ql + c[2]) * ql + c[3]) * ql + c[4]) * ql + c[5]) / \
               ((((d[0] * ql + d[1]) * ql + d[2]) * ql + d[3]) * ql + 1)
    if q > phigh:
        ql = math.sqrt(-2 * math.log(1 - q))
        return -(((((c[0] * ql + c[1]) * ql + c[2]) * ql + c[3]) * ql + c[4]) * ql + c[5]) / \
               ((((d[0] * ql + d[1]) * ql + d[2]) * ql + d[3]) * ql + 1)
    ql = q - 0.5
    r = ql * ql
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * ql / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
