"""Statistical significance engine: the product around evalkit.stats.

Turns per-example result files into "Model B wins by 3.2% ± 1.1% (p<0.05, n=400)" with the
supporting evidence a reviewer needs: paired bootstrap CI, McNemar (binary), permutation
cross-check, effect-size band, minimum detectable effect at this n and a power warning,
Holm-Bonferroni across many comparisons, per-strata breakdown with a Simpson's check, and
seed-variance analysis for multi-seed runs. The math lives in evalkit.stats; this module
loads, pairs, interprets and renders.

Formulas (see also evalkit.stats docstrings)
-------------------------------------------
paired bootstrap    resample example indices with replacement, delta_b = mean(b - a) on each
                    resample; CI = percentiles alpha/2, 1-alpha/2; p = 2 * P(sign flips).
permutation         sign-flip test on per-example differences (exchangeable under H0).
McNemar (exact)     binomial test on the discordant pairs (A right/B wrong vs the reverse).
effect size         Cohen's h = 2 asin(sqrt(p_b)) - 2 asin(sqrt(p_a)) for proportions,
                    Cohen's d = (mean_b - mean_a) / pooled sd otherwise. Bands (Cohen 1988):
                    |es| < 0.2 negligible, < 0.5 small, < 0.8 medium, else large.
MDE (paired)        smallest detectable delta at n with power 1-beta:
                    (z_{1-alpha/2} + z_{1-beta}) * sd(b - a) / sqrt(n).
required n (paired) ((z_{1-alpha/2} + z_{1-beta}) * sd(b - a) / delta)^2.
required n (two-arm proportions, unpaired) evalkit.stats.required_sample_size.
Holm-Bonferroni     sort p ascending; adj_p_(i) = max_{j<=i} min(1, (m - j + 1) p_(j)).
Simpson's check     paired data -> the overall delta is the size-weighted mean of per-strata
                    deltas, so it can never fully reverse; we flag strata whose sign disagrees
                    with the overall and a disagreement between the micro (size-weighted) and
                    macro (equal-weight) winners, which is the paradox in its paired form.
seed variance       mean/std of run means per model (evalkit.stats.seed_variance); warn when
                    |delta between model means| < max within-model std.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from evalkit.io import read_jsonl
from evalkit.report import md_table
from evalkit.stats import (
    DeltaReport,
    _z,
    mcnemar_exact,
    paired_bootstrap,
    permutation_test,
    required_sample_size,
    seed_variance,
)

METRIC_CANDIDATES = ("score", "success", "correct", "value")


# ---------------------------------------------------------------- loading & pairing


def detect_format(rows: Sequence[dict]) -> str:
    """'runs' ({id, model, seed, ...}), 'paired' ({id, a, b}) or 'per_example' ({id, score|success})."""
    if not rows:
        raise ValueError("empty results file")
    keys = set(rows[0])
    if {"model", "seed"} <= keys:
        return "runs"
    if {"a", "b"} <= keys:
        return "paired"
    if "id" in keys and any(k in keys for k in METRIC_CANDIDATES):
        return "per_example"
    raise ValueError(f"unrecognised results format; keys={sorted(keys)}")


def pick_metric(rows: Sequence[dict], metric: str | None) -> str:
    if metric:
        if metric not in rows[0]:
            raise ValueError(f"metric {metric!r} not in rows; have {sorted(rows[0])}")
        return metric
    for k in METRIC_CANDIDATES:
        if k in rows[0]:
            return k
    raise ValueError("no metric field found; pass metric=")


def _strata_of(row: dict, field_name: str) -> str | None:
    v = row.get(field_name)
    if v is None and field_name == "strata":
        v = row.get("tags")
    if isinstance(v, (list, tuple)):
        return str(v[0]) if v else None
    return None if v is None else str(v)


@dataclass
class PairedData:
    ids: list[str]
    a: np.ndarray
    b: np.ndarray
    metric: str
    strata: list[str | None] = field(default_factory=list)
    unpaired_a: list[str] = field(default_factory=list)
    unpaired_b: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def warnings(self) -> list[str]:
        out = []
        if self.unpaired_a or self.unpaired_b:
            out.append(
                f"{len(self.unpaired_a)} ids only in A and {len(self.unpaired_b)} only in B were "
                f"dropped; comparing the {self.n} paired ids (pass strict=True to refuse)"
            )
        return out


def pair_results(
    rows_a: Sequence[dict],
    rows_b: Sequence[dict],
    metric: str | None = None,
    strict: bool = False,
    strata_field: str = "strata",
) -> PairedData:
    """Join two per-example result lists on `id`. Unpaired ids are reported and dropped, or
    refused with ValueError when strict=True. Duplicate ids are refused."""
    metric = pick_metric(rows_a, metric)
    da = {r["id"]: r for r in rows_a}
    db = {r["id"]: r for r in rows_b}
    if len(da) != len(rows_a) or len(db) != len(rows_b):
        raise ValueError("duplicate ids in a results file")
    common = [i for i in da if i in db]
    only_a, only_b = [i for i in da if i not in db], [i for i in db if i not in da]
    if strict and (only_a or only_b):
        raise ValueError(
            f"unpaired ids: {len(only_a)} only in A (e.g. {only_a[:3]}), "
            f"{len(only_b)} only in B (e.g. {only_b[:3]})"
        )
    if not common:
        raise ValueError("no ids in common between A and B")
    return PairedData(
        ids=common,
        a=np.array([float(da[i][metric]) for i in common]),
        b=np.array([float(db[i][metric]) for i in common]),
        metric=metric,
        strata=[_strata_of(da[i], strata_field) or _strata_of(db[i], strata_field) for i in common],
        unpaired_a=only_a,
        unpaired_b=only_b,
    )


def pair_from_single(
    rows: Sequence[dict], metric: str = "score", strata_field: str = "strata"
) -> PairedData:
    """One file with {id, a, b} per example."""
    return PairedData(
        ids=[r["id"] for r in rows],
        a=np.array([float(r["a"]) for r in rows]),
        b=np.array([float(r["b"]) for r in rows]),
        metric=metric,
        strata=[_strata_of(r, strata_field) for r in rows],
    )


def load_pair(
    path_a: str | Path,
    path_b: str | Path | None = None,
    metric: str | None = None,
    strict: bool = False,
    strata_field: str = "strata",
) -> PairedData:
    rows_a = read_jsonl(path_a)
    fmt = detect_format(rows_a)
    if fmt == "paired":
        return pair_from_single(rows_a, metric or "score", strata_field)
    if fmt == "runs":
        raise ValueError("runs format: use `seeds --runs` (or split by model/seed first)")
    if path_b is None:
        raise ValueError("per-example format needs two files (--a and --b)")
    return pair_results(rows_a, read_jsonl(path_b), metric, strict, strata_field)


# ---------------------------------------------------------------- effect size, power


def effect_band(es: float) -> str:
    """Cohen (1988) bands for |h| or |d|."""
    a = abs(es)
    return "negligible" if a < 0.2 else "small" if a < 0.5 else "medium" if a < 0.8 else "large"


def is_binary(a: np.ndarray, b: np.ndarray) -> bool:
    return set(np.unique(a)) | set(np.unique(b)) <= {0.0, 1.0}


def paired_mde(sd_diff: float, n: int, power: float = 0.8, alpha: float = 0.05) -> float:
    """Minimum detectable paired delta: (z_{1-a/2} + z_power) * sd_diff / sqrt(n)."""
    if n <= 0:
        return float("inf")
    return (_z(1 - alpha / 2) + _z(power)) * sd_diff / math.sqrt(n)


def paired_required_n(sd_diff: float, delta: float, power: float = 0.8, alpha: float = 0.05) -> int:
    """Samples needed to detect `delta` with a paired test: ((z_a + z_b) sd_diff / delta)^2."""
    if delta == 0:
        return 0
    return int(math.ceil(((_z(1 - alpha / 2) + _z(power)) * sd_diff / delta) ** 2))


def post_hoc_power(sd_diff: float, delta: float, n: int, alpha: float = 0.05) -> float:
    """Power to detect |delta| at n (normal approx): Phi(|delta| sqrt(n) / sd - z_{1-a/2})."""
    if sd_diff == 0 or n == 0:
        return 1.0 if delta else 0.0
    z = abs(delta) * math.sqrt(n) / sd_diff - _z(1 - alpha / 2)
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def bootstrap_distribution(
    a: Sequence[float], b: Sequence[float], n_boot: int = 10_000, seed: int = 0
) -> np.ndarray:
    """The bootstrap deltas behind paired_bootstrap (same RNG stream, so identical draws)."""
    diff = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    return diff[idx].mean(axis=1)


# ---------------------------------------------------------------- single comparison


@dataclass
class ComparisonResult:
    report: DeltaReport
    binary: bool
    mcnemar: dict | None
    permutation_p: float
    effect_band: str
    sd_diff: float
    mde: float
    power: float
    required_n_paired: int
    required_n_unpaired: int | None
    power_warning: str | None
    warnings: list[str] = field(default_factory=list)
    strata: dict | None = None
    boots: np.ndarray | None = None
    label_a: str = "A"
    label_b: str = "B"

    def headline(self) -> str:
        return executive_headline(
            self.report, self.power_warning is not None, self.required_n_paired
        )

    def to_dict(self) -> dict:
        d = {
            "headline": self.headline(),
            "report": self.report.to_dict(),
            "binary": self.binary,
            "mcnemar": self.mcnemar,
            "permutation_p": self.permutation_p,
            "effect_band": self.effect_band,
            "sd_diff": self.sd_diff,
            "mde": self.mde,
            "post_hoc_power": self.power,
            "required_n_paired": self.required_n_paired,
            "required_n_unpaired": self.required_n_unpaired,
            "power_warning": self.power_warning,
            "warnings": self.warnings,
            "strata": self.strata,
            "labels": {"A": self.label_a, "B": self.label_b},
        }
        return d

    def to_markdown(self) -> str:
        r = self.report
        rows = [
            ["metric", r.metric],
            ["n (paired)", r.n],
            [f"A = {self.label_a} mean", f"{r.a_mean:.4f}"],
            [f"B = {self.label_b} mean", f"{r.b_mean:.4f}"],
            ["delta (B - A)", f"{r.delta:+.4f}"],
            [f"95% bootstrap CI ({r.n_boot} resamples)", f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]"],
            ["bootstrap p (two-sided)", f"{r.p_value:.4f}"],
            ["permutation p (cross-check)", f"{self.permutation_p:.4f}"],
        ]
        if self.mcnemar:
            m = self.mcnemar
            rows.append(
                [
                    "McNemar exact p (B fixes / B breaks)",
                    f"{m['p_value']:.4f} ({m['b_fixes']} / {m['b_breaks']})",
                ]
            )
        rows += [
            [
                f"effect size (Cohen's {'h' if self.binary else 'd'})",
                f"{r.effect_size:+.3f} ({self.effect_band})",
            ],
            ["sd of per-example difference", f"{self.sd_diff:.4f}"],
            ["minimum detectable effect at this n (power 0.8)", f"{self.mde:.4f}"],
            ["post-hoc power for the observed delta", f"{self.power:.2f}"],
            ["required n (paired) for this delta", self.required_n_paired],
        ]
        if self.required_n_unpaired is not None:
            rows.append(
                [
                    "required n per arm (unpaired proportions, evalkit.required_sample_size)",
                    self.required_n_unpaired,
                ]
            )
        rows.append(
            ["verdict", f"{r.winner} {'(significant)' if r.significant else '(not significant)'}"]
        )
        out = [f"**{self.headline()}**", "", md_table(["quantity", "value"], rows)]
        for w in self.warnings:
            out.append(f"\n> WARNING: {w}")
        if self.power_warning:
            out.append(f"\n> POWER WARNING: {self.power_warning}")
        if self.strata:
            out.append("\n" + strata_markdown(self.strata))
        return "\n".join(out)


def executive_headline(
    r: DeltaReport, underpowered: bool = False, required_n: int | None = None
) -> str:
    """Exactly the post's format: 'Model B wins by 3.2% ± 1.1% (p<0.05, n=400)'."""
    pct, hw = abs(r.delta) * 100, r.half_width * 100
    if r.significant:
        p = "p<0.05" if r.p_value < 0.05 else f"p<{r.p_value:.2g}"
        return f"Model {r.winner} wins by {pct:.1f}% ± {hw:.1f}% ({p}, n={r.n})"
    sign = "+" if r.delta >= 0 else "-"
    tail = f" - underpowered, need n≈{required_n}" if underpowered and required_n else ""
    return f"No significant difference: Model B {sign}{pct:.1f}% ± {hw:.1f}% (p={r.p_value:.2f}, n={r.n}){tail}"


def compare(
    a: Sequence[float],
    b: Sequence[float],
    metric: str = "score",
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
    power: float = 0.8,
    strata: Sequence[str | None] | None = None,
    keep_boots: bool = False,
    label_a: str = "A",
    label_b: str = "B",
) -> ComparisonResult:
    """Full comparison of paired per-example results (B relative to A)."""
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    rep = paired_bootstrap(a_arr, b_arr, metric=metric, n_boot=n_boot, alpha=alpha, seed=seed)
    binary = is_binary(a_arr, b_arr)
    mc = mcnemar_exact(a_arr.astype(int), b_arr.astype(int)) if binary else None
    perm = permutation_test(a_arr, b_arr, n_perm=min(n_boot, 10_000), seed=seed)
    diff = b_arr - a_arr
    sd = float(diff.std(ddof=1)) if diff.size > 1 else 0.0
    mde = paired_mde(sd, rep.n, power, alpha)
    pw = post_hoc_power(sd, rep.delta, rep.n, alpha)
    req_paired = paired_required_n(sd, rep.delta, power, alpha)
    req_unpaired = (
        required_sample_size(rep.a_mean, rep.delta, power, alpha) if binary and rep.delta else None
    )
    warning = None
    if abs(rep.delta) < mde:
        pts = abs(rep.delta) * 100
        if rep.significant:
            warning = (
                f"significant but underpowered: a {pts:.1f}-point delta on {rep.n} samples has post-hoc "
                f"power {pw:.2f} (< {power}); expect the effect to shrink on replication. "
                f"n≈{req_paired} paired samples would detect it reliably."
            )
        else:
            warning = (
                f"a {pts:.1f}-point delta on {rep.n} samples is noise: the minimum detectable effect at "
                f"n={rep.n} is {mde * 100:.1f} points (power {power}). You need n≈{req_paired} paired "
                f"samples"
                + (f" (or ≈{req_unpaired} per arm unpaired)" if req_unpaired else "")
                + " to call this."
            )
    strata_res = (
        strata_breakdown(
            a_arr, b_arr, strata, metric, n_boot=min(n_boot, 2000), seed=seed, alpha=alpha
        )
        if strata is not None and any(s is not None for s in strata)
        else None
    )
    return ComparisonResult(
        report=rep,
        binary=binary,
        mcnemar=mc,
        permutation_p=perm,
        effect_band=effect_band(rep.effect_size),
        sd_diff=sd,
        mde=mde,
        power=pw,
        required_n_paired=req_paired,
        required_n_unpaired=req_unpaired,
        power_warning=warning,
        strata=strata_res,
        boots=bootstrap_distribution(a_arr, b_arr, n_boot, seed) if keep_boots else None,
        label_a=label_a,
        label_b=label_b,
    )


def compare_paired(data: PairedData, **kw: Any) -> ComparisonResult:
    res = compare(data.a, data.b, metric=data.metric, strata=data.strata, **kw)
    res.warnings = data.warnings + res.warnings
    return res


# ---------------------------------------------------------------- strata & Simpson's check


def strata_breakdown(
    a: np.ndarray,
    b: np.ndarray,
    strata: Sequence[str | None],
    metric: str,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    overall = paired_bootstrap(a, b, metric, n_boot=n_boot, alpha=alpha, seed=seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(strata):
        groups[s if s is not None else "untagged"].append(i)
    per: dict[str, dict] = {}
    for s, idx in sorted(groups.items()):
        if len(idx) < 2:
            continue
        r = paired_bootstrap(a[idx], b[idx], metric, n_boot=n_boot, alpha=alpha, seed=seed)
        per[s] = {
            "n": r.n,
            "a_mean": r.a_mean,
            "b_mean": r.b_mean,
            "delta": r.delta,
            "ci": [r.ci_lo, r.ci_hi],
            "p_value": r.p_value,
            "winner": r.winner,
            "direction": "B" if r.delta > 0 else "A" if r.delta < 0 else "tie",
        }
    overall_dir = "B" if overall.delta > 0 else "A" if overall.delta < 0 else "tie"
    macro = float(np.mean([v["delta"] for v in per.values()])) if per else overall.delta
    macro_dir = "B" if macro > 0 else "A" if macro < 0 else "tie"
    conflicting = [s for s, v in per.items() if v["direction"] not in ("tie", overall_dir)]
    sig_conflicting = [
        s
        for s, v in per.items()
        if v["winner"] not in ("tie", overall.winner) and overall.winner != "tie"
    ]
    paradox = macro_dir != overall_dir and overall_dir != "tie"
    note = None
    if paradox:
        note = (
            f"Simpson's check: the size-weighted (micro) winner is {overall_dir} but the equal-weight "
            f"(macro) winner is {macro_dir}; the overall result is driven by strata mix, not by a uniform gain."
        )
    elif conflicting:
        note = (
            f"Simpson's check: overall direction {overall_dir} but {', '.join(conflicting)} go the other "
            f"way; the gain is not uniform across strata."
        )
    return {
        "overall_direction": overall_dir,
        "overall_winner": overall.winner,
        "micro_delta": overall.delta,
        "macro_delta": macro,
        "macro_direction": macro_dir,
        "per_strata": per,
        "conflicting_strata": conflicting,
        "significantly_conflicting_strata": sig_conflicting,
        "simpson_paradox": paradox,
        "note": note,
    }


def strata_markdown(s: dict) -> str:
    rows = [
        [
            k,
            v["n"],
            f"{v['a_mean']:.3f}",
            f"{v['b_mean']:.3f}",
            f"{v['delta']:+.3f}",
            f"[{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]",
            f"{v['p_value']:.3f}",
            v["winner"],
        ]
        for k, v in s["per_strata"].items()
    ]
    out = [
        "### Per-strata breakdown",
        "",
        md_table(["strata", "n", "A", "B", "delta", "95% CI", "p", "winner"], rows),
        "",
        f"micro delta {s['micro_delta']:+.4f} ({s['overall_direction']}), macro delta {s['macro_delta']:+.4f} ({s['macro_direction']})",
    ]
    if s["note"]:
        out.append(f"\n> {s['note']}")
    return "\n".join(out)


# ---------------------------------------------------------------- multiple comparisons


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> list[dict]:
    """Holm's step-down adjustment. Returns [{p, p_adj, reject}] in input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return [{"p": float(p), "p_adj": float(q), "reject": q <= alpha} for p, q in zip(pvals, adj)]


@dataclass
class MultiResult:
    baseline: str
    comparisons: list[dict]
    alpha: float

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline,
            "alpha": self.alpha,
            "comparisons": self.comparisons,
            "raw_rejections": sum(c["significant_raw"] for c in self.comparisons),
            "holm_rejections": sum(c["significant_holm"] for c in self.comparisons),
        }

    def to_markdown(self) -> str:
        rows = [
            [
                c["name"],
                c["n"],
                f"{c['delta']:+.4f}",
                f"[{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]",
                f"{c['p']:.4f}",
                f"{c['p_adj']:.4f}",
                "yes" if c["significant_raw"] else "no",
                "yes" if c["significant_holm"] else "no",
                c["winner_holm"],
            ]
            for c in self.comparisons
        ]
        d = self.to_dict()
        out = [
            f"# {len(self.comparisons)} candidates vs baseline `{self.baseline}` (Holm-Bonferroni, alpha={self.alpha})",
            "",
            md_table(
                [
                    "candidate",
                    "n",
                    "delta",
                    "95% CI",
                    "p",
                    "p (Holm)",
                    "sig raw",
                    "sig Holm",
                    "winner",
                ],
                rows,
            ),
            "",
            f"{d['raw_rejections']} comparisons significant before correction, {d['holm_rejections']} after.",
        ]
        for c in self.comparisons:
            if c.get("power_warning"):
                out.append(f"\n> {c['name']}: {c['power_warning']}")
        return "\n".join(out)


def multi_compare(
    baseline_rows: Sequence[dict],
    candidates: dict[str, Sequence[dict]],
    metric: str | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
    strict: bool = False,
    baseline_name: str = "baseline",
) -> MultiResult:
    """Compare several candidates against one baseline; Holm-Bonferroni across the family."""
    comps = []
    for name, rows in candidates.items():
        data = pair_results(baseline_rows, rows, metric, strict)
        res = compare(data.a, data.b, metric=data.metric, n_boot=n_boot, seed=seed, alpha=alpha)
        comps.append(
            {
                "name": name,
                "n": res.report.n,
                "delta": res.report.delta,
                "ci": [res.report.ci_lo, res.report.ci_hi],
                "p": res.report.p_value,
                "effect_size": res.report.effect_size,
                "effect_band": res.effect_band,
                "significant_raw": res.report.significant,
                "winner_raw": res.report.winner,
                "power_warning": res.power_warning,
                "unpaired": len(data.unpaired_a) + len(data.unpaired_b),
            }
        )
    for c, h in zip(comps, holm_bonferroni([c["p"] for c in comps], alpha)):
        c["p_adj"], c["significant_holm"] = h["p_adj"], h["reject"]
        c["winner_holm"] = "tie" if not h["reject"] else ("B" if c["delta"] > 0 else "A")
    return MultiResult(baseline=baseline_name, comparisons=comps, alpha=alpha)


# ---------------------------------------------------------------- seeds


def seed_analysis(
    rows: Sequence[dict], metric: str | None = None, baseline_model: str | None = None
) -> dict:
    """Multi-seed runs: mean ± std across seeds per model; warn when the between-model delta
    is smaller than the within-model seed std."""
    metric = pick_metric(rows, metric)
    by: dict[str, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[str(r["model"])][r["seed"]].append(float(r[metric]))
    models = sorted(by)
    if baseline_model is None:
        baseline_model = models[0]
    per_model = {
        m: {
            **seed_variance([by[m][s] for s in sorted(by[m])]),
            "seeds": {str(s): float(np.mean(by[m][s])) for s in sorted(by[m])},
        }
        for m in models
    }
    comparisons = []
    for m in models:
        if m == baseline_model:
            continue
        delta = per_model[m]["mean"] - per_model[baseline_model]["mean"]
        within = max(per_model[m]["std"], per_model[baseline_model]["std"])
        warn = abs(delta) < within
        comparisons.append(
            {
                "model": m,
                "baseline": baseline_model,
                "delta": delta,
                "within_model_std": within,
                "delta_over_std": (abs(delta) / within) if within else float("inf"),
                "warning": warn,
                "note": (
                    f"delta {delta:+.4f} between {m} and {baseline_model} is smaller than the within-model "
                    f"seed std {within:.4f}: a single-seed comparison can flip sign on re-run. Average "
                    f"over seeds (or more seeds) before claiming a winner."
                )
                if warn
                else (
                    f"delta {delta:+.4f} is {abs(delta) / within:.1f}x the within-model seed std {within:.4f}."
                    if within
                    else f"delta {delta:+.4f}; no within-model seed variance observed."
                ),
            }
        )
    return {
        "metric": metric,
        "baseline": baseline_model,
        "models": per_model,
        "comparisons": comparisons,
        "any_warning": any(c["warning"] for c in comparisons),
    }


def seeds_markdown(s: dict) -> str:
    rows = [
        [
            m,
            v["n_runs"],
            f"{v['mean']:.4f}",
            f"{v['std']:.4f}",
            f"{v['min']:.4f}",
            f"{v['max']:.4f}",
        ]
        for m, v in s["models"].items()
    ]
    out = [
        f"# Seed variance on `{s['metric']}`",
        "",
        md_table(["model", "seeds", "mean", "std", "min", "max"], rows),
        "",
    ]
    for c in s["comparisons"]:
        out.append(f"- {c['model']} vs {c['baseline']}: {c['note']}")
        if c["warning"]:
            out[-1] = "- WARNING " + out[-1][2:]
    return "\n".join(out)


# ---------------------------------------------------------------- power table


def power_table(
    baseline: float, deltas: Sequence[float], power: float = 0.8, alpha: float = 0.05
) -> list[dict]:
    return [
        {"delta": d, "required_n_per_arm": required_sample_size(baseline, d, power, alpha)}
        for d in deltas
    ]


# ---------------------------------------------------------------- SVG


def svg_histogram(
    boots: np.ndarray,
    ci_lo: float,
    ci_hi: float,
    delta: float,
    title: str = "bootstrap delta",
    bins: int = 40,
    width: int = 640,
    height: int = 360,
) -> str:
    """Pure-string SVG histogram of the bootstrap distribution with the CI shaded."""
    boots = np.asarray(boots, float)
    counts, edges = np.histogram(boots, bins=bins)
    ml, mr, mt, mb = 50, 20, 40, 50
    pw, ph = width - ml - mr, height - mt - mb
    x0, x1 = float(edges[0]), float(edges[-1])
    if x1 == x0:
        x1 = x0 + 1e-9
    xs = lambda v: ml + (v - x0) / (x1 - x0) * pw
    ymax = max(1, int(counts.max()))
    ys = lambda c: mt + ph - c / ymax * ph
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="24" text-anchor="middle" font-family="sans-serif" font-size="14">{title}</text>',
        f'<rect x="{xs(ci_lo):.1f}" y="{mt}" width="{max(0.0, xs(ci_hi) - xs(ci_lo)):.1f}" height="{ph}" fill="#9ecae1" opacity="0.45"/>',
    ]
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        parts.append(
            f'<rect x="{xs(e0):.1f}" y="{ys(c):.1f}" width="{max(0.5, xs(e1) - xs(e0) - 0.5):.1f}" '
            f'height="{(mt + ph - ys(c)):.1f}" fill="#3182bd"/>'
        )
    if x0 <= 0 <= x1:
        parts.append(
            f'<line x1="{xs(0):.1f}" y1="{mt}" x2="{xs(0):.1f}" y2="{mt + ph}" stroke="#d62728" stroke-dasharray="4 3"/>'
        )
    parts.append(
        f'<line x1="{xs(delta):.1f}" y1="{mt}" x2="{xs(delta):.1f}" y2="{mt + ph}" stroke="black" stroke-width="2"/>'
    )
    parts.append(f'<line x1="{ml}" y1="{mt + ph}" x2="{ml + pw}" y2="{mt + ph}" stroke="black"/>')
    for v in np.linspace(x0, x1, 6):
        parts.append(
            f'<text x="{xs(v):.1f}" y="{mt + ph + 18}" text-anchor="middle" font-family="sans-serif" '
            f'font-size="11">{v * 100:+.1f}%</text>'
        )
    parts.append(
        f'<text x="{width / 2}" y="{height - 8}" text-anchor="middle" font-family="sans-serif" font-size="12">'
        f"delta = {delta * 100:+.2f}%, 95% CI [{ci_lo * 100:+.2f}%, {ci_hi * 100:+.2f}%] (shaded); dashed = 0</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)
