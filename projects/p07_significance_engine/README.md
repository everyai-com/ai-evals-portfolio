# p07 — Statistical Significance Engine

## What

A bootstrapping utility that turns two per-example result files into

```
Model B wins by 3.5% ± 2.1% (p<0.05, n=400)
```

instead of two raw accuracy numbers, with everything a reviewer needs underneath: paired
bootstrap CI, exact McNemar for binary metrics, a permutation-test cross-check, effect-size
band, the minimum detectable effect at this n, and a **power warning** when the observed
delta is inside the noise floor. It also handles many candidates vs one baseline
(Holm–Bonferroni), per-strata breakdowns with a Simpson's-paradox check, multi-seed runs
(seed variance vs between-model delta) and an SVG plot of the bootstrap distribution.

The math is `evalkit.stats` (`paired_bootstrap`, `permutation_test`, `mcnemar_exact`,
`required_sample_size`, `seed_variance`); this project is the product around it.

## Why

"B is 72.5% and A is 70.1%" is not a result. On 50 examples a 4-point gap is a coin flip;
on 400 paired examples a 3.5-point gap is real. Reporting the CI, the p-value, the power and
the sample size needed makes the difference between a model decision and a seed lottery.

## How to run

```bash
D=projects/p07_significance_engine/data
.venv/bin/python -m projects.p07_significance_engine.cli compare --a $D/model_a.jsonl --b $D/model_b.jsonl --metric success [--n-boot 10000] [--seed 0] [--json [out.json]] [--plot out.svg]
.venv/bin/python -m projects.p07_significance_engine.cli compare --a $D/small_a.jsonl --b $D/small_b.jsonl --metric success   # exit 1: power warning
.venv/bin/python -m projects.p07_significance_engine.cli multi --baseline $D/model_a.jsonl --candidates $D/model_b.jsonl $D/model_c.jsonl --metric success
.venv/bin/python -m projects.p07_significance_engine.cli power --baseline 0.7 --delta 0.02
.venv/bin/python -m projects.p07_significance_engine.cli seeds --runs $D/runs.jsonl --metric success           # exit 1: seed-variance warning
.venv/bin/python -m projects.p07_significance_engine.data.make_results --seed 0                               # regenerate fixtures
.venv/bin/pytest projects/p07_significance_engine -q
```

Input formats (joined on `id`; unpaired ids are reported and dropped, or refused with
`--strict`): two JSONL files of `{id, score}` / `{id, success}` (plus optional `strata` or
`tags`), one file of `{id, a, b}`, or a runs file of `{id, model, seed, score}` for `seeds`.
`--json` with no path prints JSON to stdout; with a path it writes the file.

## Sample output

`compare --a model_a.jsonl --b model_b.jsonl --metric success` (n=400, B truly +3%):

```
**Model B wins by 3.5% ± 2.1% (p<0.05, n=400)**

| quantity | value |
|---|---|
| metric | success |
| n (paired) | 400 |
| A = model_a mean | 0.6700 |
| B = model_b mean | 0.7050 |
| delta (B - A) | +0.0350 |
| 95% bootstrap CI (10000 resamples) | [+0.0150, +0.0575] |
| bootstrap p (two-sided) | 0.0014 |
| permutation p (cross-check) | 0.0015 |
| McNemar exact p (B fixes / B breaks) | 0.0026 (17 / 3) |
| effect size (Cohen's h) | +0.076 (negligible) |
| sd of per-example difference | 0.2211 |
| minimum detectable effect at this n (power 0.8) | 0.0310 |
| post-hoc power for the observed delta | 0.89 |
| required n (paired) for this delta | 314 |
| required n per arm (unpaired proportions, evalkit.required_sample_size) | 2752 |
| verdict | B (significant) |

### Per-strata breakdown

| strata | n | A | B | delta | 95% CI | p | winner |
|---|---|---|---|---|---|---|---|
| code | 100 | 0.550 | 0.640 | +0.090 | [+0.030, +0.160] | 0.006 | B |
| math | 100 | 0.590 | 0.610 | +0.020 | [-0.020, +0.060] | 0.424 | tie |
| qa | 100 | 0.840 | 0.850 | +0.010 | [+0.000, +0.030] | 0.739 | tie |
| summarize | 100 | 0.700 | 0.720 | +0.020 | [-0.020, +0.060] | 0.441 | tie |

micro delta +0.0350 (B), macro delta +0.0350 (B)
```

Same effect on n=50 (`small_a` / `small_b`), exit code 1:

```
**No significant difference: Model B +4.0% ± 5.0% (p=0.25, n=50) - underpowered, need n≈193**
...
| McNemar exact p (B fixes / B breaks) | 0.5000 (2 / 0) |
| minimum detectable effect at this n (power 0.8) | 0.0784 |
| post-hoc power for the observed delta | 0.30 |
| required n (paired) for this delta | 193 |
| required n per arm (unpaired proportions, evalkit.required_sample_size) | 1789 |
| verdict | tie (not significant) |

> POWER WARNING: a 4.0-point delta on 50 samples is noise: the minimum detectable effect at n=50 is 7.8 points (power 0.8). You need n≈193 paired samples (or ≈1789 per arm unpaired) to call this.
```

`multi` (B is a real +3%, C is a null candidate):

```
# 2 candidates vs baseline `model_a` (Holm-Bonferroni, alpha=0.05)

| candidate | n | delta | 95% CI | p | p (Holm) | sig raw | sig Holm | winner |
|---|---|---|---|---|---|---|---|---|
| model_b | 400 | +0.0350 | [+0.0150, +0.0575] | 0.0014 | 0.0028 | yes | yes | B |
| model_c | 400 | +0.0175 | [-0.0125, +0.0475] | 0.2984 | 0.2984 | no | no | tie |

1 comparisons significant before correction, 1 after.

> model_c: a 1.8-point delta on 400 samples is noise: the minimum detectable effect at n=400 is 4.4 points (power 0.8). You need n≈2498 paired samples (or ≈11176 per arm unpaired) to call this.
```

`seeds --runs runs.jsonl --metric success` (3 seeds × 2 models), exit code 1:

```
| model | seeds | mean | std | min | max |
|---|---|---|---|---|---|
| model_a | 3 | 0.6633 | 0.0491 | 0.6075 | 0.7000 |
| model_b | 3 | 0.7108 | 0.0227 | 0.6850 | 0.7275 |

- WARNING model_b vs model_a: delta +0.0475 between model_b and model_a is smaller than the within-model seed std 0.0491: a single-seed comparison can flip sign on re-run. Average over seeds (or more seeds) before claiming a winner.
```

`power --baseline 0.7 --delta 0.02`:

```
To detect a 2.0-point change from 70% you need **n=8080 per arm** (unpaired two-proportion test; a paired design needs fewer).

| delta | required n per arm |
|---|---|
| +0.010 | 32647 |
| +0.020 | 8080 |
| +0.030 | 3554 |
| +0.050 | 1251 |
| +0.100 | 294 |
```

`--plot out.svg` writes a histogram of the 10 000 bootstrap deltas with the 95% CI shaded,
the observed delta as a solid line and zero as a dashed line (pure string SVG, no matplotlib).

## Design notes

- **Headline.** `executive_headline()` emits exactly `Model B wins by X% ± Y% (p<0.05, n=N)`
  where Y is the half-width of the percentile bootstrap CI; non-significant results say so and
  append the paired n needed. (`DeltaReport.headline()` in evalkit includes the metric name;
  the engine's version is the post's format verbatim.)
- **Power.** Minimum detectable paired effect `MDE = (z_{1-α/2} + z_{0.8}) · sd(b−a) / √n` and
  `required n = ((z_{1-α/2} + z_{0.8}) · sd / δ)²` use the per-example difference, which is why
  pairing needs 314 samples where the unpaired two-proportion formula
  (`evalkit.stats.required_sample_size`) needs 2752 per arm. The warning fires when
  |δ| < MDE: "noise" when not significant, "significant but underpowered (winner's curse)"
  when it is.
- **Cross-checks.** A paired sign-flip permutation p and, for 0/1 metrics, exact McNemar on
  the discordant pairs; if these disagree with the bootstrap you have a modelling problem,
  not a model result.
- **Holm–Bonferroni** step-down across the candidate family: sort p ascending,
  `p_adj(i) = max_{j≤i} min(1, (m−j+1)·p(j))`. Ten null candidates yield zero rejections after
  correction (tested).
- **Simpson's check.** With paired data the overall delta is the size-weighted mean of the
  per-strata deltas, so a *full* reversal is impossible; the check flags strata whose sign
  disagrees with the overall and a disagreement between the micro (size-weighted) and macro
  (equal-weight) winner, which is the paradox in its paired form.
- **Seed variance.** `evalkit.stats.seed_variance` per model; warn when |Δ between model
  means| < max within-model std. The demo runs file is built with seed offsets of sd 0.04 so a
  single-seed +3% is inside seed noise, which is the lesson.
- **evalkit gap worked around:** `paired_bootstrap` does not return its bootstrap samples, so
  `bootstrap_distribution()` re-runs the identical RNG stream for the plot (verified equal in
  tests). Returning the samples from evalkit would remove the duplication.
