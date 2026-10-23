```
$ -m projects.p07_significance_engine.cli compare --a projects/p07_significance_engine/data/model_a.jsonl --b projects/p07_significance_engine/data/model_b.jsonl --metric success --n-boot 10000 --seed 0

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
