```
$ -m projects.p07_significance_engine.cli compare --a projects/p07_significance_engine/data/small_a.jsonl --b projects/p07_significance_engine/data/small_b.jsonl --metric success --seed 0

**No significant difference: Model B +4.0% ± 5.0% (p=0.25, n=50) - underpowered, need n≈193**

| quantity | value |
|---|---|
| metric | success |
| n (paired) | 50 |
| A = small_a mean | 0.7400 |
| B = small_b mean | 0.7800 |
| delta (B - A) | +0.0400 |
| 95% bootstrap CI (10000 resamples) | [+0.0000, +0.1000] |
| bootstrap p (two-sided) | 0.2504 |
| permutation p (cross-check) | 0.4937 |
| McNemar exact p (B fixes / B breaks) | 0.5000 (2 / 0) |
| effect size (Cohen's h) | +0.094 (negligible) |
| sd of per-example difference | 0.1979 |
| minimum detectable effect at this n (power 0.8) | 0.0784 |
| post-hoc power for the observed delta | 0.30 |
| required n (paired) for this delta | 193 |
| required n per arm (unpaired proportions, evalkit.required_sample_size) | 1789 |
| verdict | tie (not significant) |

> POWER WARNING: a 4.0-point delta on 50 samples is noise: the minimum detectable effect at n=50 is 7.8 points (power 0.8). You need n≈193 paired samples (or ≈1789 per arm unpaired) to call this.

### Per-strata breakdown

| strata | n | A | B | delta | 95% CI | p | winner |
|---|---|---|---|---|---|---|---|
| code | 13 | 0.615 | 0.692 | +0.077 | [+0.000, +0.231] | 0.691 | tie |
| math | 12 | 0.750 | 0.833 | +0.083 | [+0.000, +0.250] | 0.719 | tie |
| qa | 13 | 0.769 | 0.769 | +0.000 | [+0.000, +0.000] | 1.000 | tie |
| summarize | 12 | 0.833 | 0.833 | +0.000 | [+0.000, +0.000] | 1.000 | tie |

micro delta +0.0400 (B), macro delta +0.0401 (B)

```
