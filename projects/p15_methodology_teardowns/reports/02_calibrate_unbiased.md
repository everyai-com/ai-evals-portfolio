```
$ -m projects.p03_calibrated_judge.cli calibrate --anchors projects/p03_calibrated_judge/data/anchors.jsonl --judge unbiased

# Judge calibration report: `mock:unbiased`

Anchors: n=500. Thresholds: position 0.1, verbosity 0.1, self-preference 0.1, min kappa 0.4.

## Biases

| bias | statistic | 95% CI | n | threshold | verdict |
|---|---|---|---|---|---|
| position (net first-position preference) | +0.003 | [-0.050, +0.057] | 500 | 0.100 | acceptable |
| position (order consistency) | 0.974 | [0.956, 0.985] | 500 | - | - |
| verbosity (longer win-rate lift over gold) | -0.030 | [-0.055, -0.007] | 493 | 0.100 | acceptable |
| verbosity (longer pick-rate when gold not longer) | 0.038 | [0.020, 0.060] | 302 | - | - |
| self-preference (self pick-rate - gold) | -0.032 | [-0.068, +0.005] | 190 | 0.100 | acceptable |

Longer-response win-rate by length ratio (judge vs gold):

| length ratio | n | judge picks longer | gold picks longer |
|---|---|---|---|
| [1.0, 1.25) | 111 | 39.6% | 42.3% |
| [1.25, 1.5) | 75 | 34.0% | 38.7% |
| [1.5, 2.0) | 96 | 33.3% | 35.4% |
| [2.0, 3.0) | 104 | 36.5% | 36.5% |
| [3.0, inf) | 107 | 34.1% | 40.2% |

## Agreement with humans

| measure | value | band |
|---|---|---|
| accuracy vs gold | 0.899 [0.873, 0.925] | - |
| kappa judge vs gold | 0.846 | almost perfect |
| kappa judge vs annotator_1 | 0.702 | substantial |
| kappa annotator_1 vs annotator_2 (human ceiling) | 0.749 | substantial |

| strata | n | accuracy | kappa vs gold | human kappa |
|---|---|---|---|---|
| code | 100 | 0.870 | 0.805 | 0.758 |
| qa | 100 | 0.885 | 0.822 | 0.750 |
| reasoning | 100 | 0.920 | 0.878 | 0.755 |
| safety | 100 | 0.935 | 0.900 | 0.801 |
| summarize | 100 | 0.885 | 0.819 | 0.674 |

## Conclusion

USABLE for pairwise ranking and regression gating across all strata: kappa 0.85 (almost perfect), at the human ceiling; no bias exceeds its threshold.

- position bias: acceptable (|+0.003| vs threshold 0.1); the same pair gets the same verdict in both orders 97% of the time
- verbosity bias: acceptable (|-0.030| vs threshold 0.1); judge picks the longer response 4% of the time when humans did not
- self-preference bias: acceptable (|-0.032| vs threshold 0.1); judge favoured its own outputs 31% vs humans 34%
- judge-vs-gold kappa 0.846 (almost perfect) against a human ceiling of 0.749 (substantial)

```
