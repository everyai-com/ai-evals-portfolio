```
$ -m projects.p03_calibrated_judge.cli calibrate --anchors projects/p03_calibrated_judge/data/anchors.jsonl --judge biased

# Judge calibration report: `mock:biased`

Anchors: n=500. Thresholds: position 0.1, verbosity 0.1, self-preference 0.1, min kappa 0.4.

## Biases

| bias | statistic | 95% CI | n | threshold | verdict |
|---|---|---|---|---|---|
| position (net first-position preference) | +0.317 | [+0.263, +0.372] | 500 | 0.100 | needs debiasing |
| position (order consistency) | 0.674 | [0.632, 0.714] | 500 | - | - |
| verbosity (longer win-rate lift over gold) | +0.151 | [+0.106, +0.195] | 493 | 0.100 | needs debiasing |
| verbosity (longer pick-rate when gold not longer) | 0.422 | [0.379, 0.469] | 302 | - | - |
| self-preference (self pick-rate - gold) | +0.205 | [+0.132, +0.284] | 190 | 0.100 | needs debiasing |

Longer-response win-rate by length ratio (judge vs gold):

| length ratio | n | judge picks longer | gold picks longer |
|---|---|---|---|
| [1.0, 1.25) | 111 | 53.2% | 42.3% |
| [1.25, 1.5) | 75 | 55.3% | 38.7% |
| [1.5, 2.0) | 96 | 55.2% | 35.4% |
| [2.0, 3.0) | 104 | 54.8% | 36.5% |
| [3.0, inf) | 107 | 51.4% | 40.2% |

## Agreement with humans

| measure | value | band |
|---|---|---|
| accuracy vs gold | 0.571 [0.534, 0.607] | - |
| kappa judge vs gold | 0.323 | fair |
| kappa judge vs annotator_1 | 0.257 | fair |
| kappa annotator_1 vs annotator_2 (human ceiling) | 0.749 | substantial |

| strata | n | accuracy | kappa vs gold | human kappa |
|---|---|---|---|---|
| code | 100 | 0.600 | 0.392 | 0.758 |
| qa | 100 | 0.600 | 0.358 | 0.750 |
| reasoning | 100 | 0.580 | 0.340 | 0.755 |
| safety | 100 | 0.595 | 0.373 | 0.801 |
| summarize | 100 | 0.480 | 0.139 | 0.674 |

## Conclusion

NOT usable: kappa vs gold is 0.32 (fair), below the 0.4 floor. Fix the rubric or pick another judge.

- position bias: needs debiasing (|+0.317| vs threshold 0.1); the same pair gets the same verdict in both orders 67% of the time
- verbosity bias: needs debiasing (|+0.151| vs threshold 0.1); judge picks the longer response 42% of the time when humans did not
- self-preference bias: needs debiasing (|+0.205| vs threshold 0.1); judge favoured its own outputs 55% vs humans 34%
- judge-vs-gold kappa 0.323 (fair) against a human ceiling of 0.749 (substantial)

```
