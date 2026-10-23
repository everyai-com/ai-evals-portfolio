```
$ -m projects.p03_calibrated_judge.cli kappa --anchors projects/p03_calibrated_judge/data/anchors.jsonl

# Inter-annotator agreement

| measure | value |
|---|---|
| n | 500 |
| raw agreement | 0.836 |
| Cohen's kappa | 0.749 |
| band (Landis & Koch) | substantial |

| strata | kappa |
|---|---|
| code | 0.758 |
| qa | 0.750 |
| reasoning | 0.755 |
| safety | 0.801 |
| summarize | 0.674 |

```
