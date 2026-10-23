```
$ -m projects.p11_counterfactual_replay.cli bisect --run projects/p11_counterfactual_replay/data/corrupted_run.jsonl

## Bisect: culprit = `search` (hop 1)

| # | override node -> golden | final answer | result |
|---|---|---|---|
| 0 | plan | The three largest Acme Corp offices have a combined headcount of 550. | still wrong |
| 1 | search | The three largest Acme Corp offices have a combined headcount of 265. | **fixed** |

```
