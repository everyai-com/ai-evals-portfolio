```
$ -m projects.p07_significance_engine.cli seeds --runs projects/p07_significance_engine/data/runs.jsonl

# Seed variance on `score`

| model | seeds | mean | std | min | max |
|---|---|---|---|---|---|
| model_a | 3 | 0.6489 | 0.0305 | 0.6184 | 0.6794 |
| model_b | 3 | 0.6528 | 0.0284 | 0.6203 | 0.6727 |

- WARNING model_b vs model_a: delta +0.0040 between model_b and model_a is smaller than the within-model seed std 0.0305: a single-seed comparison can flip sign on re-run. Average over seeds (or more seeds) before claiming a winner.

```
