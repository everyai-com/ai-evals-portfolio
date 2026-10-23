# p03 — Calibrated LLM-as-a-Judge (+ golden dataset engineering)

## What

A calibration pipeline that measures an LLM judge's **position**, **verbosity** and
**self-preference** biases, and its **agreement with humans**, against 500 human-labelled
pairwise anchors, then says whether the judge is usable and for what.

It ships with the golden dataset itself (`data/anchors.jsonl`, versioned and fingerprinted),
the annotation protocol (`data/ANNOTATION_GUIDELINES.md`), a stratified sampler, a pairwise
and a pointwise judge, a swap-debiasing wrapper, and a mock judge with *injectable* biases so
the pipeline can be proven to detect what it claims to detect.

## Why

A judge that prefers the first response it sees, the longer one, or its own model family
will silently invert leaderboards and let regressions through gates. You cannot trust a
judge's verdicts until you have measured its biases against human labels with confidence
intervals, and you cannot measure anything without a golden set whose own reliability
(inter-annotator kappa) you know.

## How to run

```bash
.venv/bin/python -m projects.p03_calibrated_judge.cli make-anchors --n 500 --seed 0   # regenerate data + CHANGELOG
.venv/bin/python -m projects.p03_calibrated_judge.cli kappa --anchors projects/p03_calibrated_judge/data/anchors.jsonl
.venv/bin/python -m projects.p03_calibrated_judge.cli calibrate --judge biased   [--json report.json]
.venv/bin/python -m projects.p03_calibrated_judge.cli calibrate --judge biased --debias
.venv/bin/python -m projects.p03_calibrated_judge.cli calibrate --judge unbiased --sample 100
.venv/bin/pytest projects/p03_calibrated_judge -q
```

`--judge biased` injects 0.3 of each bias into the mock (`--position-bias/--verbosity-bias/
--self-preference` override individually); `--judge unbiased` follows gold ~90%.
`EVALKIT_LLM=anthropic` calibrates a real model with the same code. Exit code 1 means at least
one bias exceeds its threshold (or kappa is below the floor), so the command works as a gate.

Library use:

```python
from projects.p03_calibrated_judge import core

judge = core.PairwiseJudge(client, rubric=core.DEFAULT_RUBRIC)
result = core.calibrate(judge, anchors)  # judges every anchor in both orders
result.position["value"], result.verbosity["ci"], result.usable_for()
safe = core.SwapDebiasedJudge(judge)  # judges (A,B) and (B,A); tie on disagreement
```

## Sample output

`calibrate --judge biased` (exit code 1):

```
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

Same judge wrapped with `--debias` (swap-debiasing): position bias goes to exactly zero and
order consistency to 100%, while the other two biases are untouched (they are not positional):

```
| position (net first-position preference) | +0.000 | [-0.046, +0.047] | 500 | 0.100 | acceptable |
| position (order consistency) | 1.000 | [0.992, 1.000] | 500 | - | - |
| verbosity (longer win-rate lift over gold) | -0.012 | [-0.063, +0.037] | 493 | 0.100 | acceptable |
...
- self-preference bias: acceptable (|+0.042| vs threshold 0.1); judge favoured its own outputs 38% vs humans 34%
- judge-vs-gold kappa 0.320 (fair) against a human ceiling of 0.749 (substantial)
```

`calibrate --judge unbiased` (exit code 0):

```
| position (net first-position preference) | +0.003 | [-0.050, +0.057] | 500 | 0.100 | acceptable |
| position (order consistency) | 0.974 | [0.956, 0.985] | 500 | - | - |
| verbosity (longer win-rate lift over gold) | -0.030 | [-0.055, -0.007] | 493 | 0.100 | acceptable |
| verbosity (longer pick-rate when gold not longer) | 0.038 | [0.020, 0.060] | 302 | - | - |
## Conclusion

USABLE for pairwise ranking and regression gating across all strata: kappa 0.85 (almost perfect), at the human ceiling; no bias exceeds its threshold.
```

`kappa` (inter-annotator agreement of the golden set):

```
| measure | value |
|---|---|
| n | 500 |
| raw agreement | 0.836 |
| Cohen's kappa | 0.749 |
| band (Landis & Koch) | substantial |
```

## Design notes

**Golden dataset.** `data/make_anchors.py` models the labelling process rather than drawing
labels at random: each response has a hidden quality, the true preference is a tie inside a
margin, difficulty is the quality gap, each annotator reproduces the truth with a
difficulty-dependent probability (easy 0.98 / medium 0.94 / hard 0.86, giving 83.6% raw
agreement and kappa 0.75), and `gold` is *adjudicated* (agreements kept, disagreements
resolved by a third reviewer). Response length is drawn independently of quality and ~40%
of pairs put `judge-self` on one side, so verbosity and self-preference are identifiable.
`CHANGELOG.md` records the row count and `evalkit.io.dataset_fingerprint` sha256 for
v1.0.0; a test fails if the committed file drifts from the changelog.

**Position bias** = P(pick first) − P(pick second), pooled over both presentation orders
(null 0), with a bootstrap CI over the ±1/0 observations; order consistency is reported
alongside with a Wilson CI. **Verbosity** is reported two ways: the classic "how often does
the judge pick the longer response when humans did not" and, driving the verdict, the lift
P(judge picks longer) − P(gold picks longer), whose null value is 0 for a judge of *any*
accuracy. **Self-preference** = P(judge picks `judge-self`) − P(gold picks `judge-self`) on
pairs with exactly one self side. All three non-position measures average the two
presentation orders per anchor, which makes them robust to position bias (a position-only
judge flags *only* position; see `test_each_bias_is_independently_detectable`).

**Mock judge.** `BiasedMockJudge` starts from gold (followed w.p. 0.9), then applies
verbosity, self-preference and position overrides with per-item hashed draws, so the same
anchor is biased the same way in both orders (which is what makes position bias show up as
inconsistency). The unbiased mock's kappa (0.85) exceeds the human ceiling (0.75) because it
follows the adjudicated gold directly while each annotator is noisy relative to it; a real
model lands below the ceiling.

**Thresholds** (0.10 for each bias, kappa floor 0.4) are deliberately strict defaults; pass
`thresholds=` to `calibrate()` to change them. `evalkit` reuse: `cohen_kappa`,
`interpret_kappa`, `bootstrap_ci`, `wilson_ci`, `accuracy`, `MockLLM`, `read/write_jsonl`,
`dataset_fingerprint`, `md_table`, `pct`.
