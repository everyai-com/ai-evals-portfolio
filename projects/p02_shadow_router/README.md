# p02 Shadow Routing Comparator

## What
A proxy that answers every request from the production ("primary") model and silently mirrors a
deterministic sample of them (default 5%) to a candidate ("shadow") model on a background thread.
Both responses are written to a JSONL store; a report then diffs cost, latency and answer quality
and prints a promote / hold / reject recommendation driven by YAML thresholds.

Users never see the shadow: `ShadowProxy.complete()` returns the primary response as soon as it
arrives, and shadow exceptions, timeouts and store failures are logged and recorded, never raised.

## Why
Offline evals tell you how a candidate does on *your* benchmark. Shadow routing tells you how it
does on *today's* traffic, at zero user risk, with paired statistics that need far fewer samples
than an A/B test because every pair sees the same prompt.

## How to run
```bash
.venv/bin/python -m projects.p02_shadow_router.cli run --traffic projects/p02_shadow_router/data/traffic.jsonl --rate 0.05 --out runs/shadow.jsonl
.venv/bin/python -m projects.p02_shadow_router.cli report --store runs/shadow.jsonl --judge contains [--json metrics.json]
.venv/bin/pytest projects/p02_shadow_router -q
```
`evalkit shadow ...` is the same CLI. `report` exits 1 on `reject` (or on `hold` with `--fail-on hold`).

Library use:
```python
from projects.p02_shadow_router import ShadowProxy, ShadowAgentRunner, TrajectoryDiff, ReportBuilder, JsonlStore
proxy = ShadowProxy(primary_client, candidate_client, sample_rate=0.05, seed=0, store=JsonlStore("runs/shadow.jsonl"))
resp = proxy.complete(prompt, request_id=req.id)        # always the primary answer
proxy.close()                                            # drain background shadow calls
print(ReportBuilder.from_store(JsonlStore("runs/shadow.jsonl")).to_markdown())
```
`ShadowAgentRunner(primary_agent, shadow_agent, ...)` does the same for whole agent runs
(`Callable[[task_id, input], evalkit.trajectory.Trajectory]`) and stores both trajectories.

## Sample output
Real run on the committed fixture (`data/traffic.jsonl`, 200 prod-like requests spread over one day,
primary `mock-system-one`, shadow `mock-system-two` with 2 ms latency and a 2% simulated failure rate):

```
$ python -m projects.p02_shadow_router.cli run --traffic projects/p02_shadow_router/data/traffic.jsonl --rate 0.05 --out runs/shadow.jsonl
replayed 200 requests; shadowed 9 (4.5%); shadow errors 0, timeouts 0; store -> runs/shadow.jsonl
$ python -m projects.p02_shadow_router.cli report --store runs/shadow.jsonl --judge contains
# Shadow routing report: `mock-system-one` (primary) vs `mock-system-two` (shadow)

**Recommendation: REJECT** - cost ratio 24.5x >= 10.0x

## Sample
| metric | value |
|---|---|
| requests observed | 200 |
| shadowed | 9 (4.5%) |
| comparable pairs | 9 |
| shadow errors | 0 (0.0%, 95% CI 0.0%-29.9%) |
| traffic span (days) | 1.00 |
| requests / day | 200.0 |

## Quality
| metric | value |
|---|---|
| agreement (judge) | 88.9% (95% Wilson CI 56.5%-98.0%) |
| exact match | 11.1% |
| normalized match | 55.6% |
| same / better / worse / unknown / error | 8 / 0 / 0 / 1 / 0 |
| mean tool-sequence edit distance | 0.00 |

## Cost and latency (paired, shadow - primary)
| metric | primary mean | shadow mean | delta | 95% bootstrap CI | p |
|---|---|---|---|---|---|
| cost / request (USD) | 0.000007 | 0.000180 | +0.000173 | [+0.000138, +0.000213] | 0.001 |
| latency (ms) | 0.0 | 4.9 | +4.9 | [+4.7, +5.0] | 0.001 |
| tokens / request | 18.7 | 20.0 | +1.3 | [+0.0, +2.7] | 0.061 |

| latency | p50 (ms) | p95 (ms) |
|---|---|---|
| primary | 0.0 | 0.0 |
| shadow | 5.0 | 5.1 |

Cost ratio shadow/primary: **24.55x**; p95 latency ratio: **788.10x**.

## Projected monthly cost at 100% rollout
| arm | USD / month |
|---|---|
| primary (today) | 0.04 |
| shadow (if promoted) | 1.08 |
| delta | +1.04 |

```
(exit code 1: the candidate is 24x more expensive per request, which trips the `reject.min_cost_ratio`
rule in `data/thresholds.yaml`. With `--rate 1.0` the same traffic yields 198 pairs: 120 same,
28 shadow_better, 20 shadow_worse, 30 different_unknown, 2 shadow_error.)

## Design notes
* **Sampling** is `sha256(f"{seed}:{request_id}")` mapped to [0, 1) and compared to the rate, so a
  request id is always or never shadowed for a given seed, the sampled share is unbiased, and
  raising the rate is monotone (everything sampled at 5% is also sampled at 10%).
* **Never raise.** The shadow call runs in a `ThreadPoolExecutor`; the worker catches every
  `Exception`, records `shadow_error`, bumps `stats`, and persists the record in a `finally`.
  `shadow_timeout_ms` is a soft limit: the result is kept for cost accounting but the pair is
  classified `shadow_error`. Primary errors propagate, because that is production behaviour.
* **Diff.** `TrajectoryDiff` treats a plain completion as a one-step `Trajectory`, so completions
  and agent runs share one code path: tool-sequence Levenshtein distance, token/cost/latency
  deltas, exact and normalised agreement, and a pluggable judge `(prompt, a, b) -> bool | label`.
  Classification: `shadow_error` > judge label > success labels differ (`shadow_better`/`_worse`)
  > agree (`same`) > `different_unknown`. For completions, an `expected` value in the request
  metadata supplies success labels by normalised containment.
* **Statistics.** Agreement and error rates carry Wilson 95% CIs (`evalkit.metrics.wilson_ci`);
  cost, latency and token deltas are paired bootstraps (`evalkit.stats.paired_bootstrap`);
  p50/p95 are numpy percentiles. Projected monthly cost = observed requests/day (span floored at
  one day) x 30 x mean cost per request in each arm. Formulas are in `report.py`'s docstring.
* **Recommendation.** `reject` rules fire first (agreement upper CI too low, error rate, worse
  rate, or cost ratio over the ceiling); `promote` needs `min_pairs` and every promote rule;
  anything else is `hold`.
