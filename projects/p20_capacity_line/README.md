# p20 — The scary line

**The mistake:** every click, page view and deploy went into the same database the whole platform
ran on, nobody decided that, and the first anyone knew was an engineer's chart showing how close
the 10 GB cap was ("the scary line"). Half of it was analytics. Events moved to an analytics store;
months later people-counts turned out ~7× too low because that store *samples*, so exact counts
moved back. **Decide the questions before deciding where the data lives, never use sampled
telemetry as a count of record, and watch the line before it is scary.**

The tests that prove the lesson is encoded: `test_migration_at_day_60_drops_the_line` (analytics was
half the closet, moving it out drops the line and multiplies days-to-cap), `test_placement_flags_planted_misplacements`
(distinct users served from a sampled table is `critical`; versions in the hot DB is a `warning`),
`test_sampling_undercount_grows_as_rate_falls_and_events_stay_unbiased`.

## What

| piece | does |
|---|---|
| `StorageSnapshot` / `load_snapshots` | per-table rows + bytes with a `category` (platform, analytics, versions, logs, other) and the store's `cap_bytes`, one row per day in `data/snapshots.jsonl` |
| `CapacityLine.analyse` | utilisation %, composition by category, Theil-Sen growth per category and in total, **days-to-cap** at current and p90 growth, biggest table and the "what if we moved its category out" projection, 50/75/90 % alerts (first day crossed or projected day), ASCII sparkline |
| `html_report` / `svg_chart` | self-contained HTML with an inline SVG line chart: no scripts, no stylesheets, no URLs |
| `recommend_placement` / `audit_layout` | maps a question's needs (exactness, retention, query shape, write rate) to a store class and flags **misplacements** in `data/layout.yaml` with a severity (critical = wrong answers, warning = eats capacity, info = odd) |
| `sampling_undercount` / `sampling_sweep` | simulates a sampled store over a skewed population of user events and shows distinct-user undercount vs exact, with `evalkit.metrics.bootstrap_ci` over repeated seeds, and the (unbiased) scaled event count next to it |

## Why

A line that is only drawn when it is scary leaves one option: panic migration. Drawn from day one
it says *which* category is the problem, *when* it bites, and what moving it buys. And the second
half of the story is the quieter failure: a sampled store is fine for "how many events" and wrong
for "how many people", because the users with one event are exactly the ones sampling drops.

## How to run

```bash
D=projects/p20_capacity_line/data
.venv/bin/python -m projects.p20_capacity_line.cli analyse --snapshots $D/snapshots.jsonl --cap-gb 10 [--until-day 59] [--window 14] [--html report.html] [--json [out.json]] [--max-util 0.9]
.venv/bin/python -m projects.p20_capacity_line.cli placement --questions $D/questions.yaml --layout $D/layout.yaml [--json]
.venv/bin/python -m projects.p20_capacity_line.cli sampling --rate 0.1 --users 10000 --seed 0 [--rates 0.5,0.2,0.1,0.05,0.01] [--skew 1.5]
.venv/bin/python -m projects.p20_capacity_line.data.make_snapshots --seed 0 --days 90   # regenerate the fixture
.venv/bin/pytest projects/p20_capacity_line -q
```

`analyse` exits 1 when utilisation is at or above `--max-util` (default 90 %), `placement` exits 1
when any critical misplacement exists, so both work as gates. `evalkit capacity ...` is the alias.

Library use:

```python
from projects.p20_capacity_line import core
snaps = core.load_snapshots("data/snapshots.jsonl")
rep = core.CapacityLine(window=14).analyse(snaps)
rep.days_to_cap, rep.composition, rep.what_if_moved, rep.alerts
core.recommend_placement({"id": "q", "exactness": "exact", "retention": "hot", "write_rate": "high"}).store_class  # exact_counter
core.sampling_undercount(core.generate_population(10_000, seed=0), sample_rate=0.1, seed=0)
```

## Sample output

`analyse --snapshots data/snapshots.jsonl --cap-gb 10 --until-day 59 --html line.html` (the day before
the migration; exit 0 because 84.9 % is under the default 90 % gate):

```
# The line: day 59 (2026-03-01)

**Utilisation 84.9%** (8.49 GB of 10.00 GB). Growth 100.6 MB/day (Theil-Sen over last 14 days), p90 150.6 MB/day. **Days to cap: 15 days** (p90 growth: 10 days).

`▃▃▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▅▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▆▇▇▇▇▇▇▇` utilisation, day 0 -> 59

## Composition

| category | GB | share | growth MB/day | p90 MB/day |
|---|---|---|---|---|
| analytics | 4.347 | 51.2% | 75.157 | 110.308 |
| platform | 1.724 | 20.3% | 10.889 | 14.614 |
| versions | 1.382 | 16.3% | 13.846 | 22.878 |
| logs | 0.946 | 11.1% | 0.684 | 40.976 |
| other | 0.094 | 1.1% | 0.085 | 0.833 |

Fastest-growing category: **analytics**. Biggest table: **events_pageview** (analytics, 2.27 GB, 17,377,763 rows).

What if **analytics** moved out: utilisation 41.5%, growth 25.4 MB/day, days to cap 236 days.

## Alerts

| threshold | status | detail |
|---|---|---|
| 50% | CROSSED | first crossed day 21 |
| 75% | CROSSED | first crossed day 50 |
| 90% | ok | projected in 5 days |

html written to line.html
```

The same command over all 90 days (after the day-60 migration) reports **49.7 %**, growth
28.1 MB/day, days to cap 183, fastest category `versions`, and the sparkline ends
`▇▇▇▄▄▄▄…` — the line dropped.

`placement --questions data/questions.yaml --layout data/layout.yaml` (exit 1: one critical):

```
| question | exactness | retention | write rate | recommended store |
|---|---|---|---|---|
| distinct_users_month | exact | hot | high | exact_counter |
| pageviews_per_day | approximate | warm | high | analytics_sampled |
| clicks_per_feature | approximate | warm | high | analytics_sampled |
| every_version_shipped | exact | cold | medium | cold_archive |
| user_profile | exact | hot | low | transactional |
| app_record | exact | hot | low | transactional |
| deploys_per_app_today | exact | hot | high | exact_counter |
| request_logs_debug | approximate | warm | high | analytics_sampled |
| compliance_access_log | exact | cold | medium | cold_archive |

## Misplacements in current layout (3)

| severity | question | table | actual | recommended | why |
|---|---|---|---|---|---|
| critical | distinct_users_month | events_pageview_sampled | analytics_sampled | exact_counter | exact question answered from a sampled store: counts will be wrong |
| warning | every_version_shipped | app_versions | transactional | cold_archive | cold data in the hot transactional store: eats capacity |
| warning | request_logs_debug | request_logs | transactional | analytics_sampled | approximate telemetry stored in the transactional store: eats capacity |
```

`sampling --rate 0.1 --users 10000 --seed 0`:

```
# Sampling audit: 10,000 users, event skew 1.5

| rate | exact users | sampled users | undercount (bootstrap 95%) | expected missed | event estimate error (95%) |
|---|---|---|---|---|---|
| 0.500 | 10000 | 7045 | 1.42x [1.41, 1.43] | 29.6% | -0.06% [-0.31, +0.19] |
| 0.200 | 10000 | 4025 | 2.48x [2.47, 2.50] | 59.7% | -0.00% [-0.35, +0.32] |
| 0.100 | 10000 | 2507 | 3.99x [3.96, 4.02] | 75.2% | +0.91% [+0.09, +1.70] |
| 0.050 | 10000 | 1451 | 6.89x [6.79, 6.98] | 85.4% | +0.23% [-1.16, +1.98] |
| 0.010 | 10000 | 367 | 27.37x [26.32, 28.36] | 96.3% | +0.03% [-3.28, +3.32] |

Counting *events* from the sampled store and scaling by 1/rate is unbiased. Counting *people* from it is not: a user with k events survives sampling with probability 1 - (1 - r)^k, so the long tail of one-event users vanishes and nothing scales it back.
```

At a 5 % sample the people-count is 6.9× too low — the talk's "~7×" — while the event count is
within 0.3 % of exact. The `expected missed` column is the closed form `mean_user (1 - r)^k` and
matches the simulation.

## Design notes

- **Growth fit.** Theil-Sen (median of all pairwise slopes over the window) rather than least
  squares, so one bad day or the migration cliff itself does not drag the projection; the p90 of
  day-over-day deltas is the pessimistic rate. Days-to-cap is `(cap - current) / slope`, `inf` when
  not growing (serialised as `null` in `--json`).
- **What-if** subtracts the biggest table's whole category from both the level and the slope, which
  is what the migration actually did.
- **Placement** is a four-row decision table (`cold` retention -> `cold_archive`; `exact` + high
  write rate -> `exact_counter`; `approximate` -> `analytics_sampled`; else `transactional`) and a
  severity matrix over (recommended, actual) pairs. Serving an exact question from a sampled store
  is the only pairing that produces wrong answers; the others cost capacity or latency.
- **Sampling model.** Users' event counts are `1 + floor(pareto(skew) * (mean - 1))`; lower `skew`
  = heavier tail = more users with one event = worse undercount. `bootstrap_ci` runs over `--reps`
  seeded simulations per rate.
- **Fixture.** `make_snapshots.py` grows analytics quadratically (every new app adds a stream), logs
  plateau at 30-day rotation, versions never shrink; on day 60 analytics drops to a 2 % residual.
- `evalkit` reuse: `read_jsonl`, `write_jsonl`, `bootstrap_ci`, `md_table`, `pct`.
