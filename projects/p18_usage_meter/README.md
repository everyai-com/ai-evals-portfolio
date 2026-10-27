# p18 Usage meter: metering, limits and chargeback

**Mistake it answers:** "let everyone use the sledgehammer on the fly" — with no metering, one
person ran thousands of OCR instances every morning and the bill arrived as a surprise; compute
was pocket change, AI usage was the thing to meter. **Lesson encoded:** metering, limits and
billing from day one, with chargeback to the cost centre that spent it. **Tests that prove it:**
`test_daily_cap_blocks_next_request_and_resets_next_day` (the cap blocks the next request and
resets at midnight on an injected clock), `test_scheduled_burst_and_one_off_spike_labels` (the
06:00 OCR pattern is labelled `scheduled_burst`, a one-off hour is `spike`) and
`test_chargeback_totals_equal_event_sum_and_csv_rows`.

## What
* `UsageEvent` (ts, tenant, team, cost_centre, user, agent_id, kind ∈ {`model_call`, `tool_call`,
  `container_seconds`, `ocr_page`}, model, tokens, units, `success`). Model calls are priced with
  `evalkit.llm.estimate_cost`; other kinds with `data/prices.yaml` × units. `Meter.ingest` dedups
  by `event_id` and aggregates by any of tenant / team / cost_centre / agent / user / day / hour.
* `Limits` from `data/limits.yaml` and `Enforcer.decide(Request) -> Decision(allow, reason,
  remaining, retry_after_s, warnings)` with an injected clock:

| check | kind | rule |
|---|---|---|
| per-request max tokens / units | hard | block if `input + output > max_tokens` (model calls) or `units > max_units` |
| per-agent / per-user daily cap | hard | block if `spent_today + estimated_cost > cap`; `retry_after_s` = seconds to midnight UTC |
| rate limit per agent | hard | token bucket: capacity `burst`, refill `requests_per_minute / 60` per second; `retry_after_s = (1 - tokens) / refill` |
| per-tenant monthly budget | soft | allowed, but 80 % and 100 % alerts fire once per (tenant, month, threshold); persisted in `enforcer.json` |

* **Anomaly detection** (`detect_anomalies`): per user and per agent, today's hourly spend `x_h`
  against the population baseline of non-zero hourly spends over the previous 14 days: with
  `m = median`, `MAD = median(|x − m|)`, flag iff `x_h > m + k · 1.4826 · MAD` and `x_h − m ≥ min_abs`.
  The entity's own history at the same hour-of-day gives recurrence `k/n`; if the Wilson 95 %
  lower bound (`evalkit.metrics.wilson_ci`) of `k/n ≥ 0.5` the label is **`scheduled_burst`**
  (the "every morning" pattern), otherwise **`spike`**.
* **Chargeback** (`chargeback`, `render_chargeback`, `chargeback_csv_rows`): per cost centre and
  team — spend by kind and model, top agents, top users, share of total, tenant budget
  consumption, anomalies, and *cost per successful outcome* (cost of outcome-bearing events /
  successes) where events carry `success`. `invoice` renders an invoice-style statement per tenant.
* Demo data: `data/make_events.py` → `data/events.jsonl` (5208 unique events + 25 duplicate
  deliveries, 3 tenants, 15 days; `u-ocr-ops` runs ~2000 OCR pages at 06:00 every day;
  `research-crawler` blows its $25 daily cap on the last day).

## Why
Compute is cheap; model tokens and OCR pages are not, and nothing about a working agent tells
you what it costs. Decisions have to be made per request against live state (caps, bucket,
budget) and the bill has to land on the team that incurred it, or the platform eats it. The
anomaly rule separates "this is new" (page someone) from "this happens every day at 06:00"
(have a conversation with the team's cost centre).

## How to run
```bash
python projects/p18_usage_meter/data/make_events.py                                   # regenerate the fixture
python -m projects.p18_usage_meter.cli ingest --events projects/p18_usage_meter/data/events.jsonl --state state/
python -m projects.p18_usage_meter.cli decide --state state/ --tenant acme --user u-research-1 --agent research-crawler --tokens 2000 --now 2026-10-24T12:30:00Z
python -m projects.p18_usage_meter.cli report --state state/ --csv chargeback.csv --json report.json
python -m projects.p18_usage_meter.cli anomalies --state state/ [--json]
python -m projects.p18_usage_meter.cli invoice --state state/ --tenant globex
.venv/bin/pytest projects/p18_usage_meter -q
```
Exit codes: `decide` 1 = blocked; `anomalies` 1 = anomalies found; `report` 1 = a tenant is over budget.

## Sample output
Real run on the committed `data/events.jsonl`, `data/prices.yaml` and `data/limits.yaml`.

```
$ python -m projects.p18_usage_meter.cli ingest --events projects/p18_usage_meter/data/events.jsonl --state state/
ingested 5208 new events (+$560.36), 25 duplicates skipped, 0 rejected; ledger now 5208 events, $560.36 (was $0.00); days 2026-10-10..2026-10-24
$ python -m projects.p18_usage_meter.cli anomalies --state state/
| label | entity | day | hour | spend | baseline median | threshold | recurrence |
|---|---|---|---|---|---|---|---|
| scheduled_burst | user=u-ocr-ops | 2026-10-24 | 06:00 | $29.90 | $0.02 | $0.52 | 14/14 (lo 0.78) |
| spike | user=u-research-1 | 2026-10-24 | 09:00 | $8.65 | $0.02 | $0.52 | 0/14 (lo 0.00) |
| spike | user=u-research-1 | 2026-10-24 | 10:00 | $13.81 | $0.02 | $0.52 | 0/14 (lo 0.00) |
| spike | user=u-research-1 | 2026-10-24 | 11:00 | $12.59 | $0.02 | $0.52 | 0/14 (lo 0.00) |
| scheduled_burst | agent_id=ocr-batch | 2026-10-24 | 06:00 | $29.90 | $0.06 | $0.56 | 14/14 (lo 0.78) |
| spike | agent_id=research-crawler | 2026-10-24 | 09:00 | $8.65 | $0.06 | $0.56 | 0/14 (lo 0.00) |
| spike | agent_id=research-crawler | 2026-10-24 | 10:00 | $13.81 | $0.06 | $0.56 | 0/14 (lo 0.00) |
| spike | agent_id=research-crawler | 2026-10-24 | 11:00 | $12.59 | $0.06 | $0.56 | 0/14 (lo 0.00) |
(exit 1: anomalies found)
$ python -m projects.p18_usage_meter.cli decide --state state/ --tenant acme --user u-research-1 --agent research-crawler --tokens 2000 --now 2026-10-24T12:30:00Z
BLOCK: agent_daily_cap: research-crawler spent $35.05 + $0.0060 > $25.00 (est $0.0060); retry after 41400s
remaining: agent_daily_usd=0.00, user_daily_usd=0.00, tenant_monthly_usd=2428.18, bucket_tokens=20.00
(exit 1: blocked)
$ python -m projects.p18_usage_meter.cli decide --state state/ --tenant globex --user u-ops-1 --agent assistant --tokens 2000 --now 2026-10-24T12:30:00Z
ALLOW: ok (est $0.0060)
remaining: agent_daily_usd=24.51, user_daily_usd=29.87, tenant_monthly_usd=25.57, bucket_tokens=19.00
WARNING budget_80pct: globex at 94.9% of $500 for 2026-10
(exit 0; a second decide one minute later prints no warning: the 80% alert fired once)
$ python -m projects.p18_usage_meter.cli report --state state/ --csv chargeback.csv
# Chargeback report

Total: **$560.36** across 5208 events, 4 cost centres.

## Cost centres
| cost centre | spend | share | events | cost / success |
|---|---|---|---|---|
| CC-100 | $25.20 | 4.5% | 1473 | $0.02 |
| CC-110 | $46.61 | 8.3% | 803 | $0.09 |
| CC-200 | $474.43 | 84.7% | 2079 | $0.02 |
| CC-300 | $14.12 | 2.5% | 853 | $0.02 |

### CC-100
| team | spend | share | by kind | by model | top agents | top users | cost / success | anomalies |
|---|---|---|---|---|---|---|---|---|
| platform | $12.26 | 2.2% | model_call $7.65, ocr_page $3.39, tool_call $0.82, container_seconds $0.41 | claude-sonnet-5-5 $3.65, claude-opus-5-5 $3.52, claude-haiku-5-5 $0.48 | assistant $2.35, doc-summariser $2.18, ticket-triage $2.13 | u-platform-4 $3.52, u-platform-3 $3.06, u-platform-1 $3.05 | $0.02 (410/464) | - |
| support | $12.94 | 2.3% | model_call $8.40, ocr_page $3.21, tool_call $1.00, container_seconds $0.33 | claude-sonnet-5-5 $4.06, claude-opus-5-5 $3.81, claude-haiku-5-5 $0.52 | code-reviewer $2.71, doc-summariser $2.22, assistant $2.19 | u-support-4 $3.28, u-support-2 $3.28, u-support-1 $3.23 | $0.02 (457/498) | - |

### CC-110
| team | spend | share | by kind | by model | top agents | top users | cost / success | anomalies |
|---|---|---|---|---|---|---|---|---|
| research | $46.61 | 8.3% | model_call $42.22, ocr_page $3.22, tool_call $0.85, container_seconds $0.32 | claude-opus-5-5 $38.43, claude-sonnet-5-5 $3.33, claude-haiku-5-5 $0.46 | research-crawler $35.05, translation $2.37, sql-helper $2.25 | u-research-1 $38.49, u-research-4 $2.81, u-research-3 $2.69 | $0.09 (469/563) | spike:u-research-1@09h, spike:u-research-1@10h, spike:u-research-1@11h, spike:research-crawler@09h, spike:research-crawler@10h, spike:research-crawler@11h |

### CC-200
| team | spend | share | by kind | by model | top agents | top users | cost / success | anomalies |
|---|---|---|---|---|---|---|---|---|
| ops | $460.57 | 82.2% | ocr_page $450.07, model_call $8.92, tool_call $1.14, container_seconds $0.43 | claude-sonnet-5-5 $4.61, claude-opus-5-5 $3.75, claude-haiku-5-5 $0.56 | ocr-batch $446.82, code-reviewer $2.53, sql-helper $2.46 | u-ocr-ops $446.82, u-ops-2 $2.89, u-ops-4 $2.77 | $0.02 (495/558) | scheduled_burst:u-ocr-ops@06h, scheduled_burst:ocr-batch@06h |
| sales-eng | $13.85 | 2.5% | model_call $8.43, ocr_page $3.81, tool_call $1.15, container_seconds $0.47 | claude-sonnet-5-5 $4.04, claude-opus-5-5 $3.81, claude-haiku-5-5 $0.57 | doc-summariser $3.07, assistant $2.48, translation $2.28 | u-sales-eng-5 $3.16, u-sales-eng-1 $3.03, u-sales-eng-4 $3.00 | $0.02 (451/516) | - |

### CC-300
| team | spend | share | by kind | by model | top agents | top users | cost / success | anomalies |
|---|---|---|---|---|---|---|---|---|
| finance | $14.12 | 2.5% | model_call $7.88, ocr_page $4.82, tool_call $1.03, container_seconds $0.39 | claude-sonnet-5-5 $4.41, claude-opus-5-5 $2.90, claude-haiku-5-5 $0.57 | sql-helper $2.61, ticket-triage $2.55, translation $2.37 | u-finance-4 $3.24, u-finance-5 $3.09, u-finance-2 $3.01 | $0.02 (467/538) | - |

## Tenant budgets
| tenant | month | spent | budget | consumed | status |
|---|---|---|---|---|---|
| acme | 2026-10 | $71.82 | $2,500.00 | 2.9% | ok |
| globex | 2026-10 | $474.43 | $500.00 | 94.9% | warn |
| initech | 2026-10 | $14.12 | $400.00 | 3.5% | ok |

## Anomalies
| entity | day | hour | spend | baseline median | threshold | label | recurrence |
|---|---|---|---|---|---|---|---|
| user=u-ocr-ops | 2026-10-24 | 06:00 | $29.90 | $0.02 | $0.52 | scheduled_burst | 14/14 (lo 0.78) |
| user=u-research-1 | 2026-10-24 | 09:00 | $8.65 | $0.02 | $0.52 | spike | 0/14 (lo 0.00) |
| user=u-research-1 | 2026-10-24 | 10:00 | $13.81 | $0.02 | $0.52 | spike | 0/14 (lo 0.00) |
| user=u-research-1 | 2026-10-24 | 11:00 | $12.59 | $0.02 | $0.52 | spike | 0/14 (lo 0.00) |
| agent_id=ocr-batch | 2026-10-24 | 06:00 | $29.90 | $0.06 | $0.56 | scheduled_burst | 14/14 (lo 0.78) |
| agent_id=research-crawler | 2026-10-24 | 09:00 | $8.65 | $0.06 | $0.56 | spike | 0/14 (lo 0.00) |
| agent_id=research-crawler | 2026-10-24 | 10:00 | $13.81 | $0.06 | $0.56 | spike | 0/14 (lo 0.00) |
| agent_id=research-crawler | 2026-10-24 | 11:00 | $12.59 | $0.06 | $0.56 | spike | 0/14 (lo 0.00) |

wrote 36 rows to chargeback.csv
$ python -m projects.p18_usage_meter.cli invoice --state state/ --tenant globex
# Statement — tenant `globex` — 2026-10

| item | quantity | unit | events | amount |
|---|---|---|---|---|
| container_seconds | 22,510 | container seconds | 175 | $0.90 |
| model_call / claude-haiku-5-5 | 842,295 | tokens | 309 | $1.13 |
| model_call / claude-opus-5-5 | 285,943 | tokens | 114 | $7.56 |
| model_call / claude-sonnet-5-5 | 1,661,082 | tokens | 651 | $8.65 |
| ocr_page | 30,259 | ocr page | 455 | $453.88 |
| tool_call | 1,147 | tool call | 375 | $2.29 |

**Subtotal: $474.43** — 94.9% of $500 monthly budget

Chargeback by team: ops $460.57, sales-eng $13.85

```
Reading the run: both planted patterns surface with the right label — `u-ocr-ops` at 06:00 is a
`scheduled_burst` (exceeded the threshold on 14/14 baseline days, Wilson lower bound 0.78) and
`research-crawler` 09:00–11:00 is a `spike` (0/14). The crawler's next request is blocked for the
rest of the UTC day with `retry after 41400s`; globex's first request after crossing 80 % of its
$500 budget carries the one-time warning, and `report` shows the same $474.43 against the budget.
The chargeback total ($560.36) equals the sum of the 5208 priced events, and the 36 CSV rows are
the distinct (cost centre, team, kind, model) combinations.

## Design notes
* **Pricing at ingest, once.** An event carrying `cost_usd` keeps it (the gateway may know a
  negotiated rate); otherwise the meter prices it. Prices never change retroactively because the
  ledger stores the computed cost.
* **Dedup is the first check.** At-least-once delivery is the norm for usage pipelines; 25 of the
  5233 demo rows are re-deliveries and ingest skips them before pricing.
* **Hard before soft, bucket last.** Per-request limits and daily caps are evaluated before the
  token bucket so a blocked request does not consume a rate-limit token; the soft budget runs
  last and only adds warnings. `retry_after_s` tells the caller when a retry can succeed instead
  of letting it hammer the gateway.
* **Population baseline for anomalies.** A user whose entire history is the 06:00 burst would be
  "normal" against their own baseline; comparing to every user's non-zero hours makes the burst
  visible, and the per-entity recurrence check is what turns it into `scheduled_burst` instead of
  paging on-call every morning. Fewer than 8 baseline hours means no verdict, not a guess.
* **Chargeback is an aggregate of the same ledger.** No second data path: totals, shares, CSV
  rows and invoices are all `Meter.aggregate` over the deduplicated events, which is why the
  tests can assert they sum to `Meter.total()`.
* **evalkit reuse:** `estimate_cost` (model pricing), `wilson_ci` (recurrence bound), `read_jsonl`
  / `write_jsonl` (state), `md_table` / `pct` (reports).
