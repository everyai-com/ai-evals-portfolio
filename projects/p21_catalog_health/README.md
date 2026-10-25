# p21 — Marketplace and registry health

**The mistake:** "community marketplaces are hard" — ~90 % of the catalog went untouched: to-do
apps, hello-worlds 0 through 100, everyone's first agent. The fix was curation and scoping:
featured apps, teams, stars, usage metrics, and the tenet **"isolate what you see to what you
need"**.

The tests that prove the lesson is encoded: `test_fixture_is_a_long_tail` (>= 88 % of items have
<= 1 user, Gini > 0.9), `test_scoped_view_shrinks_by_over_90pct_and_keeps_featured_and_team_items`,
`test_dormancy_rule_respects_stars_and_window`, `test_retire_queue_one_row_per_dormant_item_with_owner`.

## What

| piece | does |
|---|---|
| `CatalogItem` / `load_catalog` | id, name, owner, team, kind (`app|agent|mcp_server|automation`), created, last_used, invocations_30d, distinct_users_30d, stars, versions, tags, risk_tier, plus `invocations_7d` / `invocations_prev_7d` and `users_by_team` |
| `long_tail_stats` | share of items with 0 / <= 1 / <= 5 users, Gini of users and invocations, Pareto ("top 1 % / 10 % of items carry y % of usage", "x % of items carry 80 %") |
| `duplicate_clusters` | normalised-name clusters: "todo app v2", "To-Do", "to do list" are one key; "hello world 0..100" are one key |
| `DormancyRule` / `retirement_queue` | dormant = idle >= N days (or never used) **and** fewer than `min_stars` stars; one queue row per item with an owner notification |
| `scoped_view` | per user (team, role): keep org-featured, team-featured, team-owned-and-used, used by >= 2 teammates, role-tagged, or global top-K; rank by score; report before vs after |
| `Featured` / `rank_usage` / `rising_items` | admin pins from `data/featured.yaml`, usage-weighted top list, week-over-week risers |
| `HealthReport` | all of the above as markdown or `--json` |

## Why

A registry with 2,000 entries and 12 that matter is worse than a list of 12: nobody finds the 12.
The long-tail numbers say how bad it is, dormancy and duplicates say what to remove, and scoping
says what each person should see instead of everything.

## How to run

```bash
D=projects/p21_catalog_health/data
.venv/bin/python -m projects.p21_catalog_health.cli report --catalog $D/catalog.jsonl [--featured $D/featured.yaml] [--days 60] [--json [out.json]] [--max-untouched 0.95]
.venv/bin/python -m projects.p21_catalog_health.cli dormant --catalog $D/catalog.jsonl --days 60 --out retire_queue.jsonl
.venv/bin/python -m projects.p21_catalog_health.cli view --catalog $D/catalog.jsonl --team sales --role manager [--limit 20]
.venv/bin/python -m projects.p21_catalog_health.cli duplicates --catalog $D/catalog.jsonl
.venv/bin/python -m projects.p21_catalog_health.data.make_catalog --seed 0 --n 2000   # regenerate catalog.jsonl + featured.yaml
.venv/bin/pytest projects/p21_catalog_health -q
```

Dates are evaluated against `--today` (default 2026-10-24, the fixture's as-of). `report` exits 1
when the share of items with <= 1 user exceeds `--max-untouched`. `evalkit catalog ...` is the alias.

## Sample output

`report --catalog data/catalog.jsonl`:

```
# Catalog health: 2,000 items as of 2026-10-24

| kind | items |
|---|---|
| agent | 521 |
| app | 492 |
| automation | 470 |
| mcp_server | 517 |

## Long tail (distinct users, last 30 days)

| stat | value |
|---|---|
| items with 0 users | 61.9% |
| items with <= 1 user | 91.4% |
| items with <= 5 users | 92.2% |
| Gini (users) | 0.942 |
| Gini (invocations) | 0.969 |
| top 1% of items carry | 69.6% of invocations |
| top 10% of items carry | 98.2% of invocations |
| items needed for 80% of usage | 2.5% |

## Duplicate-name clusters (111; 1,502 items)

| normalised | items | users | names |
|---|---|---|---|
| helloworld | 389 | 118 | hello world 0; hello world 1; hello world 10; hello world 100; hello world 11 ... |
| todo | 184 | 62 | TODO (copy); To-Do; Todo App; my todo; to do list ... |
| first | 96 | 37 | My First Agent; first agent (copy); my first agent |
| chat | 71 | 27 | Chat Bot; chat-bot |
| notes | 65 | 22 | Notes app; notes |
| demo | 51 | 17 | demo; demo app |
| untitled | 42 | 8 | Untitled app; untitled |
| playground | 30 | 4 | playground |

## Dormancy: 992 items (49.6%) idle >= 60 days with < 1 star(s)

## Top by usage

| id | name | team | users | invocations | stars |
|---|---|---|---|---|---|
| item-0000 | Sales Deal Desk | sales | 860 | 3529 | 112 |
| item-0001 | HR Policy Assistant | hr | 740 | 5435 | 97 |
| item-0002 | IT Helpdesk Agent | ops | 690 | 6538 | 163 |
| item-0003 | Expense Bot | finance | 520 | 5959 | 23 |
| item-0004 | Onboarding Guide | hr | 480 | 3229 | 97 |
| item-0005 | Code Review Agent | engineering | 410 | 1841 | 115 |
| item-0006 | Meeting Notes Summariser | ops | 390 | 4556 | 120 |
| item-0007 | Benefits FAQ | hr | 330 | 1756 | 189 |
| item-0008 | Contract Clause Finder | legal | 260 | 1981 | 39 |
| item-0009 | Customer Churn Signals | support | 240 | 1669 | 160 |

## Rising (week over week)

| id | name | team | 7d | prev 7d | growth |
|---|---|---|---|---|---|
| item-0153 | engineering digest v3 | engineering | 64 | 8 | +700% |
| item-0062 | hr bot | hr | 40 | 5 | +700% |
| item-0171 | ops bot v3 | ops | 56 | 8 | +600% |
| item-0059 | hr digest v2 | hr | 36 | 6 | +500% |
| item-0047 | finance faq beta | finance | 30 | 5 | +500% |
| item-0053 | hr helper v3 | hr | 30 | 5 | +500% |
| item-0056 | sales dashboard v2 | sales | 30 | 5 | +500% |
| item-0032 | engineering tracker v3 | engineering | 25 | 5 | +400% |
| item-0031 | support planner | support | 18 | 9 | +100% |

## What each team sees before vs after scoping

| team | before | after | hidden |
|---|---|---|---|
| engineering | 2000 | 155 | 92.2% |
| finance | 2000 | 145 | 92.8% |
| hr | 2000 | 151 | 92.5% |
| legal | 2000 | 155 | 92.2% |
| marketing | 2000 | 151 | 92.5% |
| ops | 2000 | 149 | 92.5% |
| sales | 2000 | 154 | 92.3% |
| support | 2000 | 150 | 92.5% |
```

`view --catalog data/catalog.jsonl --team sales --role manager --limit 10`:

```
# Scoped view for team=sales role=manager: 2000 items -> 164 (91.8% hidden)

| # | id | name | kind | team users | users | stars | score | why |
|---|---|---|---|---|---|---|---|---|
| 1 | item-0000 | Sales Deal Desk | agent | 402 | 860 | 112 | 43260 | org-featured,team-featured,team-owned,team-used,top-global |
| 2 | item-0002 | IT Helpdesk Agent | agent | 45 | 690 | 163 | 7266 | org-featured,team-used,top-global |
| 3 | item-0001 | HR Policy Assistant | agent | 49 | 740 | 97 | 7007 | org-featured,team-used,top-global |
| 4 | item-0004 | Onboarding Guide | app | 37 | 480 | 97 | 5799 | org-featured,team-used,top-global |
| 5 | item-0005 | Code Review Agent | agent | 35 | 410 | 115 | 5775 | org-featured,team-used,top-global |
| 6 | item-0003 | Expense Bot | automation | 35 | 520 | 23 | 4860 | org-featured,team-used,top-global |
| 7 | item-0006 | Meeting Notes Summariser | app | 34 | 390 | 120 | 4774 | team-used,role,top-global |
| 8 | item-0007 | Benefits FAQ | app | 23 | 330 | 189 | 4311 | team-used,top-global |
| 9 | item-0009 | Customer Churn Signals | app | 25 | 240 | 160 | 4215 | team-used,top-global |
| 10 | item-0027 | sales triage v3 | app | 26 | 56 | 4 | 3521 | team-featured,team-owned,team-used |
```

`dormant --catalog data/catalog.jsonl --days 60 --out retire_queue.jsonl`:

```
# Retirement queue: 992 of 2000 items idle >= 60 days with < 1 star(s)

| id | name | owner | team | last used | idle days |
|---|---|---|---|---|---|
| item-0172 | user2247's app | user2247@example.com | marketing | never | - |
| item-0173 | My First Agent | user2133@example.com | sales | never | - |
| item-0175 | user2796's app | user2796@example.com | support | never | - |
| item-0179 | hello world 13 | user2187@example.com | ops | never | - |
| item-0180 | sample | user2910@example.com | engineering | never | - |
...
queue written: 992 rows -> retire_queue.jsonl
```

Each queue row carries `owner`, `last_used`, `days_idle`, `reason` and a ready-to-send
`notification` ("... will be archived in 14 days unless you use it or star it").

## Design notes

- **Gini** uses the sorted-index form `2 Σ i·x(i) / (n Σ x) − (n+1)/n`; one item holding everything
  gives `(n−1)/n`, which is why the test expects 0.75 for `[0, 0, 0, 10]`.
- **Scoping rule.** Ownership alone earns nothing: a team's own dead to-do apps are hidden from it
  too (`team-owned` requires at least one teammate to have used it this month). Without that the
  reduction is ~79 %; with it ~92 %. `--min-team-users` (default 2) is the bar for `team-used`.
- **Dormancy** keys on `last_used` (the date of the last invocation) rather than `invocations_30d`
  so windows longer than 30 days work, and a single star vetoes auto-retirement: stars are the
  cheapest "a human wants this kept" signal the platform has.
- **Name normalisation** strips version/copy/test/app/agent/list tokens and bare numbers, folds
  "to-do"/"to do" and "hello-world", then drops non-alphanumerics, so the cluster key for
  "Todo App v2 (copy)" is `todo`.
- **Rising** needs at least `min_inv` (10) invocations this week so a 0 -> 3 blip does not rank.
- **Fixture** (`make_catalog.py`, seed 0): 12 hubs with 150-860 users, 160 middle-band team tools
  (8 % of them planted as rising), and a 1,830-item tail of 0- or 1-user items.
- `evalkit` reuse: `read_jsonl`, `write_jsonl`, `md_table`, `pct`.
