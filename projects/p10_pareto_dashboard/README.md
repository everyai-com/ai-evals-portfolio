# p10 Cost-Quality Pareto Dashboard

## What
Maps the exact tradeoff between inference cost and task success for every routing policy, per
tenant and pooled. From a JSONL of eval results (one row per tenant x policy x task) it computes
per-policy success rate with Wilson 95% CI, mean cost/task, p95 latency, cost per successful task
and tokens; finds the **Pareto frontier** (policies no other policy beats on both cost and
success); and for every dominated policy says which frontier policy to move to and what that
saves. A price-drop sensitivity re-runs the frontier with System-Two's price scaled (0.5x, 0.25x)
to show what happens when a frontier lab cuts prices.

Outputs: a markdown report (stdout), `metrics.json`, and a self-contained `dashboard.html`
(inline SVG scatter per tenant + global, log-scale cost, CI error bars, dashed frontier line,
one table per panel, and a vanilla-JS price slider that re-computes the frontier from the
embedded JSON). No external resources, works from `file://` with no network, light/dark via CSS
variables.

## Why
"Use the cheap model" and "use the best model" are both usually wrong; the frontier shows which
cascade threshold or router sits at the knee, what the next success point costs at the margin,
and which policies are simply dominated. Since the frontier is a function of prices, the slider
turns a vendor price cut into an immediate routing decision.

## How to run
```bash
.venv/bin/python -m projects.p10_pareto_dashboard.cli build --results projects/p10_pareto_dashboard/data/results.jsonl --out dashboard.html --json metrics.json [--price-scale 0.5]
.venv/bin/python -m projects.p10_pareto_dashboard.data.make_results --seed 0   # regenerate the fixture
.venv/bin/pytest projects/p10_pareto_dashboard -q
```
`evalkit pareto build ...` is the same CLI. Input rows need `tenant, routing_policy,
model_or_cascade, task_id, success, cost_usd, latency_ms, tokens`; an optional `cost_breakdown`
({model: usd}) makes price scaling exact (without it, rows whose `model_or_cascade` is exactly the
System-Two model are scaled and cascades are not).

## Sample output
Real run on the committed fixture (`data/results.jsonl`: 3 tenants x 6 policies x 150 shared
tasks from `data/make_results.py`, seed 0; prices from `evalkit.llm.PRICES_PER_MTOK`). The first
tenant block and the sensitivity table are shown; the other blocks have the same shape.

```
$ python -m projects.p10_pareto_dashboard.cli build --results projects/p10_pareto_dashboard/data/results.jsonl --out dashboard.html --json metrics.json
# Cost-quality Pareto report (System-Two price scale 1x, 2700 eval rows)

## pizza-uk

| policy | n | success (95% CI) | cost/task | cost/success | p95 ms | tokens | frontier | move to | save/task | save % |
|---|---|---|---|---|---|---|---|---|---|---|
| system_one_only | 150 | 58.7% (50.7%-66.2%) | $0.000304 | $0.000518 | 1019 | 724 | yes | - | - | - |
| cascade_confidence_0.5 | 150 | 66.0% (58.1%-73.1%) | $0.001271 | $0.001926 | 2437 | 826 | yes | - | - | - |
| cascade_confidence_0.7 | 150 | 83.3% (76.6%-88.4%) | $0.003385 | $0.004062 | 3569 | 1051 | yes | - | - | - |
| router_classifier | 150 | 71.3% (63.6%-78.0%) | $0.003516 | $0.004929 | 3046 | 748 | no | cascade_confidence_0.7 | $0.000131 | 3.7% |
| cascade_confidence_0.9 | 150 | 85.3% (78.8%-90.1%) | $0.005861 | $0.006868 | 3844 | 1314 | yes | - | - | - |
| system_two_only | 150 | 84.0% (77.3%-89.0%) | $0.007359 | $0.008761 | 3577 | 780 | no | cascade_confidence_0.9 | $0.001498 | 20.4% |

Frontier (cheapest first): system_one_only -> cascade_confidence_0.5 -> cascade_confidence_0.7 -> cascade_confidence_0.9

Marginal cost between adjacent frontier points:

| from | to | +success pts | +cost/task | cost per +1 success pt |
|---|---|---|---|---|
| system_one_only | cascade_confidence_0.5 | +7.3 | $+0.000967 | $0.000132 |
| cascade_confidence_0.5 | cascade_confidence_0.7 | +17.3 | $+0.002113 | $0.000122 |
| cascade_confidence_0.7 | cascade_confidence_0.9 | +2.0 | $+0.002476 | $0.001238 |

Recommendations:
- `system_two_only` is dominated: switch to `cascade_confidence_0.9` for 20.4% lower cost ($1.50 per 1k tasks) and +1.3 success points (n=150 tasks).
- `router_classifier` is dominated: switch to `cascade_confidence_0.7` for 3.7% lower cost ($0.13 per 1k tasks) and +12.0 success points (n=150 tasks).

...(pizza-us, wings-ca and pooled blocks omitted)...
## Price-drop sensitivity (frontier per System-Two price scale)

| tenant | 1x | 0.5x | 0.25x |
|---|---|---|---|
| pizza-uk | system_one_only, cascade_confidence_0.5, cascade_confidence_0.7, cascade_confidence_0.9 | system_one_only, cascade_confidence_0.5, router_classifier, cascade_confidence_0.7, cascade_confidence_0.9 | system_one_only, cascade_confidence_0.5, router_classifier, cascade_confidence_0.7, cascade_confidence_0.9 |
| pizza-us | system_one_only, cascade_confidence_0.5, router_classifier, cascade_confidence_0.7, cascade_confidence_0.9 | system_one_only, cascade_confidence_0.5, router_classifier, cascade_confidence_0.7, cascade_confidence_0.9 | system_one_only, cascade_confidence_0.5, router_classifier, cascade_confidence_0.7, cascade_confidence_0.9 |
| wings-ca | system_one_only, cascade_confidence_0.5, cascade_confidence_0.7, cascade_confidence_0.9 | system_one_only, cascade_confidence_0.5, router_classifier, cascade_confidence_0.7, cascade_confidence_0.9 | system_one_only, cascade_confidence_0.5, router_classifier, cascade_confidence_0.7, system_two_only, cascade_confidence_0.9 |
| All tenants (pooled) | system_one_only, cascade_confidence_0.5, cascade_confidence_0.7, cascade_confidence_0.9 | system_one_only, cascade_confidence_0.5, router_classifier, cascade_confidence_0.7, cascade_confidence_0.9 | system_one_only, cascade_confidence_0.5, router_classifier, cascade_confidence_0.7, cascade_confidence_0.9 |

dashboard -> ./dashboard.html; metrics -> ./metrics.json
```

`dashboard.html` built from the same data was opened from a `file://` URL with no network: four
`<svg>` panels, no console errors, and dragging the slider to 0.25x moves `router_classifier`
onto every tenant's frontier and `system_two_only` onto wings-ca's, exactly as the table above
predicts. Reading the charts:

- filled point = on the frontier, hollow = dominated; vertical bars = Wilson 95% CI
- dashed line = frontier, cheapest to best
- hover a point for its numbers; the table under each chart lists "move to" and savings

## Design notes
* **Frontier.** Sort policies by (cost asc, success desc) and keep each one whose success beats
  every cheaper policy. Ties: equal cost and success keeps the first; equal success at higher cost
  is dominated. `tests/test_frontier.py` pins this on a hand-built 5-point case.
* **Savings.** For a dominated policy P, target = cheapest frontier policy with success >= P's
  (always exists: whatever dominates P is on the frontier). Savings = cost_P - cost_target per task
  (reported per 1k tasks too) plus the success gain. Distance to frontier = Euclidean distance to
  the nearest frontier vertex in (log10 cost, success) space.
* **Marginal cost** between adjacent frontier points = delta cost / delta success points; the
  jump from `cascade_confidence_0.7` to `_0.9` costs ~10x more per success point than the
  previous step, which is the knee of the curve.
* **Price sensitivity** scales only the System-Two component of each row's `cost_breakdown`, so
  cascades (which pay for both legs) re-price correctly. Python precomputes 1x/0.5x/0.25x for the
  report; the browser recomputes any scale in [0.05, 1.5] from per-policy `mean_cost_one` and
  `mean_cost_two` embedded in the JSON, using the same frontier algorithm in JS.
* **Generator.** Each task draws a difficulty from a per-tenant Beta; System-One success is
  `0.97 - 0.75 d`, System-Two `0.99 - 0.25 d`; a pre-drawn uniform per (task, model) makes the
  cascade's System-Two leg agree with `system_two_only` on the same task (paired design).
  Confidence is correlated with correctness so cascades escalate the right tasks most of the time;
  the router sees difficulty with N(0, 0.2) noise.
