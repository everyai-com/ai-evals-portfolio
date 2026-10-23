# p04 · CI/CD Regression Gate

**What.** A GitHub Action (plus a reusable composite action) that runs a parameterised eval on every
pull request and blocks the merge when task success drops more than 2 points versus `main`, or when
p95 latency / cost-per-task spike. Failing cases are posted to the PR as a comment that updates in
place on re-runs.

**Why.** "The eval went from 90 to 88" is not a regression unless the sample size can tell 88 from
90. The gate therefore pairs the PR run with the baseline on the *same* items, bootstraps the delta,
and only blocks when the drop is both above the budget **and** its 95% CI excludes zero. Latency and
cost are noisy day-to-day, so they need `k` consecutive breaches (hysteresis) before they block and
`k` clean runs before they clear. Items that flip between seeds are excluded from the decision and
reported separately so one flaky example cannot hold a merge hostage.

## Components

| piece | file | what it does |
|---|---|---|
| `EvalConfig` | `core.py` | `data/eval_config.yaml`: datasets, model, params, seeds, cost budget, gate thresholds |
| `EvalRunner` | `core.py` | runs every (example, seed) through a SUT callable, writes `results.json` with per-example `{id, success, latency_ms, cost_usd, tags}` and aggregates |
| `EvalCache` | `core.py` | content-addressed cache in `.eval-cache/`, key = `sha256(prompt, model, params, dataset_row)`; hit rate reported |
| cost budget | `core.py` | projected spend (cache misses × `est_per_call_usd`) and running spend are checked against `max_cost_usd`; exceeding it exits 2 |
| `Gate`, `GateState` | `core.py` | success / latency / cost verdicts, hysteresis counters in `gate_state.json`, flaky exclusion |
| `render_comment` | `core.py` | markdown PR comment (badge, delta table with CIs, top-N failing cases, flaky list, cache + cost) |
| `post_comment.py` | | posts or PATCHes the PR comment via GitHub REST (urllib); hidden marker makes it idempotent |
| demo SUT | `demo_sut.py` | MockLLM-backed agent whose behaviour is picked by `--variant` / `$EVAL_VARIANT` |
| workflow | `.github/workflows/eval-gate.yml` | PR job (uv, cache restore, run, gate, comment, artifact) + `update-baseline` job on push to main |
| composite action | `.github/actions/eval-gate/action.yml` | inputs `config`, `baseline`, `max-drop`, `token`, `variant`; outputs `blocked`, `task_success`, … |

Demo variants (`demo_sut.py`): `main` (90% success), `pr_good` (+1 pt), `pr_minor` (−1 pt, warns),
`pr_regressed` (−5 pts, blocks), `pr_slow` (p95 ×2, blocks on the 2nd consecutive run), `pr_flaky`
(15 items flip between seeds; excluded).

## How to run

```bash
# from the repo root; P is just an abbreviation (all --config/--baseline flags default to these paths)
P=projects/p04_ci_regression_gate
python -m projects.p04_ci_regression_gate.cli baseline --config $P/data/eval_config.yaml --out $P/baseline/main.json
python -m projects.p04_ci_regression_gate.cli run  --config $P/data/eval_config.yaml --variant pr_regressed --out results.json
python -m projects.p04_ci_regression_gate.cli gate --results results.json --baseline $P/baseline/main.json \
    --config $P/data/eval_config.yaml --comment gate_comment.md --state gate_state.json
# exit 0 = pass, 1 = blocked, 2 = usage error / cost budget exceeded
python -m projects.p04_ci_regression_gate.post_comment --repo owner/name --pr 12 --body-file gate_comment.md
.venv/bin/pytest projects/p04_ci_regression_gate -q
```

Reuse in another repo:

```yaml
- uses: ./.github/actions/eval-gate
  with:
    config: projects/p04_ci_regression_gate/data/eval_config.yaml
    baseline: projects/p04_ci_regression_gate/baseline/main.json
    max-drop: "2"
    token: ${{ secrets.GITHUB_TOKEN }}
```

## Sample output

Real run on the committed fixtures (300 tasks × 2 seeds):

```text
$ python -m projects.p04_ci_regression_gate.cli run --config $P/data/eval_config.yaml --variant pr_regressed --out results.json
## eval run `pr_regressed`

- task_success: 85.0% over 300 items x 2 seeds
- latency p50/p95: 115 / 169 ms
- cost: $0.0045 ($0.000015 per task)
- cache: 0 hits / 600 misses (hit rate 0%)
- wrote results.json

$ python -m projects.p04_ci_regression_gate.cli run --config $P/data/eval_config.yaml --variant pr_regressed --out results.json   # second run, warm cache
## eval run `pr_regressed`

- task_success: 85.0% over 300 items x 2 seeds
- latency p50/p95: 115 / 169 ms
- cost: $0.0045 ($0.000015 per task)
- cache: 600 hits / 0 misses (hit rate 100%)
- wrote results.json

$ python -m projects.p04_ci_regression_gate.cli gate --results results.json --baseline $P/baseline/main.json --config $P/data/eval_config.yaml --comment gate_comment.md --state gate_state.json; echo "exit=$?"
<!-- eval-gate-comment -->
## Eval gate: ❌ **BLOCKED** — merge gate failed

`pr_regressed` vs baseline `main` · model `mock-system-one` · 300 paired items · seeds [0, 1]

| metric | baseline | PR | Δ | 95% CI | threshold | status | note |
|---|---|---|---|---|---|---|---|
| task_success | 90.0% | 85.0% | -5.0 pts | [-7.7, -2.7] pts | drop <= 2 pts | ❌ block | drop of 5.0 pts > 2 and significant (95% CI excludes 0, p=0.0002) |
| latency_p95 | 169 ms | 169 ms | +0.0% | — | increase <= 25% | ✅ pass | within threshold |
| cost_per_task | $0.00002 | $0.00002 | +0.0% | — | increase <= 30% | ✅ pass | within threshold |

### Top failing cases (10 shown, 10 new vs baseline)

| id | new? | input | expected | actual | tags |
|---|---|---|---|---|---|
| t015 | 🔴 regression | Reverse the word 'olive'. Answer with just the word. | evilo | olivex | reverse,easy |
| t034 | 🔴 regression | Uppercase the word 'toppingsauce'. Answer with just the wor… | TOPPINGSAUCE | ECUASGNIPPOTx | upper,hard |
| t035 | 🔴 regression | How many letters are in 'mushroom'? Answer with just the nu… | 8 | 80 | count,easy |
| t046 | 🔴 regression | Uppercase the word 'crust'. Answer with just the word. | CRUST | TSURCx | upper,easy |
| t055 | 🔴 regression | What is 48 - 14? Answer with just the number. | 34 | 340 | sub,easy |
| t071 | 🔴 regression | How many letters are in 'sauce'? Answer with just the numbe… | 5 | 50 | count,easy |
| t084 | 🔴 regression | What is 39 + 7? Answer with just the number. | 46 | 460 | add,easy |
| t085 | 🔴 regression | What is 20 - 3? Answer with just the number. | 17 | 170 | sub,easy |
| t104 | 🔴 regression | What is 5 * 2? Answer with just the number. | 10 | 100 | mul,easy |
| t127 | 🔴 regression | What is 10 - 47? Answer with just the number. | -37 | 73-x | sub,easy |

### Run stats

- cache: 600 hits / 0 misses (hit rate 100%)
- cost: $0.0045 total, $0.000015 per task over 600 calls
- latency: p50 115 ms, p95 169 ms

<sub>Decision rule: block when task-success drop > max_drop *and* the paired-bootstrap 95% CI excludes zero; latency/cost block after k consecutive breaches (hysteresis).</sub>

exit=1

$ # pr_slow twice against the same gate_state.json (latency hysteresis k=2)
## Eval gate: ⚠️ **PASSED with warnings**
| latency_p95 | 169 ms | 343 ms | +103.0% | — | increase <= 25% | ⚠️ warn | +103.0% > 25% (breach 1/2; blocks at 2) |
exit=0
## Eval gate: ❌ **BLOCKED** — merge gate failed
| latency_p95 | 169 ms | 343 ms | +103.0% | — | increase <= 25% | ❌ block | +103.0% > 25% on 2 consecutive run(s) |
exit=1
```

## Design notes

- **Decision rule for task success.** Per item, success is the mean over seeds. The PR and baseline
  are paired on common, non-flaky item ids and `evalkit.stats.paired_bootstrap` (10k resamples by
  default, 5k in the demo config) gives Δ and its 95% percentile CI. Block iff
  `drop > max_drop` **and** `ci_hi < 0`. A drop above the budget whose CI includes zero is reported as
  *"not significant at this n"* and only warns, which is the honest answer at small n.
- **Hysteresis.** `gate_state.json` keeps, per metric, `consecutive_breaches`, `consecutive_recoveries`
  and `blocking`. A metric enters the blocking state after `k` consecutive breaches and leaves it after
  `k` consecutive recoveries (`k=1` for success, `2` for latency and cost by default). The state
  file should be persisted between runs (cache it, or commit it on the PR branch).
- **Flaky detection.** With `s` seeds the per-item success vector has population std `σ`; an item
  is flaky when `σ > flake_threshold` (0 by default, i.e. any disagreement). Flaky items are listed in
  the comment and excluded from the paired test so that they neither block nor mask a regression.
- **Cache correctness.** The key hashes the prompt, the model, *all* params (including the variant
  and seed) and the dataset row, so changing any of them is a miss and a PR can never reuse `main`'s
  answers. `actions/cache` restores `.eval-cache/` keyed on the dataset hash.
- **Cost budget.** Before any call the runner projects `misses × est_per_call_usd`; if that exceeds
  `max_cost_usd` it aborts with exit 2 and a message, so a config typo cannot burn a budget. The
  running total is checked after every uncached call as well.
- **Latency in the demo** is simulated by the SUT (gamma jitter around a per-difficulty base) so the
  suite is fast and reproducible; `EvalRunner` uses wall-clock latency whenever a SUT does not report one.
- **Not re-implemented:** bootstrap (`evalkit.stats`), markdown tables (`evalkit.report`), JSONL and
  hashing (`evalkit.io`), the mock model (`evalkit.llm`).
