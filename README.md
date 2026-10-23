# AI Evals Engineering Portfolio

Fifteen working systems (plus a shared core and a unified harness) that measure, secure and gate
LLM agents. Every project runs **offline and deterministically** against a scripted mock model,
has a CLI, committed fixtures, a README with a real sample run, and tests that assert the exact
claim the project makes. Set `EVALKIT_LLM=anthropic` to point the same code at a live model.

See [LEARNING.md](LEARNING.md) for the checklist that maps the two source posts (15 systems,
12 stages) to the code and tests.

## Map

| # | Project | One line |
|---|---------|----------|
| core | `evalkit/` | JSONL I/O, P/R/F1, kappa, Wilson + bootstrap CIs, IR metrics, paired bootstrap / McNemar / power, trajectory schema, MockLLM, pytest plugin |
| 1 | `p01_trajectory_grader` | Score each tool call against a golden DAG; hallucinated params or skipped safety gates fail the run |
| 2 | `p02_shadow_router` | Mirror 5% of traffic to a candidate model in the background; diff trajectories; cost/quality report |
| 3 | `p03_calibrated_judge` | 500 human-labelled anchors; measure position, verbosity, self-preference bias; kappa vs humans |
| 4 | `p04_ci_regression_gate` | GitHub Action: block merge on a significant >2-pt drop or p95 spike; hysteresis; flaky detection; posts failing cases |
| 5 | `p05_rag_adversarial` | Separate retrieval / generation gates; adversarial variants force abstention; confident-lie rate |
| 6 | `p06_dpo_flywheel` | Thumbs-down → PII-scrubbed DPO pairs → golden cases → nightly LoRA (dry-run without GPU) → promote if not worse |
| 7 | `p07_significance_engine` | "Model B wins by 3.2% ± 1.1% (p<0.05)"; power warnings; Holm correction; seed variance |
| 8 | `p08_agent_redteam_fuzzer` | 500 generated attacks (injection, malformed schemas, loops, exfil); taxonomy; repro files; guardrail ablation |
| 9 | `p09_drift_monitor` | Nightly 5% stratified sample; offline trajectory evals; CUSUM / PSI / CI alerts |
| 10 | `p10_pareto_dashboard` | Cost vs success Pareto frontier per tenant; price-drop sensitivity; dependency-free HTML |
| 11 | `p11_counterfactual_replay` | Record every hop; swap one node; replay the rest from cache; bisect the step that broke it |
| 12 | `p12_synthetic_edgecases` | Mutation operators + metamorphic relations + model-driven OOD cases; difficulty filter; coverage matrix |
| 13 | `p13_context_eviction` | Fill memory with noise; test truncate / summarise / retrieve strategies on fact recall and staleness |
| 14 | `p14_contamination_checker` | 8/13-gram, MinHash LSH and embedding tiers; decontaminated split; contamination-aware reporting |
| 15 | `p15_methodology_teardowns` | Three published-style deep-dives: rubrics, judge prompts, statistical framework, seeds |
| 16 | `p16_unified_harness` | promptfoo / DeepEval / RAGAS adapters (with offline fallbacks) behind one CLI; cross-framework agreement |

## Test inventory

258 tests, all offline, full suite in about 16 s.

| Suite | Tests |
|-------|-------|
| `tests` | 17 |
| `projects/p01_trajectory_grader` | 19 |
| `projects/p02_shadow_router` | 19 |
| `projects/p03_calibrated_judge` | 13 |
| `projects/p04_ci_regression_gate` | 16 |
| `projects/p05_rag_adversarial` | 15 |
| `projects/p06_dpo_flywheel` | 14 |
| `projects/p07_significance_engine` | 12 |
| `projects/p08_agent_redteam_fuzzer` | 12 |
| `projects/p09_drift_monitor` | 17 |
| `projects/p10_pareto_dashboard` | 8 |
| `projects/p11_counterfactual_replay` | 12 |
| `projects/p12_synthetic_edgecases` | 13 |
| `projects/p13_context_eviction` | 14 |
| `projects/p14_contamination_checker` | 13 |
| `projects/p15_methodology_teardowns` | 32 |
| `projects/p16_unified_harness` | 12 |

## Quick start
```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/evalkit --help          # umbrella CLI: evalkit <project> ...
```

## Conventions
[CONTRIBUTING.md](CONTRIBUTING.md): layout, offline rule, exit codes, "tests prove the README".
