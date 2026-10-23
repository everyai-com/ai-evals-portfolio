# 15. Public Eval Methodology Teardowns

Three deep-dives written the way a senior evals engineer publishes: every rubric, judge prompt,
weight, threshold and statistic is quoted from the code in this repo, with the seed that
reproduces each number.

| Teardown | What it dissects | Code it cites |
|----------|------------------|---------------|
| [01: Grading the trajectory, not the answer](01_trajectory_grading_methodology.md) | golden-path DAG, verdict taxonomy, scoring weights, safety gates, distance metrics, counterfactual replay | `p01_trajectory_grader`, `p11_counterfactual_replay` |
| [02: Calibrating an LLM judge against 500 human anchors](02_judge_calibration_methodology.md) | annotation guidelines, kappa ceiling, the exact judge prompt, position / verbosity / self-preference tests, debiasing | `p03_calibrated_judge`, `evalkit.metrics` |
| [03: Agent vs naive baseline, a contamination-aware benchmark report](03_agent_vs_baseline_benchmark.md) | paired bootstrap, McNemar, Holm, power, seed variance, contamination tiers, red-team resilience, reproducible seeds | `p07_significance_engine`, `p14_contamination_checker`, `p08_agent_redteam_fuzzer`, `p05_rag_adversarial` |

## Reproduce every number
```bash
.venv/bin/python -m projects.p15_methodology_teardowns.reproduce   # writes reports/*.md
```
`tests/test_citations.py` parses the three documents and fails if any cited file path or
`module:symbol` no longer exists, so the teardowns cannot silently rot.
