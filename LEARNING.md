# Learning checklist: both posts, mapped to code

Two source posts drove this repo: **"15 systems an AI Evals Engineer must build"** and
**"6 months to become an AI Evals Engineer" (12 stages)**. Every item below points at the code
that implements it and the test that proves it. Nothing is a slide; everything runs offline with
`pytest`.

## Part A: the 15 systems

| # | System | Project | Proof (test that asserts the claim) |
|---|--------|---------|-------------------------------------|
| 1 | Trajectory Grading Engine | `projects/p01_trajectory_grader` | hallucinated param fails run; skipped safety gate zeroes score |
| 2 | Shadow Routing Comparator | `projects/p02_shadow_router` | caller always gets primary; shadow errors never surface; 5% deterministic sampling |
| 3 | Calibrated LLM-as-a-Judge | `projects/p03_calibrated_judge` | injected position / verbosity / self-preference bias is detected vs 500 anchors |
| 4 | CI/CD Regression Gate | `projects/p04_ci_regression_gate` + `.github/workflows/eval-gate.yml` | >2-pt significant drop blocks (exit 1); latency hysteresis; flaky exclusion |
| 5 | RAG Adversarial Harness | `projects/p05_rag_adversarial` | confident-lie rate drops to 0 under strict grounding; retrieval and generation gated separately |
| 6 | Automated DPO Flywheel | `projects/p06_dpo_flywheel` | thumbs-down → PII-scrubbed pair → golden case → dry-run LoRA manifest → promote only if not worse |
| 7 | Statistical Significance Engine | `projects/p07_significance_engine` | headline matches `Model B wins by X% ± Y% (p<0.05, n=400)`; n=50 flagged as noise |
| 8 | Agent Red-Team Fuzzer | `projects/p08_agent_redteam_fuzzer` | 500 attacks; indirect injection succeeds with guardrails off, blocked on; repro replays |
| 9 | Production Drift Monitor | `projects/p09_drift_monitor` | no alert on stable days; alert within 2 days of injected drift; PSI known value |
| 10 | Cost-Quality Pareto Dashboard | `projects/p10_pareto_dashboard` | frontier on hand-built case; price-scale moves frontier; dependency-free HTML |
| 11 | Counterfactual Replay Debugger | `projects/p11_counterfactual_replay` | zero model calls before override; bisect finds corrupted node |
| 12 | Synthetic Edge-Case Generator | `projects/p12_synthetic_edgecases` | operators dedupe; difficulty filter keeps only cases baseline fails; coverage gaps listed |
| 13 | Context Window Eviction Tester | `projects/p13_context_eviction` | truncation loses early facts, retrieval keeps them; latest value wins over stale |
| 14 | Dataset Contamination Checker | `projects/p14_contamination_checker` | exact / partial / paraphrase leaks caught by the right tier; FPR < 2% |
| 15 | Public Eval Methodology Teardown | `projects/p15_methodology_teardowns` | three deep-dives with rubrics, judge prompts, stats, reproducible seeds |

## Part B: the 12 stages

| Stage | Learn | Practice item from the post | Where it lives |
|-------|-------|-----------------------------|----------------|
| 1 Testing foundations & stats | pytest fixtures/parametrize, JSONL, P/R/F1, CIs, kappa | pytest plugin that loads a JSONL dataset and emits a metric report | `evalkit/pytest_plugin.py` (`--eval-dataset`), `evalkit/metrics.py`, `tests/test_plugin_demo.py` |
| 2 LLM failure modes | sampling, seeds, tokenization edges, drift, hallucination taxonomies | fuzz one model with 500 adversarial inputs, classify into a taxonomy with repro steps | `p08_agent_redteam_fuzzer` (`generate_attacks(500)`, `data/taxonomy.yaml`, `repro/`) |
| 3 Golden dataset engineering | guidelines, stratified sampling, IAA, versioning, contamination | 300-case golden set, 2 annotators, kappa, versioned with changelog | `p03_calibrated_judge/data/` (500 anchors, 2 annotators, `ANNOTATION_GUIDELINES.md`, `CHANGELOG.md`, fingerprint); `p14` for contamination |
| 4 Scoring & LLM-as-judge | exact/fuzzy/embedding match, rubrics, judge biases | judge with rubric, calibrated vs 100 human labels, published agreement | `p03_calibrated_judge` (500 labels, kappa, bias report) |
| 5 RAG evaluation | hit rate, MRR, NDCG, recall@k, faithfulness vs relevance, abstention | harness with separate retrieval and generation gates + must-abstain queries | `p05_rag_adversarial`, metrics in `evalkit/metrics.py` |
| 6 Agent & trajectory eval | tool-call correctness, step grading, trajectory distance, counterfactual replay | trajectory grader vs golden path that blocks dangerous sequences | `p01_trajectory_grader`, `p11_counterfactual_replay`, `evalkit/trajectory.py` |
| 7 Frameworks & custom harnesses | DeepEval, promptfoo, RAGAS, ... | 3 frameworks behind one CLI with a unified report | `p16_unified_harness` |
| 8 Statistical rigor | bootstrap CIs, paired tests, effect size, sample size, seed variance | "Model B wins by 3.2% ± 1.1% (p<0.05)" report | `evalkit/stats.py`, `p07_significance_engine` |
| 9 CI/CD gates | Actions eval jobs, hysteresis, caching, cost budgets, flaky detection | gate that blocks on >2% drop and posts failing cases to the PR | `p04_ci_regression_gate`, `.github/workflows/eval-gate.yml` |
| 10 Production monitoring | sampling strategies, drift metrics, spike alerts, feedback-trace correlation | nightly 5% sample, offline evals, alert on decay | `p09_drift_monitor`, `p02_shadow_router` |
| 11 Data flywheels | feedback→dataset loops, DPO pairs, eval-driven optimisation | thumbs-down → labeled case → golden set → nightly run → diff report | `p06_dpo_flywheel`, `p12_synthetic_edgecases` |
| 12 Red-teaming, benchmarks, portfolio | injection/jailbreak suites, methodology, contamination-aware reporting | public teardown of agent vs naive baseline, full methodology, seeds | `p08_agent_redteam_fuzzer`, `p14_contamination_checker`, `p15_methodology_teardowns` |

## Part C: TELUS "12 mistakes building an agent platform for 100,000 employees"

Source: Cloudflare conference talk by two TELUS engineers (Oct 2026). Each mistake maps to a
working system here or to a process rule; process rules are not code and are listed as such.

| TELUS mistake | What they changed | System here | Proof |
|---|---|---|---|
| "Draw the rest of the owl": builders stuck on "upload a zip"; their AI tools produced apps the platform could not deploy | llms.txt + skill for agents, a hosted studio, watch someone try it | `projects/p24_paved_road` | validator flags an own-server app with a builder-language fix; first-run study returns `ready_to_call_easy: false` until 3 non-engineers finish |
| Sledgehammer on the fly: no metering, limits or billing until the bill arrived | Metering, limits, billing, chargeback | `projects/p18_usage_meter` | daily cap blocks the next request; 6am scheduled burst flagged; chargeback totals equal event sum |
| "Sign in with Google" only, 30 ways to sign in | One email box; passkey / IdP / magic link | `projects/p19_signin_router` | reachability audit finds every unknown-domain user; magic link single-use and expiring |
| Scalability bolted on late; thousands joining at once | Plan for 1k→10k early | `projects/p25_surge_test` | knee found near configured capacity; rate limiter turns 5xx into bounded 429s |
| The closet: analytics in the platform DB; sampled store undercounted people ~7× | "Scary line" capacity view; exact counts for people | `projects/p20_capacity_line` | days-to-cap on a known series; placement advisor flags distinct-users-from-sampled-table |
| Marketplace: ~90% of catalog untouched | Featured, teams, stars, usage; show only what you need | `projects/p21_catalog_health` | scoped view shrinks the list >90% and keeps featured + team items; dormancy queue |
| "It's fine": no observability; status page after the first outage; far-region false alarms | Status page day one, own worker, cron, right regions | `projects/p17_status_page` | single far region never opens an incident; home-region failure does; staleness self-check |
| One true god of tooling: rewrote the project repeatedly | Stop rewriting | process rule (CONTRIBUTING: one core, no re-platforming) | n/a |
| "Works on my machine": builders were in chat clients, not terminals | Ask where people work; MCP connector inside the chat tool | `evalkit/cli.py` umbrella CLI + MCP-ready adapters in `p16_unified_harness` | n/a |
| One more platform to replace all platforms | Platform that supports the others | process rule (federate; see Pizza Hut playbook B3) | n/a |
| Vibe-code bloat: 157 commits in 2 days, three ways to do one thing, 18k comment lines, ~100k lines deleted | Book the refactor before the bloat | `projects/p23_bloat_detector` | duplicate bodies found across identifier renames; `book_the_refactor` true on the bloated fixture, false on the clean one |
| Lost in the vibes: only the outcome handed to the agent | Architect first; validate | process rule (G2 design walkthrough in the Pizza Hut checklist) | n/a |
| Q&A: vibe-coded apps need secret scanning + automated review on every deploy | Scan on push, send findings back to the builder's agent | `projects/p22_deploy_gate` | planted AWS key detected and masked; placeholders not flagged; clean artifact exits 0 |

## How to tick a box yourself
```bash
.venv/bin/pytest -q                                  # all claims above, in one run
.venv/bin/pytest projects/p01_trajectory_grader -q   # one system
.venv/bin/python -m projects.p07_significance_engine.cli compare \
    --a projects/p07_significance_engine/data/model_a.jsonl \
    --b projects/p07_significance_engine/data/model_b.jsonl
```
