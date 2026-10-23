# Teardown 03: Agent vs naive baseline, a contamination-aware benchmark report

**Claim under test.** "Model B seems better" is not a result. A result is a paired delta with an
interval, a p-value, an effect size, a power statement, a seed, and proof the eval set was not in
the training data.

**Code.** `projects/p07_significance_engine/core.py`, `evalkit/stats.py`,
`projects/p14_contamination_checker/core.py`, `projects/p08_agent_redteam_fuzzer/core.py`,
`projects/p05_rag_adversarial/core.py`. Data: `projects/p07_significance_engine/data/model_a.jsonl`
(baseline, "A"), `projects/p07_significance_engine/data/model_b.jsonl` (agent, "B"), n = 400 paired ids,
`projects/p07_significance_engine/data/runs.jsonl` (3 seeds × 2 systems).

## 1. Setup

| | Baseline (A) | Agent (B) |
|---|---|---|
| System | single model call, no tools | tool-using loop with trajectory grading |
| Eval set | same 400 ids, strata tags | same |
| Seeds | 0, 1, 2 | 0, 1, 2 |
| Metric | task success ∈ {0,1} (binary) | same |

Per-example results are paired on `id`. Unpaired ids are refused, not dropped silently
(`pair_results` reports them). Pairing is the single biggest statistical lever: the variance of the
difference on the same examples is far smaller than the variance of two independent accuracies.

## 2. The headline, and exactly how it is computed

`evalkit.stats:paired_bootstrap`: resample the 400 paired differences with replacement 10,000
times (`numpy.random.default_rng(seed=0)`), take the mean of each resample, report the 2.5th and
97.5th percentiles as the 95% interval. The two-sided bootstrap p is twice the share of resampled
deltas on the far side of zero, floored at 1/n_boot. Effect size is Cohen's h for binary metrics,
Cohen's d otherwise. `projects/p07_significance_engine/core.py:executive_headline` prints it in
exactly the format the post asks for (seed 0, 10,000 resamples):

```
Model B wins by 3.5% ± 2.1% (p<0.05, n=400)
```
The generator plants a true +3% effect; the paired estimate lands at 3.5 with the interval
covering 3. `reproduce.py` re-runs it.

Cross-checks reported next to it, because one test is a vibe with a number:

- **Exact McNemar** (`evalkit.stats:mcnemar_exact`) on the discordant pairs: how many examples B
  fixed vs broke. "B fixes 31, breaks 17" is more actionable than the delta.
- **Sign-flip permutation test** (`evalkit.stats:permutation_test`), 10,000 permutations.
- **Effect band** (`effect_band`): h < 0.2 small, < 0.5 medium, else large. A significant small
  effect is still small.

## 3. Power: the 50-sample trap

The same generator at n = 50 (`data/small_a.jsonl`, `data/small_b.jsonl`), same true effect:
the engine reports "not significant" and attaches a power warning computed from
`paired_mde` (minimum detectable effect at this n) and `paired_required_n` (samples needed for
the observed delta at 80% power). For a 70% baseline and a 2-point delta,
`evalkit.stats:required_sample_size` says ≈ 7–9k per arm unpaired; paired on the same examples
it is in the hundreds. The report states the number so the reader cannot mistake "we did not
see it" for "it is not there".

## 4. Multiple comparisons and strata

When several candidates are compared to one baseline, p-values are Holm–Bonferroni corrected
(`holm_bonferroni`); the test suite shows the correction holds false positives down on 10 null
candidates. Per-strata deltas are computed with the same paired bootstrap and the report runs a
**Simpson's check**: if the overall winner differs from the winner in every stratum, that is
printed as a warning, because it usually means the strata are unbalanced between runs.

## 5. Seed variance

`seed_analysis` on `runs.jsonl`: mean ± std across seeds per system. The rule in the report: if
|Δ between systems| < within-system seed std, the comparison is labelled *inconclusive at this
seed count*, whatever the single-seed p-value said. Non-determinism is a variance term; it has to
be in the denominator.

## 6. Contamination-aware reporting

Before any number above is quoted, the eval set is scanned against the fine-tuning corpus with
project 14: 8-gram overlap and a 13-gram exact tier (`NgramIndex`), MinHash LSH for near-duplicates,
and a hashed TF-IDF cosine tier for paraphrases (`EmbeddingScanner`). Items are tiered
`exact / near_duplicate / partial / semantic_match / clean`. `clean_split` writes the
decontaminated eval set, and the report shows the headline **on the full set and on the clean
set**. If they disagree, the clean one is the result and the full one is a footnote.

## 7. Robustness, not just accuracy

Two more gates are reported with the benchmark, because an agent that is 3 points more accurate
and follows instructions found in tool outputs is not a better agent:

- **Red-team resilience** (project 8): 500 generated attacks (injection, indirect injection via
  tool output, exfiltration lures, malformed schemas, loop inducers, privilege escalation,
  multi-agent relay), failure rate per category with Wilson intervals, guardrails on vs off, and
  an ablation that names which guardrail toggle fixes which category. Every failure has a repro
  file under `repro/`.
- **Grounding** (project 5): on the RAG tasks, the generation gate is evaluated on gold context so
  retrieval failures do not leak into it, and the *confident ungrounded lie rate* (answered without
  support when it should have abstained) is reported per adversarial variant.

## 8. Reproduce

```bash
.venv/bin/python -m projects.p15_methodology_teardowns.reproduce
# or piecewise
.venv/bin/python -m projects.p07_significance_engine.cli compare \
  --a projects/p07_significance_engine/data/model_a.jsonl \
  --b projects/p07_significance_engine/data/model_b.jsonl --metric success --n-boot 10000 --seed 0
.venv/bin/python -m projects.p07_significance_engine.cli seeds --runs projects/p07_significance_engine/data/runs.jsonl
```
All seeds are 0 unless stated; the data generators are committed next to the data.

## 9. What this report refuses to say

It does not say "B is better". It says: on these 400 paired examples, after decontamination, B's
task success is higher by an interval that excludes zero, with a small effect size, stable across
3 seeds, with no regression in red-team resilience and a lower confident-lie rate. Each clause is
a number in the output and each number has a seed.
