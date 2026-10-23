# p16 - Unified Eval Harness (3 frameworks behind one CLI)

## What
One CLI that runs several eval frameworks over the same datasets and the same system under test
(SUT) and prints one report: a dataset x framework x unified-metric table, cross-framework agreement
(Cohen's kappa on pass/fail), the list of examples the frameworks disagree on, cost and runtime per
framework, and an evalkit pytest-plugin-compatible JSONL.

- `adapters/base.py` - the `Framework` protocol (`name`, `available()`, `run(dataset_path, config) ->
  UnifiedResult`) and `UnifiedResult(framework, metrics, per_example, cost_usd, duration_s, warnings,
  backend)`.
- `adapters/promptfoo_adapter.py` - if the `promptfoo` CLI is on PATH: generate a promptfooconfig that
  grades the SUT's pre-computed outputs through the `echo` provider, shell out, parse the JSON.
  Otherwise a native implementation of `equals`, `contains`, `icontains`, `regex` and `llm-rubric`
  (MockLLM judge); `javascript` asserts warn and are skipped.
- `adapters/deepeval_adapter.py` - `GEval` + `AnswerRelevancyMetric` if importable; fallback = rubric
  (coverage of the expected answer by the output) + lexical relevancy.
- `adapters/ragas_adapter.py` - `faithfulness` / `answer_relevancy` / `context_precision` if
  importable; fallback = the same three definitions on token overlap (`textmetrics.py`).
- `adapters/native_adapter.py` - evalkit metrics directly (token F1, exact match, faithfulness,
  pass rate with a Wilson CI).
- `data/harness.yaml` - datasets (JSONL `id, input, expected, context?, tags, asserts?`), the SUT by
  name (`sut.py`: a scripted MockLLM FAQ bot with deliberate flaws, or `echo_expected`), frameworks,
  per-framework options and the metric mapping table that normalises framework-specific names to
  `correctness`, `faithfulness`, `relevancy`, `pass_rate`.

Every fallback is labelled `backend: fallback` in the table and JSON; `list` shows what is real.

## Why
Each framework defines "pass" differently: substring match, rubric score, faithfulness-to-context,
token F1. Run only one and you learn its definition, not your model's quality. Running them side by
side on identical outputs makes the definitions visible: kappa tells you how much the scaffolding
agrees with itself, and the disagreement list is where to look.

## Metrics
- Unified metrics per (dataset, framework): `correctness`, `faithfulness`, `relevancy`, `pass_rate`
  (n/a when the framework has no mapped metric for it).
- Agreement: Cohen's kappa `(p_o - p_e) / (1 - p_e)` between two frameworks' pass/fail vectors on
  the same example ids, with Landis & Koch bands (`evalkit.metrics.cohen_kappa`, `interpret_kappa`).
- Fallback definitions: rubric / GEval-style correctness = fraction of the expected answer's content
  tokens present in the output; relevancy = fraction of the question's content tokens echoed in the
  answer; faithfulness = fraction of answer sentences with >= 50% token coverage by some context
  chunk; context precision@K = sum_k(precision@k * rel_k) / #relevant.

## How to run
```bash
.venv/bin/python -m projects.p16_unified_harness.cli list
.venv/bin/python -m projects.p16_unified_harness.cli run --config projects/p16_unified_harness/data/harness.yaml \
    [--frameworks promptfoo,deepeval,ragas,native] [--json out.json] [--out runs/p16] [--min-pass-rate 0.5]
.venv/bin/pytest projects/p16_unified_harness -q
# replay the exported rows through the evalkit pytest plugin:
#   pytest tests/test_plugin_demo.py --eval-dataset runs/p16/results.jsonl   (rows carry id/input/expected/tags)
```
Exit code 0 = ran, 1 = `--min-pass-rate` floor not met, 2 = usage error (unknown framework).

## Sample output
`list` on this machine (none of the external frameworks installed, so every adapter falls back):

```
| framework | available | backend |
|---|---|---|
| promptfoo | False | fallback |
| deepeval | False | fallback |
| ragas | False | fallback |
| native | True | native (evalkit) |
```

`run --config data/harness.yaml` (real output; two datasets, four frameworks, scripted `faq_bot` SUT):

```
# Unified eval harness report

SUT: `faq_bot`  |  frameworks: promptfoo, deepeval, ragas, native

## Dataset x framework x unified metric
| dataset | framework | backend | n | correctness | faithfulness | relevancy | pass_rate | cost_usd | runtime |
|---|---|---|---|---|---|---|---|---|---|
| qa_small | promptfoo | fallback | 8 | 0.688 | n/a | n/a | 0.625 | 0.00000 | 0.9ms |
| qa_small | deepeval | fallback | 8 | 0.688 | n/a | 0.519 | 0.625 | 0.00000 | 0.2ms |
| qa_small | ragas | fallback | 8 | n/a | 0.812 | 0.519 | 0.625 | 0.00000 | 0.4ms |
| qa_small | native | native | 8 | 0.400 | 0.812 | n/a | 0.375 | 0.00000 | 0.3ms |
| rag_small | promptfoo | fallback | 6 | 0.667 | n/a | n/a | 0.500 | 0.00000 | 0.5ms |
| rag_small | deepeval | fallback | 6 | 0.667 | n/a | 0.574 | 0.667 | 0.00000 | 0.3ms |
| rag_small | ragas | fallback | 6 | n/a | 0.833 | 0.574 | 0.833 | 0.00000 | 0.4ms |
| rag_small | native | native | 6 | 0.287 | 0.833 | n/a | 0.333 | 0.00000 | 0.4ms |

## Agreement between frameworks (Cohen's kappa on pass/fail)
| dataset | A | B | kappa | interpretation | disagree/n |
|---|---|---|---|---|---|
| qa_small | promptfoo | deepeval | 0.467 | moderate | 2/8 |
| qa_small | promptfoo | ragas | -0.067 | poor | 4/8 |
| qa_small | promptfoo | native | 0.529 | moderate | 2/8 |
| qa_small | deepeval | ragas | 0.467 | moderate | 2/8 |
| qa_small | deepeval | native | 0.059 | slight | 4/8 |
| qa_small | ragas | native | 0.059 | slight | 4/8 |
| rag_small | promptfoo | deepeval | 0.667 | substantial | 1/6 |
| rag_small | promptfoo | ragas | 0.333 | fair | 2/6 |
| rag_small | promptfoo | native | 0.000 | slight | 3/6 |
| rag_small | deepeval | ragas | 0.571 | moderate | 1/6 |
| rag_small | deepeval | native | 0.400 | fair | 2/6 |
| rag_small | ragas | native | 0.182 | slight | 3/6 |

## Disagreements (the scaffolding lies)
| dataset | id | expected | output | promptfoo | deepeval | ragas | native |
|---|---|---|---|---|---|---|---|
| qa_small | qa-02 | 8 | A spider has eight legs. | FAIL (0.00) | FAIL (0.00) | pass (1.00) | FAIL (0.00) |
| qa_small | qa-03 | blue | The sky is blue on a clear day. | pass (1.00) | pass (1.00) | pass (1.00) | FAIL (0.40) |
| qa_small | qa-04 | William Shakespeare | Hamlet was written by Shakespeare. | FAIL (0.50) | pass (0.50) | pass (1.00) | FAIL (0.40) |
| qa_small | qa-05 | 100 degrees Celsius | Water boils at 100 degrees Celsius at sea level. Adding salt | pass (1.00) | pass (1.00) | FAIL (0.50) | FAIL (0.40) |
| qa_small | qa-06 | Mars | Mars. | pass (1.00) | FAIL (1.00) | FAIL (1.00) | pass (1.00) |
| rag_small | rag-01 | 2 years | The Nimbus 3 comes with a 2 year warranty. | FAIL (1.00) | pass (1.00) | pass (1.00) | pass (0.50) |
| rag_small | rag-02 | 12 | A Nimbus mesh network supports at most twelve nodes. | FAIL (0.00) | FAIL (0.00) | pass (1.00) | FAIL (0.00) |
| rag_small | rag-03 | 4 dollars | Nimbus Cloud Plus costs 4 dollars per month. | pass (1.00) | pass (1.00) | pass (1.00) | FAIL (0.40) |
| rag_small | rag-04 | Frankfurt | European data is stored in Frankfurt, and US data in Oregon. | pass (1.00) | pass (1.00) | pass (1.00) | FAIL (0.25) |

## Warnings
| dataset | framework | warning |
|---|---|---|
| qa_small | promptfoo | qa-05: assertion type 'javascript' not supported in fallback; skipped |

14 rows written to runs/p16/results.jsonl (pytest --eval-dataset compatible)
```

## Design notes
- **Same outputs, different verdicts.** Every adapter runs the same deterministic SUT, so the only
  thing that differs between rows is the framework's definition of a pass. `qa-02` ("eight legs" vs
  expected "8") passes RAGAS (faithful and relevant) and fails the three correctness-based
  frameworks; `qa-06` ("Mars.") passes correctness checks but fails lexical relevancy; the native
  token-F1 gate fails correct full-sentence answers. Those are the rows worth reading.
- **Fallbacks are explicit, not silent.** `available()` is checked per run; if a real backend exists
  but errors (missing API key, etc.) the adapter still returns a result, marked `fallback`, with the
  error in `warnings`.
- **promptfoo real path grades, it does not generate.** The generated config uses the `echo` provider
  with `{{output}}` as the prompt, so promptfoo scores the SUT output the harness already has; the
  SUT is never re-run by an external tool.
- **Cost is per framework.** `cost_usd` = SUT calls + any judge calls that adapter made (MockLLM is
  priced at 0; set `EVALKIT_LLM=anthropic` to see real numbers).
