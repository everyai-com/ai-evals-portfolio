# p12 · Synthetic Edge-Case Generator

**What.** A pipeline that grows a golden dataset with out-of-distribution inputs and boundary
conditions from three sources: deterministic, seeded **mutation operators** (no model needed),
**metamorphic relations** (follow-up inputs whose label is derived from the parent's, so no human is
needed), and a **frontier-model generator** prompted with the task spec, examples, the operator list
and the failure taxonomy. Candidates are de-duplicated (normalised text + character 5-gram Jaccard),
schema-validated against the task spec, filtered for difficulty (run through a baseline SUT N times;
keep failures and disagreements), and accounted for in an operator × tag × taxonomy coverage matrix.
Unlabelled cases go to a review queue; `promote` merges reviewed cases into the golden set with a
semver bump and a CHANGELOG entry.

**Why.** Golden sets drift towards the inputs the team happened to think of. Systematic mutation
gives every taxonomy category a shot at every input family, the difficulty filter keeps the
generated set honest ("the baseline already passes this" is not a test), and the coverage report
says exactly which (tag × category) cells are still empty.

Demo task: *parse a shipping request into `{qty, sku, date}`*. `sut.py` is a tiny rule-based
baseline with deliberate blind spots (ASCII-only digits, no two-digit years, no quantity words, no
range or multi-intent checks) so the filter has something to find.

## Components

| piece | file | what it does |
|---|---|---|
| `TaskSpec`, `validate_case` | `spec.py` | loads `data/task_spec.yaml` (input schema, output schema with types/ranges/patterns, error reasons, invariants, failure taxonomy, operator→category map); validates candidate records |
| 14 mutation operators | `operators.py` | boundary, type_confusion, unicode_digits, unicode_encoding (zero-width, RTL, emoji, homoglyphs, combining marks), length_extremes, format_drift (date formats, locale, case, whitespace), two_digit_year, quantity_words, negation_scope, injection, numeric_precision, temporal_edges, multi_intent, contradiction; each yields `Mutation(mutated_input, expected_or_None, rationale, operator_name, parent_id)` |
| `MetamorphicRelations` | `metamorphic.py` | paraphrase / add_irrelevant / reorder_clauses → same answer; scale_qty → qty doubled; shift_date → date + 1 day |
| `LLMGenerator` | `llm_gen.py` | prompts the model for N JSON cases `{input, expected, why_hard, category}`, rejects malformed rows with a reason; `demo_llm_handler` is the MockLLM batch (10 good + 3 malformed rows) |
| `Deduper` | `core.py` | exact normalised match (global) + 5-gram Jaccard ≥ 0.85 vs the parent golden case and accepted siblings |
| `DifficultyEstimator` | `core.py` | N baseline runs; keep if any failure or disagreement; unlabelled cases are kept for review |
| `coverage_matrix` | `core.py` | counts per operator × tag × category; `uncovered` = (tag, category) cells with zero cases |
| `generate_cases`, `write_outputs` | `core.py` | the pipeline; writes `generated.jsonl` (with provenance: operator, parent_id, generator, seed, category, rationale, difficulty), `review_queue.jsonl`, `dropped.jsonl`, `coverage.md`, `stats.json` |
| `promote` | `core.py` | merges approved reviewed cases (and optionally all auto-labelled hard cases) into the golden set, bumps `data/VERSION` (minor), appends to `data/CHANGELOG.md` |
| baseline SUT | `sut.py` | regex parser with known blind spots |
| demo data | `data/make_golden.py` → `data/golden_seed.jsonl` | 40 in-distribution cases over 5 phrasing templates and 3 date styles |

## How to run

```bash
# from the repo root (flags default to the committed fixtures)
P=projects/p12_synthetic_edgecases
python -m projects.p12_synthetic_edgecases.cli generate --golden $P/data/golden_seed.jsonl --spec $P/data/task_spec.yaml --n 200 --seed 0 --out out/
python -m projects.p12_synthetic_edgecases.cli coverage --generated out/generated.jsonl --spec $P/data/task_spec.yaml
python -m projects.p12_synthetic_edgecases.cli promote --generated out/generated.jsonl --reviewed reviewed.jsonl --golden $P/data/golden_seed.jsonl
#   --include-labelled also promotes the auto-labelled hard cases; --keep-easy (generate) keeps cases the baseline passes
EVALKIT_LLM=anthropic python -m projects.p12_synthetic_edgecases.cli generate ...   # real frontier model for the LLM slice
.venv/bin/pytest projects/p12_synthetic_edgecases -q
```

Exit codes: `0` ok, `1` nothing generated / nothing promoted, `2` usage error.

## Sample output

Real run on the committed fixtures (40 seed cases, MockLLM for the model slice):

```text
$ python -m projects.p12_synthetic_edgecases.cli generate --golden data/golden_seed.jsonl --spec data/task_spec.yaml --n 200 --seed 0 --out out/
## generate (seed 0)

| candidates | kept | labelled | review queue | llm rejected | by generator |
|---|---|---|---|---|---|
| 2421 | 200 | 148 | 52 | 3 | {"rule": 175, "llm": 10, "metamorphic": 15} |

dropped: baseline_always_passes=965, exact_duplicate=159, near_duplicate_of_golden=147, near_duplicate_of_sibling=41

### Sample of kept cases

| operator | input | expected | baseline output |
|---|---|---|---|
| boundary | Please send 0 x SKU-0907 by 2026-08-16 | {"error": "qty_out_of_range"} | {"qty": 0, "sku": "SKU-0907", "date": "2 |
| contradiction | Ship 5 units of SKU-8158 on 2026-09-02. To be clear: 9 units | (review) | {"qty": 5, "sku": "SKU-8158", "date": "2 |
| format_drift | Please send 500 x SKU-8764 by 29.01.2026 | {"date": "2026-01-29", "qty": 500, "sku" | {"error": "parse_failed"} |
| llm | Ship a dozen of SKU-1234 on 2026-03-14 | {"qty": 12, "sku": "SKU-1234", "date": " | {"error": "parse_failed"} |
| multi_intent | Need 5 pcs SKU-6153 delivered 2026-10-06 and also 5 units of | {"error": "multi_intent"} | {"qty": 5, "sku": "SKU-6153", "date": "2 |
| negation_scope | Order: 24 boxes of SKU-6066, ship date 12/21/2026 — actually | (review) | {"qty": 24, "sku": "SKU-6066", "date": " |
| numeric_precision | Ship 1e308 units of SKU-5715 on 2026-04-03 | {"error": "qty_out_of_range"} | {"error": "parse_failed"} |
| paraphrase | Could you ship SKU-8158 (qty 5) on 2026-09-02? | {"date": "2026-09-02", "qty": 5, "sku":  | {"error": "parse_failed"} |
| quantity_words | Ship twelve hundred units of SKU-6369 on 2026-07-06 | {"date": "2026-07-06", "qty": 1200, "sku | {"error": "parse_failed"} |
| temporal_edges | Order: 2 boxes of SKU-8631, ship date 10000-01-01 | {"error": "year_out_of_range"} | {"error": "parse_failed"} |
| two_digit_year | Ship 12 units of SKU-7870 on 04/26/26 | {"date": "2026-04-26", "qty": 12, "sku": | {"error": "parse_failed"} |
| type_confusion | SKU-3651: 1.0 units, dispatch on 2026-08-21 | {"date": "2026-08-21", "qty": 1, "sku":  | {"qty": 0, "sku": "SKU-3651", "date": "2 |

uncovered tag x category cells: 20
- out/generated.jsonl
- out/review_queue.jsonl
- out/dropped.jsonl
- out/coverage.md
- out/stats.json

$ python -m projects.p12_synthetic_edgecases.cli coverage --generated out/generated.jsonl --spec data/task_spec.yaml
# Coverage report

- candidates: 200 -> kept 200 (dropped: {})
- labelled 148, review queue 52

## By operator

| operator | cases |
|---|---|
| boundary | 15 |
| contradiction | 15 |
| format_drift | 15 |
| llm | 10 |
| multi_intent | 15 |
| negation_scope | 15 |
| numeric_precision | 15 |
| paraphrase | 15 |
| quantity_words | 15 |
| temporal_edges | 14 |
| two_digit_year | 14 |
| type_confusion | 14 |
| unicode_digits | 14 |
| unicode_encoding | 14 |

## By taxonomy category

| category | cases |
|---|---|
| unicode_digits | 15 |
| two_digit_year | 14 |
| quantity_words | 17 |
| date_format | 18 |
| boundary | 15 |
| type_confusion | 14 |
| unicode_text | 14 |
| length | 0 |
| injection | 0 |
| numeric_precision | 16 |
| temporal_edge | 15 |
| negation | 15 |
| multi_intent | 15 |
| contradiction | 16 |
| paraphrase | 16 |

## Tag x category

| tag | unicode_digits | two_digit_year | quantity_words | date_format | boundary | type_confusion | unicode_text | length | injection | numeric_precision | temporal_edge | negation | multi_intent | contradiction | paraphrase |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iso_date | 8 | 8 | 8 | 8 | 10 | 12 | 8 | 0 | 0 | 7 | 6 | 10 | 11 | 7 | 9 |
| long_date | 4 | 6 | 6 | 5 | 3 | 2 | 3 | 0 | 0 | 5 | 4 | 2 | 2 | 5 | 4 |
| need_pcs | 2 | 4 | 3 | 2 | 1 | 4 | 5 | 0 | 0 | 3 | 1 | 2 | 6 | 1 | 1 |
| order_boxes | 3 | 1 | 3 | 4 | 3 | 0 | 3 | 0 | 0 | 2 | 5 | 3 | 1 | 4 | 5 |
| send_x | 2 | 2 | 3 | 3 | 5 | 2 | 3 | 0 | 0 | 2 | 1 | 5 | 3 | 4 | 4 |
| ship_units | 3 | 4 | 2 | 3 | 2 | 4 | 3 | 0 | 0 | 3 | 4 | 2 | 2 | 2 | 4 |
| sku_first | 4 | 3 | 4 | 3 | 4 | 4 | 0 | 0 | 0 | 5 | 3 | 3 | 3 | 4 | 1 |
| us_date | 2 | 0 | 1 | 2 | 2 | 0 | 3 | 0 | 0 | 3 | 4 | 3 | 2 | 3 | 2 |

## Uncovered cells (20)

- iso_date x length
- iso_date x injection
- long_date x length
- long_date x injection
- need_pcs x length
- need_pcs x injection
- order_boxes x type_confusion
- order_boxes x length
- order_boxes x injection
- send_x x length
- send_x x injection
- ship_units x length
- ship_units x injection
- sku_first x unicode_text
- sku_first x length
- sku_first x injection
- us_date x two_digit_year
- us_date x type_confusion
- us_date x length
- us_date x injection

## Categories with no cases at all

- length
- injection


$ head -2 out/review_queue.jsonl
{"approved": null, "baseline_output": {"date": "2026-09-02", "qty": 5, "sku": "SKU-8158"}, "expected": null, "id": "syn_e289bee0bc45", "input": "Ship 5 units of SKU-8158 on 2026-09-02. To be clear: 9 units, not 5.", "operator": "contradiction", "rationale": "quantity contradicted later in the text"}
{"approved": null, "baseline_output": {"date": "2026-12-21", "qty": 24, "sku": "SKU-6066"}, "expected": null, "id": "syn_97d230cf92b2", "input": "Order: 24 boxes of SKU-6066, ship date 12/21/2026 — actually, cancel that.", "operator": "negation_scope", "rationale": "request then cancelled (scope flip)"}

$ # a reviewer labels two queue items (reviewed.jsonl), then promote
$ cat reviewed.jsonl
{"approved": true, "expected": {"date": "2026-09-02", "qty": 9, "sku": "SKU-8158"}, "id": "syn_e289bee0bc45"}
{"approved": true, "expected": {"error": "no_shipment"}, "id": "syn_97d230cf92b2"}
$ python -m projects.p12_synthetic_edgecases.cli promote --generated out/generated.jsonl --reviewed reviewed.jsonl --golden golden_copy.jsonl --version VERSION --changelog CHANGELOG.md
## promote

| added | skipped | version |
|---|---|---|
| 2 | {} | 0.2.0 |

$ cat VERSION CHANGELOG.md
0.2.0
# Golden set changelog

- 0.1.0: initial seed of 40 in-distribution cases (data/make_golden.py)
- 0.2.0 (2026-10-24): +2 synthetic cases promoted (contradiction=1, negation_scope=1); total 42
```

## Design notes

- **Labels follow the invariants.** A mutation that preserves meaning (unicode digits, another date
  format, injection text, padding) keeps the parent's expected; one that violates an invariant gets
  `{"error": reason}` with `reason` drawn from the spec's `error_reasons`; genuinely ambiguous ones
  (negation, contradiction, `null` in place of a number, homoglyph SKUs) get `None` and land in the
  review queue. `promote` re-validates reviewer labels against the spec before merging.
- **Determinism.** Each operator gets `numpy.random.default_rng(hash(seed, operator, parent_id))`,
  so adding an operator or a parent never reshuffles the others; the LLM slice is deterministic under
  MockLLM; selection is a seeded round-robin over operators so every operator keeps a share when
  `--n` is smaller than the pool. Same seed ⇒ byte-identical `generated.jsonl` (tested).
- **Difficulty = evidence, not opinion.** With a deterministic baseline, "hard" means the baseline
  fails; with a stochastic SUT the N-run disagreement check also catches instability. The baseline
  parses injection strings and padded inputs fine, so the `injection` and `length` categories end up
  with zero *hard* cases and the coverage report says so. That is the honest reading: those cells are
  covered by candidates but not by failures. Use `--keep-easy` to carry them as regression guards.
- **Dedup cost.** The 5-gram Jaccard check is scoped to the parent golden case and its accepted
  siblings (cross-parent near-duplicates differ in qty/sku/date and are caught by the exact check),
  with a set-size pre-filter, so 2.4k candidates dedup in well under a second.
- **Metamorphic labels are derived, not guessed.** `relation(expected_parent)` produces the
  follow-up label, and `relation_holds` is asserted for every pair in the tests.
- **Reused from evalkit:** `MockLLM`/`get_client`/`LLMResponse.json`, `read_jsonl`/`write_jsonl`/`stable_hash`, `md_table`.
