# p23 · Vibe-code bloat detector

**The mistake it answers.** *"We all got carried away."* TELUS let AI agents write the whole
platform: two of the busiest days saw 157 commits each, and underneath were three different ways of
doing the same thing, wrappers around wrappers, and 18,000 lines of comments that "read more like a
diary than code". A senior engineer became chief refactor officer, three ways became one and
~100,000 lines were deleted. **Lesson: book the refactor before the bloat.** The test that encodes
it: `test_thresholds_book_the_refactor_on_bloated_not_on_clean` — the bloated fixture flips
`book_the_refactor` to true, the refactored twin of the same package does not.

**What.** `Analyzer.scan(repo_dir)` measures a source tree and `RefactorBudget` turns the
measurements into a verdict plus a ranked refactor plan with line estimates:

| signal | how it is measured (Python via `tokenize` + `ast`; JS/TS get the regex subset) |
|---|---|
| code / comment / docstring / blank lines | a code line holds a non-comment token outside a docstring; a comment line holds only a comment; docstrings are neither |
| comment-to-code ratio | `comment_lines / max(1, code_lines)` — docstrings excluded on purpose |
| diary comments | `long_block` (>= 8 consecutive comment lines), `narration` (`# increment i` above `i += 1`: a short comment starting with a narration verb that names an identifier on the next code line), `first_person` ("I", "we", "let's", "now we", "our" — never in docstrings) |
| duplicate function bodies | identifiers renamed to `_v0, _v1, ...` in first-appearance order, docstring dropped, `ast.dump` hashed; clusters with >= 2 members; `duplicated_lines = sum(lines) - max(lines)` |
| near-duplicate implementations | token 5-gram Jaccard >= 0.6 between functions whose token counts are within 2x (the p14 n-gram idea, local implementation); exact duplicates and parent/nested pairs excluded |
| wrapper chains | a body that is exactly `return g(<own parameters, same order>)`; bare-name targets resolve repo-wide, attribute targets (`_DB.get`) only in the same file; forwarding to an unknown callee is the real implementation. Depth = hops |
| concept redundancy | names split on snake/camel case, verbs mapped to a synonym class (`parse`, `to`, `load`, `read`, ... -> `parse`), generic affixes dropped: `parse_date`, `parseDate`, `date_parse`, `to_date` -> one `parse:date` cluster. Module-level functions only — `to_dict` on every dataclass is a protocol |
| dead code candidates | top-level functions never referenced outside their own body anywhere in the tree; `main`, dunders, `pytest_*`, `test_*`, test files, route-ish decorators and `__all__` exports excluded |
| file size outliers | >= 400 code lines, or > 3x the median when the tree has >= 4 files |
| git velocity (optional) | `git log --numstat` through an injectable runner: commits/day, max commits in any 48 h sliding window, added/deleted ratio |

The plan claims `(file, line)` pairs per action, so an action's `lines_removed` is only what no
earlier action already deletes and `projected_lines == lines_before - sum(lines_removed)` holds by
construction (tested). Actions, in claiming order: merge duplicates -> fold near-duplicates ->
collapse wrapper chains -> unify concepts -> delete diary comments -> delete dead code -> split
outliers; then ranked by lines removed.

## Components

| piece | file |
|---|---|
| `Analyzer`, line classifiers, `detect_diary`, `find_duplicate_clusters`, `find_near_duplicates`, `find_wrapper_chains`, `find_concept_clusters`, `find_dead_code`, `find_file_outliers`, `RepoMetrics` | `core.py` |
| `RefactorBudget` (thresholds -> breaches, plan), `render_scan` / `render_plan` / `render_velocity` (uses `evalkit.report.md_table`) | `budget.py` |
| `git_velocity(repo, days, runner)`, `parse_log`, `max_in_window`, `compute_velocity` | `velocity.py` |
| thresholds (`null` disables a rule) | `data/thresholds.yaml` |
| bloated fixture: 3 date parsers, an identical-body SKU cleaner, a depth-3 wrapper chain, a copy-pasted loader, a diary block, an unused shim | `data/bloated_repo/shopkit/` |
| the same package after the chief refactor officer | `data/clean_repo/cleankit/` |

## How to run

```bash
P=projects/p23_bloat_detector
python -m projects.p23_bloat_detector.cli scan --dir $P/data/bloated_repo [--json out.json] [--plan plan.md]
python -m projects.p23_bloat_detector.cli scan --dir evalkit                 # a real package
python -m projects.p23_bloat_detector.cli scan --dir . --velocity --days 30  # whole monorepo + git velocity
python -m projects.p23_bloat_detector.cli velocity --dir . [--days 30]
evalkit bloat scan --dir $P/data/clean_repo
.venv/bin/pytest projects/p23_bloat_detector -q
```

Exit codes: `0` no refactor needed, `1` `book_the_refactor: true` (or `velocity` hit the 48 h
commit threshold), `2` usage error / not a directory / not a git work tree.

## Sample output

Real run on the bloated fixture (exit 1):

```text
$ python -m projects.p23_bloat_detector.cli scan --dir projects/p23_bloat_detector/data/bloated_repo
## bloat scan: `projects/p23_bloat_detector/data/bloated_repo`

| files | code | comment | docstring | blank | comment ratio | diary lines | dup lines | dup % |
|---|---|---|---|---|---|---|---|---|
| 13 | 139 | 15 | 10 | 52 | 0.11 | 12 | 6 | 4.3% |

| duplicate clusters | near-duplicates | wrapper chains (max depth) | concept clusters (max) | dead code | file outliers |
|---|---|---|---|---|---|
| 1 | 1 | 2 (3) | 3 (4) | 1 | 0 |

### Duplicate function bodies (identifiers normalised)

| members | lines each | removable |
|---|---|---|
| shopkit/core/sku.py:normalize_sku, shopkit/utils/helpers.py:clean_sku | [6, 6] | 6 |

### Near-duplicate implementations (5-gram Jaccard)

| a | b | jaccard | lines |
|---|---|---|---|
| shopkit/io/loaders.py:load_orders | shopkit/io/loaders.py:load_customers | 0.689 | 14/14 |

### Wrapper chains

| chain | depth |
|---|---|
| shopkit/api/orders.py:get_order -> shopkit/api/orders.py:fetch_order -> shopkit/api/orders.py:_fetch_order_impl -> shopkit/api/orders.py:_db_get | 3 |
| shopkit/legacy/compat.py:to_date -> shopkit/legacy/compat.py:date_parse | 1 |

### Concept redundancy (same stem, different names)

| stem | members |
|---|---|
| parse:date | shopkit/core/dates.py:parse_date, shopkit/legacy/compat.py:date_parse, shopkit/legacy/compat.py:to_date, shopkit/utils/helpers.py:parseDate |
| fetch:order | shopkit/api/orders.py:_fetch_order_impl, shopkit/api/orders.py:fetch_order |
| normalize:sku | shopkit/core/sku.py:normalize_sku, shopkit/utils/helpers.py:clean_sku |

### Dead code candidates (top-level, never referenced)

| function | lines |
|---|---|
| shopkit/legacy/compat.py:convert_old_format | 9 |

### Diary comments (11 findings, 12 lines)

| file:line | kind | text |
|---|---|---|
| shopkit/utils/helpers.py:1-9 | long_block | helpers.py |
| shopkit/utils/helpers.py:2 | first_person | ok so this file started as a scratchpad while I was figuring out the order flow. |
| shopkit/utils/helpers.py:3 | first_person | we needed a date parser that was a bit more lenient than the one in core, so I |
| shopkit/utils/helpers.py:4 | first_person | wrote parseDate here. let's keep it for now, we can clean it up later. |
| shopkit/utils/helpers.py:5 | first_person | I also copied the sku cleaner here because importing from core felt circular. |
| shopkit/utils/helpers.py:6 | first_person | now we have two of them but they do the same thing, which is fine for the demo. |
| shopkit/utils/helpers.py:7 | first_person | TODO: ask someone whether we still need this file at all. |
| shopkit/utils/helpers.py:8 | first_person | (we probably do, the api module imports it) |
| shopkit/utils/helpers.py:22 | narration | check the iso match |
| shopkit/utils/helpers.py:25 | narration | return the date |
| shopkit/utils/helpers.py:27 | first_person | now we fall back to the slash format, I think this is right |

## refactor plan

**book_the_refactor: true** — duplicate_pct=0.0432 (duplicate_pct > 0.03); max_concept_cluster=4 (max_concept_cluster >= 3); max_wrapper_depth=3 (max_wrapper_depth > 2)

| # | action | lines removed |
|---|---|---|
| 1 | merge 4 implementations of `parse:date` (`parse_date`, `date_parse`, `to_date`, `parseDate`) -> keep `shopkit/core/dates.py:parse_date` | 21 |
| 2 | fold `load_customers` into `load_orders` (5-gram Jaccard 0.69); parameterise the difference | 14 |
| 3 | delete unused `convert_old_format` in shopkit/legacy/compat.py (never referenced) | 9 |
| 4 | delete 9 lines of diary comments in shopkit/utils/helpers.py (first_person, long_block, narration) | 9 |
| 5 | collapse wrapper chain `get_order` -> `fetch_order` -> `_fetch_order_impl` -> `_db_get` (depth 3); call the target directly | 7 |
| 6 | merge 2 identical implementations (`normalize_sku`, `clean_sku`) -> keep `shopkit/core/sku.py:normalize_sku` | 6 |
| 7 | collapse wrapper chain `to_date` -> `date_parse` (depth 1); call the target directly | 2 |

source lines before: 164 -> projected after refactor: 96 (-68, 41.5%)
```

The refactored twin (exit 0):

```text
$ python -m projects.p23_bloat_detector.cli scan --dir projects/p23_bloat_detector/data/clean_repo | tail -6
## refactor plan

**book_the_refactor: false** — every threshold respected

no actions: nothing to merge, collapse or delete.

source lines before: 54 -> projected after refactor: 54 (-0, 0.0%)
```

### A real example: this monorepo's `evalkit/`

Reported as found, warts included (exit 0):

```text
$ python -m projects.p23_bloat_detector.cli scan --dir evalkit
## bloat scan: `evalkit`

| files | code | comment | docstring | blank | comment ratio | diary lines | dup lines | dup % |
|---|---|---|---|---|---|---|---|---|
| 9 | 608 | 9 | 68 | 140 | 0.01 | 0 | 0 | 0.0% |

| duplicate clusters | near-duplicates | wrapper chains (max depth) | concept clusters (max) | dead code | file outliers |
|---|---|---|---|---|---|
| 0 | 1 | 0 (0) | 0 (0) | 18 | 0 |

### Near-duplicate implementations (5-gram Jaccard)

| a | b | jaccard | lines |
|---|---|---|---|
| io.py:write_jsonl | io.py:append_jsonl | 0.762 | 10/10 |

### Dead code candidates (top-level, never referenced)

| function | lines |
|---|---|
| stats.py:paired_bootstrap | 34 |
| metrics.py:fleiss_kappa | 19 |
| metrics.py:cohen_kappa | 14 |
| metrics.py:interpret_kappa | 13 |
| stats.py:permutation_test | 12 |
| metrics.py:bootstrap_ci | 11 |
| stats.py:mcnemar_exact | 11 |
| io.py:append_jsonl | 10 |
| metrics.py:mrr | 10 |
| io.py:dataset_fingerprint | 9 |
| metrics.py:ndcg_at_k | 9 |
| report.py:md_table | 9 |

## refactor plan

**book_the_refactor: false** — every threshold respected

| # | action | lines removed |
|---|---|---|
| 1 | delete unused `paired_bootstrap` in stats.py (never referenced) | 34 |
| 2 | delete unused `fleiss_kappa` in metrics.py (never referenced) | 19 |
| 3 | delete unused `cohen_kappa` in metrics.py (never referenced) | 14 |
| 4 | delete unused `interpret_kappa` in metrics.py (never referenced) | 13 |
| 5 | delete unused `permutation_test` in stats.py (never referenced) | 12 |
| 6 | delete unused `bootstrap_ci` in metrics.py (never referenced) | 11 |
| 7 | delete unused `mcnemar_exact` in stats.py (never referenced) | 11 |
| 8 | delete unused `mrr` in metrics.py (never referenced) | 10 |
| 9 | fold `append_jsonl` into `write_jsonl` (5-gram Jaccard 0.76); parameterise the difference | 10 |
| 10 | delete unused `dataset_fingerprint` in io.py (never referenced) | 9 |
| 11 | delete unused `md_table` in report.py (never referenced) | 9 |
| 12 | delete unused `ndcg_at_k` in metrics.py (never referenced) | 9 |
| 13 | delete unused `required_sample_size` in stats.py (never referenced) | 9 |
| 14 | delete unused `precision_recall_f1` in metrics.py (never referenced) | 8 |
| 15 | delete unused `recall_at_k` in metrics.py (never referenced) | 7 |
| 16 | delete unused `hit_rate` in metrics.py (never referenced) | 6 |
| 17 | delete unused `iter_jsonl` in io.py (never referenced) | 6 |
| 18 | delete unused `seed_variance` in stats.py (never referenced) | 5 |

source lines before: 685 -> projected after refactor: 483 (-202, 29.5%)
```

Reading it honestly:

* **No diary, 0.01 comment ratio, no duplicate clusters, no wrapper chains** — the library is small
  and hand-kept.
* **`write_jsonl` / `append_jsonl` at Jaccard 0.76** is a real finding: the two differ only in the
  open mode. A `mode="w"|"a"` parameter would fold them.
* **The 18 "dead" functions are a scoping artefact**, which is why `dead_code_max` is `null` by
  default: `evalkit` is a library and its public API is called from `projects/`. Scanning the whole
  monorepo (`scan --dir .`, 203 files, ~4 s) leaves exactly one evalkit candidate, `io.iter_jsonl`.
  The whole-repo scan also finds what a library scan cannot — 10 byte-identical `main()` functions
  across `projects/*/cli.py` and four identical `_emit` helpers, which is the strongest case in this
  repo for a shared `evalkit.cli` runner:

```text
$ python -m projects.p23_bloat_detector.cli scan --dir . | head -10
## bloat scan: `.`

| files | code | comment | docstring | blank | comment ratio | diary lines | dup lines | dup % |
|---|---|---|---|---|---|---|---|---|
| 203 | 17988 | 340 | 1410 | 3726 | 0.02 | 15 | 155 | 0.9% |

| duplicate clusters | near-duplicates | wrapper chains (max depth) | concept clusters (max) | dead code | file outliers |
|---|---|---|---|---|---|
| 16 | 38 | 2 (3) | 9 (6) | 6 | 28 |
```

### Velocity

The monorepo was imported in one commit, so the numbers are flat (a fake `git log` with 2 x 157
commits in 48 h is covered by `test_synthetic_157_commit_days`):

```text
$ python -m projects.p23_bloat_detector.cli velocity --dir . --days 30
### git velocity

| days | commits | max in 48h | busiest window start | added | deleted | add/delete |
|---|---|---|---|---|---|---|
| 30 | 1 | 1 | 2026-10-23 20:06 | 44627 | 0 | 44627.000 |

busiest days: 2026-10-23 (1)
```

## Tests (19, ~1 s)

| claim | test |
|---|---|
| duplicate bodies found regardless of identifier names (and a one-operator change is *not* a duplicate) | `test_duplicate_bodies_found_regardless_of_identifier_names` |
| near-duplicate loader pair found at 0.6 <= J < 1; exact duplicates are not double-reported | `test_near_duplicate_found_but_not_exact_duplicate` |
| depth-3 wrapper chain; swapped or dropped arguments are not wrappers | `test_wrapper_chain_depth` |
| `_DB.get` does not link to another file's `Cache.get`; unknown callee ends the chain | `test_attribute_call_wrapper_does_not_cross_files_and_unknown_target_ends_chain` |
| `parse_date` / `parseDate` / `date_parse` / `to_date` form one cluster; `format_date` does not join | `test_concept_cluster_for_date_parsers` |
| long block, narration, first person flagged; docstrings, `I/O`, contract sentences not flagged | `test_diary_detection_hand_built_cases_and_docstrings_not_flagged`, `test_normal_docstrings_only_file_has_no_diary` |
| unused helper found; `main`, routes, dunders, tests, recursion-only excluded | `test_dead_code_candidate_and_exclusions` |
| nested function is not a near-duplicate of its parent; dense 2-line twins still cluster | `test_nested_function_is_not_a_near_duplicate_of_its_parent_and_tiny_exact_dups_cluster` |
| `book_the_refactor` true on bloated, false on clean; each rule can fire alone; `null` never fires | `test_thresholds_book_the_refactor_on_bloated_not_on_clean`, `test_each_rule_can_fire_alone` |
| line estimates sum to the projection; shared members are claimed once; keep = `core.parse_date` | `test_refactor_plan_line_estimates_sum_correctly_and_never_double_count` |
| fake git runner: parse, 48 h window, add/delete ratio, binary files, non-repo -> None | `test_fake_git_runner_velocity_numbers`, `test_synthetic_157_commit_days` |
| CLI exit codes, `--json`, `--plan` | `test_cli_scan_exit_codes_and_outputs` |

## Design notes

* **Why the comment ratio ignores docstrings.** The diary problem is prose *between* statements.
  A documented public API has a high docstring count and that is not bloat; counting it would make
  the clean fixture look worse than the bloated one.
* **Why narration needs a verb and a shared identifier.** `# Raises FileNotFoundError when the path
  is missing` starts with a verb and mentions `path`, but it is a contract; a six-word cap keeps
  narration to `# increment i`-sized comments.
* **Why duplicate detection normalises names but not attributes.** `.strip()` vs `.lower()` is a
  behavioural difference; `rows` vs `items` is not.
* **Why dead code is informational by default.** Reference analysis is scoped to the scanned tree;
  a library's API always looks dead from inside. Scan the consumer too, or set `dead_code_max`.
* **What the plan does not do.** It estimates lines, not effort. Collapsing a wrapper chain also
  means editing every caller; the plan lists the files involved so a human can price that.
* **evalkit gap noticed while building this.** Ten projects carry a byte-identical `main()` and four
  an identical `_emit`; a shared `evalkit.cli.run(parser, argv)` would remove ~80 lines, which is
  this tool's own finding on the monorepo.
