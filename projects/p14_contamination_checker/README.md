# p14 — Dataset Contamination Checker

## What

A scanner that checks whether a golden eval set has leaked into a fine-tuning / pre-training
corpus, with three detection tiers that catch different kinds of leak:

| tier | detector | catches |
|---|---|---|
| `exact` | normalised text identical (blake2b of the token string) | verbatim copies, re-cased or re-punctuated |
| `near_duplicate` | hashed 8-gram overlap ≥ 0.8 | copies with small insertions / deletions |
| `semantic_match` | cosine ≥ 0.9 on a hashed char-n-gram TF-IDF embedding (or sentence-transformers) | paraphrases: synonym swaps, reordering |
| `partial` | 8-gram overlap ≥ 0.3 **or** any 13-gram hit (the GPT-3/GPT-4 report criterion) | a copied span inside otherwise original text |
| `clean` | none of the above | |

Plus a `MinHashLSH` (128 permutations, 32 bands × 4 rows, implemented here, no `datasketch`)
that finds near-duplicate candidates in sub-linear time and reports estimated Jaccard, a
`ContaminationReport` (rate with Wilson CI, by tier, by strata, evidence with the overlapping
span highlighted), `clean_split()` that writes a decontaminated eval set plus the removed rows,
and contamination-aware reporting: the metric on the full vs clean eval set.

## Why

A 95% score on a benchmark the model has seen is a 95% score on nothing. Exact-match checks
miss paraphrased leaks, embeddings alone miss short copied spans, and neither tells you how
much the leak moved your number. The scanner runs all three tiers and, given a results file,
reports the metric you should actually quote.

## How to run

```bash
D=projects/p14_contamination_checker/data
.venv/bin/python -m projects.p14_contamination_checker.cli scan --eval $D/eval.jsonl --train $D/train.jsonl [--ngram 8] [--cosine 0.9] [--json [out.json]] [--clean-out clean_eval.jsonl] [--removed-out removed.jsonl] [--results $D/results.jsonl]
.venv/bin/python -m projects.p14_contamination_checker.cli scan --backend st        # sentence-transformers if installed, else hashed
.venv/bin/python -m projects.p14_contamination_checker.cli fingerprint --dataset $D/eval.jsonl
.venv/bin/python -m projects.p14_contamination_checker.data.make_corpus --seed 0      # regenerate fixtures
.venv/bin/pytest projects/p14_contamination_checker -q
```

Rows need an `id` and a text field (`--field`, default `text`); an optional `strata` field
drives the per-strata table. `scan` exits 1 when the contamination rate exceeds `--max-rate`
(default 0, i.e. any leak blocks), so it works as a CI gate.

Library use:

```python
from projects.p14_contamination_checker import core

report = core.scan(eval_rows, train_rows, ngram=8, cosine=0.9)
report.rate, report.by_tier(), report.evidence()[:5]
core.contamination_aware_metrics(report, results_rows)  # full vs clean metric
core.clean_split(report, eval_rows, "clean_eval.jsonl", "removed.jsonl")
idx = core.NgramIndex(n=8, exact_n=13).build(train_rows)  # reusable across eval sets
lsh = core.MinHashLSH().build(train_rows)
lsh.query(text)  # [(train_id, est_jaccard)]
```

## Sample output

`scan --eval data/eval.jsonl --train data/train.jsonl --results data/results.jsonl --clean-out clean_eval.jsonl --removed-out removed.jsonl`
(demo corpus: 5000 train rows, 200 eval rows with 10 exact copies, 10 paraphrases and 10
copied 15-token spans planted; exit code 1):

```
# Contamination report: 200 eval rows vs 5000 train rows

**Contamination rate: 15.0%** (30/200, Wilson 95% CI [10.7%, 20.6%]). Tiers: exact, near_duplicate (8-gram overlap >= 0.8), semantic_match (cosine >= 0.9, hashed-char-ngram-tfidf), partial (overlap >= 0.3 or a 13-gram hit).

| tier | count | share |
|---|---|---|
| exact | 10 | 5.0% |
| near_duplicate | 0 | 0.0% |
| semantic_match | 10 | 5.0% |
| partial | 10 | 5.0% |
| clean | 170 | 85.0% |

## By strata

| strata | n | contaminated | rate |
|---|---|---|---|
| code | 50 | 8 | 16.0% |
| math | 50 | 7 | 14.0% |
| qa | 50 | 8 | 16.0% |
| summarize | 50 | 7 | 14.0% |

## Evidence (eval_id -> train_id; overlapping span in [[ ]])

- `eval-0000` -> `train-01210` (exact): identical after normalisation: [[ the violin beyond the lonely fortress steadily escorts the monastery firmly the telescope observes a subtle ledger behind the eager ... ]]
- `eval-0001` -> `train-01033` (exact): identical after normalisation: [[ the narrow tribunal ignores the crimson harbour beyond the estuary warmly the orchard calibrates a lonely compiler beside the faded ... ]]
...
- `eval-0010` -> `train-01034` (semantic_match): ... within a rapid workshop [[ every silent courier replaces the theorem beside the lonely physician quietly a ]] gentle lighthouse loosely observes ... (span 12 tokens, overlap 0.34)
- `eval-0011` -> `train-04950` (semantic_match): [[ every polished fortress repairs the archive beside the silent violin softly ]] the humble turbine reroutes ... (span 11 tokens, overlap 0.33)
- ... 18 more (see --json)

## Contamination-aware reporting

`success` on the full eval set: **0.725** (n=200); on the clean subset: **0.700** (n=170); on contaminated rows: 0.867 (n=30). Leakage inflates the headline metric by +2.5 points; report the clean number.

clean split: kept 170 rows -> clean_eval.jsonl; removed 30 -> removed.jsonl
```

`fingerprint --dataset data/eval.jsonl`:

```
| field | value |
|---|---|
| path | projects/p14_contamination_checker/data/eval.jsonl |
| rows | 200 |
| sha256 | 42596b60e427619e47baac7b47b5066dcf7cf2f890d45f869f57114ff1cf947f |
| first row hash | 47b2101e743aca63 |
```

On the demo corpus every planted exact copy is `exact`, every paraphrase is `semantic_match`
with no 13-gram hit (cosine 0.91–0.95), every copied span is `partial` via the 13-gram tier,
and 0 of 170 clean rows are flagged (max clean cosine 0.53, max 8-gram overlap 0.04).

## Design notes

- **Normalisation** lowercases, maps non-alphanumerics to spaces and collapses whitespace, so
  `"The Quick, Brown!"` and `"the quick brown"` are the same row.
- **N-gram index.** Each token n-gram is hashed with `hashlib.blake2b(digest_size=8)` to a
  64-bit int; the corpus's hashes are stored as one sorted `uint64` array with a parallel
  `int32` owner-row array (8 + 4 bytes per distinct n-gram, ~100 MB for 100k rows × 60
  tokens × two n-gram sizes), queried with `np.searchsorted`. Overlap = fraction of the eval
  row's n-grams present; the longest run of consecutive hits gives the longest common span
  (`run + n − 1` tokens) and the train row that owns it, which is what the evidence snippet
  highlights. The 13-gram tier is a second index of the same shape.
- **MinHash.** Shingles are token 3-grams hashed to 31 bits; `sig_i = min_s (a_i·h(s) + b_i)
  mod (2^31 − 1)` over 128 seeded (a, b) pairs, computed with one numpy broadcast per
  document. P(sig_i(A) = sig_i(B)) = J(A, B), so the estimate is the fraction of equal rows
  (test: within 0.1 of the true Jaccard). A smaller prime than the usual 2^61 − 1 is used on
  purpose: with 32-bit shingle hashes, `a·x` must wrap the modulus many times to behave like
  a permutation. LSH splits the signature into 32 bands of 4 rows; any shared band makes a
  candidate (P ≈ 0.5 at J = 0.42), and candidates are ranked by estimated Jaccard.
- **Embedding.** The default backend is a hashed character 3/4/5-gram TF-IDF (dim 4096,
  sublinear tf, idf from the train corpus, L2-normalised) vectorised with numpy rolling
  polynomial hashes, so 5k rows embed in well under a second and the train side is processed
  in chunks (memory bounded by `chunk`, not the corpus). It is deliberately lexical-but-fuzzy:
  synonym swaps and reordering keep most character n-grams, so paraphrases score ≥ 0.9 while
  unrelated rows score ~0.5. `--backend st` swaps in `sentence-transformers` behind a
  try/import with the hashed backend as fallback.
- **Tier precedence** is exact > near_duplicate > semantic_match > partial > clean; a row is
  reported once at its most severe tier, with every underlying score kept in `--json`.
- **Scale.** Index build is linear in corpus tokens (blake2b per n-gram; ~10–15 s per 100k
  rows), lookups are per-eval-row and independent of corpus size beyond the log factor of
  `searchsorted`; pre-built `NgramIndex` / `EmbeddingScanner` / `MinHashLSH` objects can be
  passed to `scan()` to amortise the train side across many eval sets.
- `evalkit` reuse: `read_jsonl`, `write_jsonl`, `dataset_fingerprint`, `wilson_ci`,
  `md_table`, `pct`.
