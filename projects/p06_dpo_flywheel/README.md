# p06 · Automated DPO Flywheel

**What.** A pipeline that captures user thumbs-downs, scrubs and filters them, turns each into a
DPO preference pair `{prompt, chosen, rejected}`, exports a `trl`-compatible train/val split, adds a
golden eval case per pair (so every captured failure becomes a regression test), and runs a nightly
job that trains a LoRA adapter (real `peft`+`trl` when `EVALKIT_TRAIN=1`, otherwise a validating
dry-run), evaluates it on the golden set before/after, and promotes it only if it is not worse.

**Why.** Feedback buttons are the cheapest labelled data a product has, but raw thumbs-downs are
not training data: they contain PII, spam, duplicates, and no *preferred* answer. The flywheel
makes the "preferred" side explicit with three strategies in strict priority order (user
correction > a thumbs-up on the same prompt > a judge-verified regeneration), keeps the provenance
and confidence on every pair, and closes the loop by turning each pair into a golden test that the
next adapter must still pass.

## Components

| piece | file | what it does |
|---|---|---|
| `scrub_pii` | `core.py` | regex scrubbing: emails → `<EMAIL>`, phones → `<PHONE>`, ≥7-digit runs → `<NUMBER>` |
| `FeedbackStore` | `core.py` | JSONL ingest with dedup by `event_id`, quality filter (min prompt length, missing response, spam/abuse keywords, per-user cap), watermark + seen-ids persisted in `state/` |
| `PairBuilder` | `core.py` | strategies `correction` (0.9) → `thumbs_up` on a near-identical prompt (normalised match or token-Jaccard ≥ 0.8, 0.7) → `regenerated` via `LLMClient` accepted only when `judge(candidate) > judge(rejected)` |
| `split_pairs` | `core.py` | grouped by `conversation_id`, stratified by majority tag; no conversation in both splits |
| `append_golden` | `core.py` | one golden case `{id, input, expected_behavior, rejected_behavior, tags}` per pair into `data/golden_additions.jsonl` + a `CHANGELOG.md` line (idempotent by id) |
| `train_lora` | `trainer.py` | `peft` + `trl.DPOTrainer` when `EVALKIT_TRAIN=1` and importable; else dry-run: validate rows, dataset stats, `training_manifest.json`, fake adapter dir with `adapter_config.json` |
| `NightlyJob` | `nightly.py` | ingest → build pairs (new thumbs-downs only) → validate (min pairs, max tenant share, judge pass rate) → train → post-train eval hook → promote; state in `state/` |
| scheduler | `nightly.sh`, `schedule.yaml` | cron wrapper and a GitHub Actions `schedule:` example |
| demo data | `data/make_feedback.py` → `data/feedback.jsonl` | 420 events: ~30% thumbs-down, 40 corrections, PII in ~8%, 20 duplicate ids, spam, empty responses, one heavy user |

## How to run

```bash
# from the repo root
python -m projects.p06_dpo_flywheel.cli ingest --events projects/p06_dpo_flywheel/data/feedback.jsonl --state state/
python -m projects.p06_dpo_flywheel.cli build-pairs --state state/ --out pairs/          # also appends golden cases
python -m projects.p06_dpo_flywheel.cli train --pairs pairs/train.jsonl --out adapters/ --dry-run
python -m projects.p06_dpo_flywheel.cli nightly --events projects/p06_dpo_flywheel/data/feedback.jsonl --state state/
projects/p06_dpo_flywheel/nightly.sh                      # cron entry: 30 2 * * * .../nightly.sh
EVALKIT_TRAIN=1 python -m projects.p06_dpo_flywheel.cli train --pairs pairs/train.jsonl --out adapters/   # real LoRA (needs peft/trl/GPU)
.venv/bin/pytest projects/p06_dpo_flywheel -q
```

Exit codes: `0` ok / adapter promoted, `1` nightly skipped or adapter not promoted, `2` usage or
data-validation error.

## Sample output

Real run on the committed fixtures (`data/feedback.jsonl`, MockLLM for strategy 3):

```text
$ python -m projects.p06_dpo_flywheel.cli ingest --events data/feedback.jsonl --state state/
## ingest

| received | duplicates | accepted | rejected | pii_scrubbed | watermark |
|---|---|---|---|---|---|
| 420 | 20 | 349 | {"short_prompt": 6, "spam": 14, "no_response": 8, "user_cap": 23} | {"number": 10, "phone": 15, "email": 15} | 2026-10-20T19:38:00Z |

349 events in state/events.jsonl

$ python -m projects.p06_dpo_flywheel.cli build-pairs --state state/ --out pairs/
## build-pairs

| counter | n |
|---|---|
| correction | 31 |
| downs | 104 |
| regenerated | 8 |
| thumbs_up | 65 |

split: {'all': 104, 'train': 82, 'val': 22} -> pairs/{all,train,val}.jsonl (trl DPO format)
golden additions: +104 -> projects/p06_dpo_flywheel/data/golden_additions.jsonl (changelog projects/p06_dpo_flywheel/data/CHANGELOG.md)

$ python -m projects.p06_dpo_flywheel.cli train --pairs pairs/train.jsonl --out adapters/ --dry-run
## train (dry-run)

| pairs | tokens | est GPU-h | lora r/alpha/dropout | beta | lr | epochs | status |
|---|---|---|---|---|---|---|---|
| 82 | 5815 | 0.005 | 16/32/0.05 | 0.1 | 5e-06 | 2 | dry_run_ok |

manifest: adapters/training_manifest.json

$ python -m projects.p06_dpo_flywheel.cli nightly --events data/feedback.jsonl --state state/ --date 2026-10-23; echo "exit=$?"
## Nightly DPO flywheel — 2026-10-23: **PROMOTED**

- ingest: 420 received, 20 duplicates, 349 accepted, rejected {'short_prompt': 6, 'spam': 14, 'no_response': 8, 'user_cap': 23}, PII scrubbed {'number': 10, 'phone': 15, 'email': 15}
- new pairs: 104 {'correction': 31, 'thumbs_up': 65, 'regenerated': 8}
- validation: {'min_pairs': True, 'max_tenant_share': True, 'judge_pass_rate': True} (judge pass rate 1.0, top tenant share 0.606)
- train: dry-run on 82 pairs, 5815 tokens, est. 0.0052 GPU-h -> state/adapters/2026-10-23
- post-train eval on 104 golden cases: Model B wins on golden_pass_rate by 85.6% ± 7.2% (p<0.05, n=104)

exit=0

$ python -m projects.p06_dpo_flywheel.cli nightly --events data/feedback.jsonl --state state/ --date 2026-10-24; echo "exit=$?"   # same events again
## Nightly DPO flywheel — 2026-10-24: **SKIPPED**

- ingest: 420 received, 420 duplicates, 0 accepted, rejected {}, PII scrubbed {}
- new pairs: 0 {}
- validation: {'min_pairs': False, 'max_tenant_share': True, 'judge_pass_rate': True} (judge pass rate 0.0, top tenant share 0.0)
- reason: validation failed: min_pairs

exit=1

$ cat data/CHANGELOG.md
# Golden additions changelog

- 2026-10-23: +104 golden cases from DPO pairs (correction=31, regenerated=8, thumbs_up=65); total 104

$ cat state/reports/eval_diff_2026-10-23.md
# Post-train eval diff — 2026-10-23

Model B wins on golden_pass_rate by 85.6% ± 7.2% (p<0.05, n=104)

| metric | before | after | Δ | 95% CI (pts) | p | n |
|---|---|---|---|---|---|---|
| golden pass rate | 0.0% | 85.6% | +85.6 pts | [+77.9, +92.3] | 0.0002 | 104 |

promote: yes

$ head -c 600 data/golden_additions.jsonl
{"expected_behavior": "I'm sorry order 40036 arrived cold. I have issued a full refund to your original payment method; it will show within 5-7 business days. I have also added a 20% credit to your account for the next order.", "id": "g_e47db92cb697b238", "input": "Refund please for order 40036 - everything was cold", "rejected_behavior": "Refunds are not possible.", "source": "correction", "tags": ["refund"]}
```

## Design notes

- **Pair priority.** A user correction is the strongest signal (they told us the answer), a
  thumbs-up on the same prompt is the next best (someone else got a good answer), and a regeneration
  is weakest because it is self-labelled; it is only kept when the judge prefers it to the rejected
  answer, with `confidence = min(0.6, judge margin)`. `heuristic_judge` is a cheap offline stand-in
  (length, prompt overlap, boilerplate penalty); swap in an LLM judge via the `judge` argument.
- **No leakage.** Conversations are the unit of splitting: each `conversation_id` is bucketed by its
  majority tag and ~`val_frac` of each bucket goes to val, so stratification holds per tag while no
  conversation straddles the split (tested).
- **Idempotency.** Dedup by `event_id` at ingest, `paired_event_ids.json` so a thumbs-down is paired
  once, and golden ids derived from `pair_id = hash(event_id, source)`, so a second nightly run over
  the same events ingests 0, builds 0 pairs and is skipped at the `min_pairs` check.
- **Validation gates before training:** `min_pairs` (20), `max_tenant_share` (0.7, so one tenant's
  taste cannot dominate the adapter), `min_judge_pass_rate` (0.6 of pairs where the judge prefers
  chosen over rejected).
- **GPU-hours estimate** (dry-run manifest): each pair is two sequences (chosen, rejected) through
  policy forward+backward and reference forward, ≈ `4 × pair_tokens` per epoch, so
  `hours = 4 · tokens · epochs / throughput / 3600` with a default throughput of 2500 tok/s
  (7B, LoRA, bf16, one A100-80GB). Tokens use `evalkit.llm.approx_tokens`.
- **Post-train eval.** The hook returns per-case pass vectors before/after on the golden additions;
  `evalkit.stats.paired_bootstrap` gives Δ with a 95% CI and the adapter is promoted unless it is
  worse (Δ < −tolerance and either significant or below −2 pts). The default `simulated_eval_hook`
  stands in for inference in dry-run mode: an adapter "knows" the prompts it trained on
  (`adapter_memory.jsonl`), everything else behaves like the base, which is why the before rate is
  0% (every golden case is a captured base failure). Replace it with real inference when training
  for real.
- **Reused from evalkit:** `paired_bootstrap`, `md_table`/`pct`, `read_jsonl`/`write_jsonl`/`stable_hash`,
  `MockLLM`/`get_client`, `approx_tokens`.
