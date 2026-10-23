# p13 — Context Window Eviction Tester

## What
A harness that fills an agent's working memory with seeded noise, plants key facts at
controlled positions (early / middle / late) with later **contradicting updates**, and probes
whether a `ContextManager` still surfaces the right (and the *latest*) semantic state under a
token budget. Three reference managers are included:

| manager | strategy |
|---|---|
| `TruncateOldest` | sliding window: keep the newest messages that fit |
| `SummarizeOldest` | compress the oldest half through an `LLMClient` whenever the context exceeds the budget; the running summary is re-summarised with each new chunk. The `MockLLM` handler keeps every line containing `KEY:` in order |
| `RetrievalContext` | keep everything in a store; at render time rank messages by BM25 for the probe query, take the best that fit, fill the rest of the budget with the most recent messages, emit in chronological order |

Scores per manager × noise level (all over the planted facts):
**recall** (any value for the key found), **latest-correct** (answer equals the latest planted
value), **updates_ok** (latest-correct on facts that were contradicted), **position breakdown**
(early/mid/late, i.e. lost-in-the-middle), **compression ratio** (rendered / total tokens) and
**false retention** (share of the rendered context that is noise). Token counting is
`evalkit.llm.approx_tokens`.

Probes are answered by a deterministic `LookupAnswerer` (last `KEY: k=v` / `KEY: k changed to v`
statement in the rendered context wins), so no model is needed. `LLMAnswerer(client)` plugs in
any `LLMClient` instead (`--answerer llm`).

## Why
"Does the agent remember?" is not one question. A context manager can keep a fact but lose its
update, keep late facts but drop early ones, or keep the facts while filling the window with
logs. Separating recall, latest-value correctness, position sensitivity and false retention
makes the failure mode visible, and seeding everything makes the harness a regression test.

## How to run
```bash
python -m projects.p13_context_eviction.cli run --budget 4000 --noise 2000,8000,32000 --facts 12 --seed 0 \
    [--managers truncate,summarize,retrieval] [--answerer lookup|llm] [--json out.json]
.venv/bin/pytest projects/p13_context_eviction -q
```
`data/default_run.json` is the committed JSON of the default run; a test checks that a fresh run
reproduces it byte for byte.

## Sample output
`run --budget 4000 --noise 2000,8000,32000 --facts 12 --seed 0 --json data/default_run.json`
```
## Context eviction matrix

| manager | noise_tok | budget | recall | latest_ok | updates_ok | recall early/mid/late | latest early/mid/late | compression | false_retention | rendered_tok |
|---|---|---|---|---|---|---|---|---|---|---|
| truncate | 2000 | 4000 | 100.0% | 100.0% | 100.0% | 100%/100%/100% | 100%/100%/100% | 1.000 | 86.6% | 2269 |
| summarize | 2000 | 4000 | 100.0% | 100.0% | 100.0% | 100%/100%/100% | 100%/100%/100% | 1.000 | 86.6% | 2269 |
| retrieval | 2000 | 4000 | 100.0% | 100.0% | 100.0% | 100%/100%/100% | 100%/100%/100% | 1.000 | 86.6% | 2269 |
| truncate | 8000 | 4000 | 66.7% | 66.7% | 60.0% | 0%/100%/100% | 0%/100%/100% | 0.481 | 94.0% | 3945 |
| summarize | 8000 | 4000 | 100.0% | 100.0% | 100.0% | 100%/100%/100% | 100%/100%/100% | 0.479 | 92.5% | 3934 |
| retrieval | 8000 | 4000 | 100.0% | 100.0% | 100.0% | 100%/100%/100% | 100%/100%/100% | 0.481 | 93.7% | 3949 |
| truncate | 32000 | 4000 | 33.3% | 33.3% | 20.0% | 0%/0%/100% | 0%/0%/100% | 0.123 | 97.0% | 3970 |
| summarize | 32000 | 4000 | 100.0% | 100.0% | 100.0% | 100%/100%/100% | 100%/100%/100% | 0.076 | 91.9% | 2452 |
| retrieval | 32000 | 4000 | 100.0% | 100.0% | 100.0% | 100%/100%/100% | 100%/100%/100% | 0.119 | 96.5% | 3826 |

- `recall`: probe found *any* value for the key; `latest_ok`: the value equals the LATEST planted one;
  `updates_ok`: latest_ok restricted to facts that received a contradicting update.
- `compression` = rendered tokens / total conversation tokens; `false_retention` = share of the
  rendered context that is noise.

json written to data/default_run.json
```
`run --budget 4000 --noise 128000 --facts 12 --seed 0` (0.8 s)
```
## Context eviction matrix

| manager | noise_tok | budget | recall | latest_ok | updates_ok | recall early/mid/late | latest early/mid/late | compression | false_retention | rendered_tok |
|---|---|---|---|---|---|---|---|---|---|---|
| truncate | 128000 | 4000 | 25.0% | 25.0% | 20.0% | 0%/0%/75% | 0%/0%/75% | 0.031 | 95.9% | 3934 |
| summarize | 128000 | 4000 | 100.0% | 100.0% | 100.0% | 100%/100%/100% | 100%/100%/100% | 0.029 | 93.1% | 3709 |
| retrieval | 128000 | 4000 | 100.0% | 100.0% | 100.0% | 100%/100%/100% | 100%/100%/100% | 0.031 | 95.5% | 3937 |

- `recall`: probe found *any* value for the key; `latest_ok`: the value equals the LATEST planted one;
  `updates_ok`: latest_ok restricted to facts that received a contradicting update.
- `compression` = rendered tokens / total conversation tokens; `false_retention` = share of the
  rendered context that is noise.
```

Reading the matrix: at 2k noise everything fits, so all three managers are identical. At 8k the
sliding window has already lost every early fact (recall 0%/100%/100%) and at 32k the middle
ones too, while summarisation and retrieval keep 100% recall *and* resolve every contradicting
update to the latest value. `false_retention` stays high for all three because any budget not
needed for facts is filled with recent raw messages, which are noise by construction; the
summariser's smaller rendered context (2452 tokens at 32k) is the cheapest.

## Design notes
- Every manager sees the identical conversation for a given (noise level, seed): the noise
  stream is `NoiseGenerator(seed*7919 + level)`, facts are `FactPlanter(seed)`, interleaved by
  position fraction in `build_conversation`.
- Noise kinds: near-duplicate user/assistant status chatter, irrelevant `list_files` tool JSON,
  and multi-line log tails. The noise vocabulary avoids the fact keys so a lexical retriever is
  tested on finding facts, not on noise that happens to mention them.
- Facts are `Note for later: KEY: budget=bud_4821`; updates are
  `Correction: KEY: budget changed to bud_9132`, planted 10–30% of the stream later.
- Per-message token counts are conservative (`approx_tokens(text + "\n") + 1`) so a budget
  enforced per message is never exceeded by the rendered string.
- BM25 formula and parameters are in the `RetrievalContext` docstring.
