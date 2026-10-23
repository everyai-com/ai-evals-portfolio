# p11 — Counterfactual Replay Debugger

## What
Record every LLM hop and tool output of an agent into an `evalkit.trajectory.Trajectory`
(one step per hop, tagged with a `node_id` and a content hash of its inputs), then **swap one
node** and replay the rest of the trajectory:

- hops *before* the override are served from the recording (zero model/tool calls),
- the overridden node returns the new value (or runs a new tool implementation),
- hops *after* it execute live, optionally on a **different model**.

`diff` compares two runs step by step (first divergence, changed steps, final-answer
divergence) and `bisect` finds the earliest node whose override to the golden value repairs
the final answer: the step that introduced the hallucination.

## Why
When a 14-step agent gives a wrong answer, the wrong *answer* is the last symptom, not the
cause. Re-running the whole agent costs model calls and is non-deterministic. Replaying from a
recording with a single counterfactual edit turns "which hop broke it?" into a deterministic,
cheap experiment.

## Components
| piece | module | role |
|---|---|---|
| `Recorder` | `core.py` | `ExecutionContext` that wraps an `LLMClient` + tool registry and records each hop (`node_id`, inputs, `input_hash`, output, model, tokens, cost) |
| `Replayer` | `core.py` | serves cached outputs before the override point, applies `{node_id: new_output}` / `tool_overrides`, runs live after; `mode="memo"` reuses recorded outputs downstream whenever a hop's input hash is unchanged |
| `diff` | `core.py` | aligned step-by-step comparison, markdown or JSON |
| `bisect` | `core.py` | for each node in order, override to the golden output (memo mode) and check the final answer |
| demo agent | `demo.py` | 7 hops: plan → search → read → extract → compute → verify → answer, all scripted on `MockLLM` handlers that are functions of their prompt |

Memo mode is what makes bisect isolate the *source* of an error rather than the first node:
overriding `plan` with an identical value leaves `search`'s input hash unchanged, so the
corrupted `search` output is reused and the answer stays wrong; overriding `search` changes
`read`'s inputs, so everything downstream re-executes and the answer is repaired.

## How to run
```bash
python -m projects.p11_counterfactual_replay.cli record --out runs/ [--corrupt search] [--task-id golden]
python -m projects.p11_counterfactual_replay.cli replay --run data/corrupted_run.jsonl \
    --override 'search=["acme-offices-2025","acme-benefits"]' [--model mock-system-two] [--mode live|memo] [--out fixed.jsonl]
python -m projects.p11_counterfactual_replay.cli diff --a data/golden_run.jsonl --b data/corrupted_run.jsonl
python -m projects.p11_counterfactual_replay.cli bisect --run data/corrupted_run.jsonl --golden data/golden_run.jsonl
.venv/bin/pytest projects/p11_counterfactual_replay -q
```
`replay` exits 1 when the replayed final answer differs from the golden answer; `bisect` exits
1 when no single-node override repairs the run. `data/golden_run.jsonl` and
`data/corrupted_run.jsonl` are committed recordings (healthy search vs. a search that returns
the wrong company's report).

## Sample output
### `record` (healthy run)
```
## Run `golden`

| # | node | kind | source | model | input_hash | output |
|---|---|---|---|---|---|---|
| 0 | plan | llm | live | mock-system-one | c2c18664 | search for: acme corp office headcount report 2025 |
| 1 | search | tool | live | - | 41a6bd77 | ["acme-offices-2025", "acme-benefits"] |
| 2 | read | tool | live | - | 0e1e0921 | {"acme-benefits": "Acme Corp benefits handbook: remote stipend, 25 days… |
| 3 | extract | llm | live | mock-system-one | fb6bd56b | [120, 85, 60] |
| 4 | compute | tool | live | - | d429b699 | 265.0 |
| 5 | verify | llm | live | mock-system-one | 4078b0ac | VERIFIED |
| 6 | answer | llm | live | mock-system-one | 73d6f717 | The three largest Acme Corp offices have a combined headcount of 265. |

**final answer:** The three largest Acme Corp offices have a combined headcount of 265.

saved to runs/golden.jsonl
```
### `diff --a data/golden_run.jsonl --b data/corrupted_run.jsonl`
```
## Trajectory diff

| # | node (A) | node (B) | B source | status | output A | output B |
|---|---|---|---|---|---|---|
| 0 | plan | plan | live | same | search for: acme corp office headcount report 2025 | search for: acme corp office headcount report 2025 |
| 1 | search | search | live | **changed** | ["acme-offices-2025", "acme-benefits"] | ["zenith-offices-2025", "acme-history"] |
| 2 | read | read | live | **changed** | {"acme-benefits": "Acme Corp benefits handbook: remote stipend, 25 days… | {"acme-history": "Acme Corp was founded in 1999 in Austin and opened it… |
| 3 | extract | extract | live | **changed** | [120, 85, 60] | [300, 210, 40] |
| 4 | compute | compute | live | **changed** | 265.0 | 550.0 |
| 5 | verify | verify | live | same | VERIFIED | VERIFIED |
| 6 | answer | answer | live | **changed** | The three largest Acme Corp offices have a combined headcount of 265. | The three largest Acme Corp offices have a combined headcount of 550. |

- first divergence: step 1 (search)
- changed steps: [1, 2, 3, 4, 6]
- final answer A: `The three largest Acme Corp offices have a combined headcount of 265.`
- final answer B: `The three largest Acme Corp offices have a combined headcount of 550.`
- final answers DIVERGE
```
### `bisect --run data/corrupted_run.jsonl --golden data/golden_run.jsonl`
```
## Bisect: culprit = `search` (hop 1)

| # | override node -> golden | final answer | result |
|---|---|---|---|
| 0 | plan | The three largest Acme Corp offices have a combined headcount of 550. | still wrong |
| 1 | search | The three largest Acme Corp offices have a combined headcount of 265. | **fixed** |
```
### `replay --run data/corrupted_run.jsonl --override 'search=[...]'` (swap only the search node back)
```
## Run `corrupted_run+replay`

| # | node | kind | source | model | input_hash | output |
|---|---|---|---|---|---|---|
| 0 | plan | llm | cache | mock-system-one | c2c18664 | search for: acme corp office headcount report 2025 |
| 1 | search | tool | override | - | 41a6bd77 | ["acme-offices-2025", "acme-benefits"] |
| 2 | read | tool | live | - | 0e1e0921 | {"acme-benefits": "Acme Corp benefits handbook: remote stipend, 25 days… |
| 3 | extract | llm | live | mock-system-one | fb6bd56b | [120, 85, 60] |
| 4 | compute | tool | live | - | d429b699 | 265.0 |
| 5 | verify | llm | live | mock-system-one | 4078b0ac | VERIFIED |
| 6 | answer | llm | live | mock-system-one | 73d6f717 | The three largest Acme Corp offices have a combined headcount of 265. |

**final answer:** The three largest Acme Corp offices have a combined headcount of 265.

cache hits: 1, live calls: 5

## Trajectory diff

| # | node (A) | node (B) | B source | status | output A | output B |
|---|---|---|---|---|---|---|
| 0 | plan | plan | cache | same | search for: acme corp office headcount report 2025 | search for: acme corp office headcount report 2025 |
| 1 | search | search | override | **changed** | ["zenith-offices-2025", "acme-history"] | ["acme-offices-2025", "acme-benefits"] |
| 2 | read | read | live | **changed** | {"acme-history": "Acme Corp was founded in 1999 in Austin and opened it… | {"acme-benefits": "Acme Corp benefits handbook: remote stipend, 25 days… |
| 3 | extract | extract | live | **changed** | [300, 210, 40] | [120, 85, 60] |
| 4 | compute | compute | live | **changed** | 550.0 | 265.0 |
| 5 | verify | verify | live | same | VERIFIED | VERIFIED |
| 6 | answer | answer | live | **changed** | The three largest Acme Corp offices have a combined headcount of 550. | The three largest Acme Corp offices have a combined headcount of 265. |

- first divergence: step 1 (search)
- changed steps: [1, 2, 3, 4, 6]
- final answer A: `The three largest Acme Corp offices have a combined headcount of 550.`
- final answer B: `The three largest Acme Corp offices have a combined headcount of 265.`
- final answers DIVERGE

saved to runs/fixed.jsonl
```

## Design notes
- Agents are plain callables `agent(ctx) -> str`; the only contract is that all side effects go
  through `ctx.llm(node_id, prompt)` / `ctx.tool(node_id, name, **args)`. The same agent code
  runs under `Recorder` and `Replayer`, so replay fidelity does not depend on the agent.
- Node lookup during replay is by `node_id` at-or-after a cursor, so an agent that takes a
  different branch after the override (different node ids) simply runs those hops live.
- Everything is `MockLLM`-scripted and offline; `EVALKIT_LLM=anthropic` with
  `evalkit.llm.get_client()` in place of `demo.make_llm` would record a real model, and the
  replayed downstream hops can be pointed at any `LLMClient`.
