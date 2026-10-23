# Project conventions (read before adding or editing a project)

Every project lives in `projects/pNN_snake_name/` and follows the same shape so one CLI, one
test command and one report format cover all of them.

## Layout
```
projects/pNN_name/
  __init__.py
  core.py            # the engine; pure functions / small classes, importable, no I/O side effects
  cli.py             # `def main(argv: list[str] | None = None) -> int` using argparse
  README.md          # What / Why / How to run / Sample output (paste a REAL run) / Design notes
  data/              # small committed fixtures (JSONL/YAML), enough to run the CLI end-to-end
  tests/test_*.py    # deterministic, offline, < 10 s total per project
```

## Rules
1. **Offline and deterministic.** Use `evalkit.llm.MockLLM` (scripted handlers, `seed=`) for any model
   call; use `numpy.random.default_rng(seed)` for randomness. `EVALKIT_LLM=anthropic` swaps in the
   real client via `evalkit.llm.get_client()` without code changes.
2. **Reuse `evalkit`.** Metrics (`evalkit.metrics`), bootstrap/paired tests (`evalkit.stats`),
   JSONL (`evalkit.io`), trajectories (`evalkit.trajectory`), markdown tables (`evalkit.report`).
   Do not re-implement these. If evalkit lacks something, add it there with a test.
3. **Dependencies:** stdlib + numpy + pyyaml only. Heavy optional deps (torch/peft/trl,
   sentence-transformers, anthropic) go behind `try: import` with a working fallback or dry-run.
4. **Every project has a CLI** reachable as `python -m projects.pNN_name.cli ...` and through
   `evalkit <alias>` (see `evalkit/cli.py`). Exit code 0 = pass, 1 = eval failed / gate blocked,
   2 = usage error. Reports go to stdout as markdown and optionally `--json out.json`.
5. **Tests prove the claim in the README.** If the README says "fails the run when a parameter is
   hallucinated", there is a test that constructs that case and asserts the failure.
6. **Explain the math.** Any statistic shown has its formula or reference in a docstring.

## Run
```bash
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
.venv/bin/pytest                      # everything
.venv/bin/pytest projects/p07_significance_engine -q
.venv/bin/python -m projects.p07_significance_engine.cli --help
```
