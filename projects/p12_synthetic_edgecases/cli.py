"""CLI for the synthetic edge-case generator.

    python -m projects.p12_synthetic_edgecases.cli generate --golden data/golden_seed.jsonl --spec data/task_spec.yaml --n 200 --seed 0 --out out/
    python -m projects.p12_synthetic_edgecases.cli coverage --generated out/generated.jsonl --spec data/task_spec.yaml
    python -m projects.p12_synthetic_edgecases.cli promote --generated out/generated.jsonl --reviewed reviewed.jsonl --golden data/golden_seed.jsonl

Exit codes: 0 ok, 1 generation produced no cases / nothing promoted, 2 usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evalkit.io import read_jsonl
from evalkit.llm import get_client
from evalkit.report import md_table

from projects.p12_synthetic_edgecases.core import coverage_matrix, generate_cases, promote, render_coverage, write_outputs
from projects.p12_synthetic_edgecases.llm_gen import demo_llm_handler
from projects.p12_synthetic_edgecases.spec import TaskSpec

HERE = Path(__file__).parent
DATA = HERE / "data"


def cmd_generate(a) -> int:
    spec = TaskSpec.load(a.spec)
    golden = read_jsonl(a.golden)
    client = None if a.no_llm else get_client(handlers=[(r".", demo_llm_handler)])
    res = generate_cases(golden, spec, n=a.n, seed=a.seed, client=client, n_runs=a.runs, llm_n=a.llm_n,
                         keep_easy=a.keep_easy)
    paths = write_outputs(res, a.out, spec)
    s = res.stats
    print(f"## generate (seed {a.seed})\n")
    print(md_table(["candidates", "kept", "labelled", "review queue", "llm rejected", "by generator"],
                   [[s["candidates"], s["kept"], s["labelled"], s["review_queue"], s["llm_rejected"], json.dumps(s["by_generator"])]]))
    print("\ndropped: " + ", ".join(f"{k}={v}" for k, v in sorted(s["dropped"].items())))
    print("\n### Sample of kept cases\n")
    rows = [[c["operator"], c["input"][:60].replace("|", "\\|").replace("\n", " "),
             json.dumps(c["expected"], ensure_ascii=False)[:40] if c["expected"] is not None else "(review)",
             json.dumps(c["difficulty"]["baseline_output"], ensure_ascii=False)[:40]] for c in res.kept[:12]]
    print(md_table(["operator", "input", "expected", "baseline output"], rows))
    print(f"\nuncovered tag x category cells: {len(res.coverage.uncovered)}")
    print("\n".join(f"- {v}" for v in paths.values()))
    return 0 if res.kept else 1


def cmd_coverage(a) -> int:
    spec = TaskSpec.load(a.spec)
    cases = read_jsonl(a.generated)
    golden = read_jsonl(a.golden)
    cov = coverage_matrix(cases, spec, golden)
    review = sum(1 for c in cases if c.get("expected") is None)
    stats = {"candidates": len(cases), "kept": len(cases), "dropped": {}, "labelled": len(cases) - review, "review_queue": review}
    print(render_coverage(cov, spec, stats))
    if a.json:
        Path(a.json).write_text(json.dumps(cov.to_dict(), indent=1))
    return 0


def cmd_promote(a) -> int:
    spec = TaskSpec.load(a.spec)
    res = promote(read_jsonl(a.generated), read_jsonl(a.reviewed) if a.reviewed else [], a.golden, a.version, a.changelog,
                  spec, include_labelled=a.include_labelled)
    print("## promote\n")
    print(md_table(["added", "skipped", "version"], [[res["added"], json.dumps(res["skipped"]), res["version"]]]))
    return 0 if res["added"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p12_synthetic_edgecases", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--golden", default=str(DATA / "golden_seed.jsonl"))
    g.add_argument("--spec", default=str(DATA / "task_spec.yaml"))
    g.add_argument("--n", type=int, default=200)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="out")
    g.add_argument("--runs", type=int, default=3, help="baseline runs per candidate for the difficulty filter")
    g.add_argument("--llm-n", type=int, default=12)
    g.add_argument("--no-llm", action="store_true")
    g.add_argument("--keep-easy", action="store_true", help="keep cases the baseline always passes")
    g.set_defaults(fn=cmd_generate)
    c = sub.add_parser("coverage")
    c.add_argument("--generated", required=True)
    c.add_argument("--spec", default=str(DATA / "task_spec.yaml"))
    c.add_argument("--golden", default=str(DATA / "golden_seed.jsonl"))
    c.add_argument("--json", default=None)
    c.set_defaults(fn=cmd_coverage)
    r = sub.add_parser("promote")
    r.add_argument("--generated", required=True)
    r.add_argument("--reviewed", default=None)
    r.add_argument("--golden", default=str(DATA / "golden_seed.jsonl"))
    r.add_argument("--spec", default=str(DATA / "task_spec.yaml"))
    r.add_argument("--version", default=str(DATA / "VERSION"))
    r.add_argument("--changelog", default=str(DATA / "CHANGELOG.md"))
    r.add_argument("--include-labelled", action="store_true", help="also promote auto-labelled hard cases")
    r.set_defaults(fn=cmd_promote)
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        a = build_parser().parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
