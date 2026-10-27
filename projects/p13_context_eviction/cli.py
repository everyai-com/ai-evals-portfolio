"""CLI for the Context Window Eviction Tester.

    python -m projects.p13_context_eviction.cli run --budget 4000 --noise 2000,8000,32000 --facts 12 --seed 0 \
        [--managers truncate,summarize,retrieval] [--answerer lookup|llm] [--json out.json]

Exit 0 on success, 2 on usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evalkit.llm import get_client
from evalkit.report import md_table, pct
from projects.p13_context_eviction.core import (
    LLMAnswerer,
    LookupAnswerer,
    ScenarioResult,
    default_managers,
    probe_handler,
    run_matrix,
    summary_handler,
)


def make_llm(model: str = "mock"):
    """MockLLM scripted for summarisation + probe answering; real client via EVALKIT_LLM=anthropic."""
    return get_client(model, handlers=[(r"^SUMMARIZE", summary_handler), (r"^Context:", probe_handler)])


def render_markdown(results: list[ScenarioResult]) -> str:
    rows = []
    for r in results:
        rp, lp = r.by_position("found"), r.by_position("latest_correct")
        rows.append([r.manager, r.noise_tokens, r.budget, pct(r.recall), pct(r.latest_correct),
                     pct(r.update_correct), f"{pct(rp['early'], 0)}/{pct(rp['middle'], 0)}/{pct(rp['late'], 0)}",
                     f"{pct(lp['early'], 0)}/{pct(lp['middle'], 0)}/{pct(lp['late'], 0)}",
                     f"{r.compression_ratio:.3f}", pct(r.false_retention), int(r.mean_rendered_tokens)])
    head = ["manager", "noise_tok", "budget", "recall", "latest_ok", "updates_ok",
            "recall early/mid/late", "latest early/mid/late", "compression", "false_retention", "rendered_tok"]
    out = ["## Context eviction matrix", "", md_table(head, rows), "",
           "- `recall`: probe found *any* value for the key; `latest_ok`: the value equals the LATEST planted one;",
           "  `updates_ok`: latest_ok restricted to facts that received a contradicting update.",
           "- `compression` = rendered tokens / total conversation tokens; `false_retention` = share of the",
           "  rendered context that is noise."]
    return "\n".join(out)


def cmd_run(args: argparse.Namespace) -> int:
    noise = [int(x) for x in args.noise.split(",") if x.strip()]
    llm = make_llm(args.model)
    managers = default_managers(llm)
    wanted = [m.strip() for m in args.managers.split(",") if m.strip()]
    unknown = [m for m in wanted if m not in managers]
    if unknown:
        raise ValueError(f"unknown managers {unknown}; choose from {sorted(managers)}")
    answerer = LLMAnswerer(llm) if args.answerer == "llm" else LookupAnswerer()
    results = run_matrix({m: managers[m] for m in wanted}, noise, args.budget, args.facts, args.seed, answerer)
    print(render_markdown(results))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps([r.to_dict() for r in results], indent=2))
        print(f"\njson written to {args.json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p13_context_eviction", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the manager x noise-level matrix")
    r.add_argument("--budget", type=int, default=4000)
    r.add_argument("--noise", default="2000,8000,32000", help="comma-separated approx token levels")
    r.add_argument("--facts", type=int, default=12)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--managers", default="truncate,summarize,retrieval")
    r.add_argument("--answerer", choices=["lookup", "llm"], default="lookup")
    r.add_argument("--model", default="mock")
    r.add_argument("--json", default=None)
    r.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    try:
        return int(args.func(args))
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
