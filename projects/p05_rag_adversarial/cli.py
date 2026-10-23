"""CLI: python -m projects.p05_rag_adversarial.cli run --corpus ... --queries ... --thresholds ..."""
from __future__ import annotations

import argparse
import json
import sys

import yaml

from evalkit.io import read_jsonl
from evalkit.llm import get_client

from .core import ALL_VARIANTS, Query, run_harness
from .rag import Corpus, Generator, build_rag_mock_handlers
from .report import render_markdown, to_json


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p05_rag_adversarial", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run retrieval + generation gates over adversarial variants")
    r.add_argument("--corpus", required=True)
    r.add_argument("--queries", required=True)
    r.add_argument("--thresholds", required=True)
    r.add_argument("--variants", default="all", help="'all' or comma list of " + ",".join(ALL_VARIANTS))
    r.add_argument("--strict-grounding", action="store_true", help="add the strict grounding instruction")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--json", nargs="?", const="-", default=None, help="write JSON to file ('-' = stdout)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    variants = ALL_VARIANTS if args.variants == "all" else [v.strip() for v in args.variants.split(",")]
    bad = [v for v in variants if v not in ALL_VARIANTS]
    if bad:
        print(f"unknown variants: {bad}", file=sys.stderr)
        return 2
    corpus = Corpus.from_jsonl(args.corpus)
    queries = [Query.from_dict(r) for r in read_jsonl(args.queries)]
    with open(args.thresholds) as f:
        thresholds = yaml.safe_load(f)
    client = get_client(handlers=build_rag_mock_handlers())
    gen = Generator(client, strict_grounding=args.strict_grounding)
    result = run_harness(corpus, queries, thresholds, gen, variants=variants, seed=args.seed)
    if args.json == "-":
        print(json.dumps(to_json(result), indent=2))
    else:
        print(render_markdown(result))
        if args.json:
            with open(args.json, "w") as f:
                json.dump(to_json(result), f, indent=2)
            print(f"\njson written to {args.json}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
