"""CLI: run --config data/harness.yaml [--frameworks a,b] [--json out.json] [--out runs/p16]; list."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .adapters import ADAPTERS
from .core import export_jsonl, load_config, render_markdown, run_harness, to_json


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p16_unified_harness", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run every configured framework and print the unified report")
    r.add_argument("--config", required=True)
    r.add_argument("--frameworks", default=None, help="comma list; default = config's list")
    r.add_argument("--json", default=None, help="write the full JSON report here")
    r.add_argument("--out", default="runs/p16", help="directory for results.jsonl")
    r.add_argument("--min-pass-rate", type=float, default=None,
                   help="exit 1 if any framework's pass_rate is below this")
    sub.add_parser("list", help="show which backends are real vs fallback")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "list":
        print("| framework | available | backend |\n|---|---|---|")
        for name, cls in ADAPTERS.items():
            a = cls()
            avail = a.available()
            backend = "native (evalkit)" if name == "native" else ("real" if avail else "fallback")
            print(f"| {name} | {avail} | {backend} |")
        return 0
    cfg = load_config(args.config)
    names = [x.strip() for x in args.frameworks.split(",")] if args.frameworks else None
    try:
        report = run_harness(cfg, names)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(render_markdown(report, cfg))
    out_dir = Path(args.out)
    jsonl = out_dir / cfg.get("output", {}).get("jsonl", "results.jsonl")
    n = export_jsonl(report, cfg, jsonl)
    print(f"\n{n} rows written to {jsonl} (pytest --eval-dataset compatible)")
    if args.json:
        Path(args.json).write_text(json.dumps(to_json(report), indent=2, default=str))
        print(f"json written to {args.json}")
    if args.min_pass_rate is not None:
        low = [r for r in report["rows"] if (r["unified"]["pass_rate"] or 0.0) < args.min_pass_rate]
        if low:
            print(f"FAIL: {len(low)} framework/dataset rows below pass_rate {args.min_pass_rate}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
