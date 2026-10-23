"""CLI: `build` aggregates eval results into a markdown report (stdout), metrics JSON and dashboard.html."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evalkit.io import read_jsonl
from projects.p10_pareto_dashboard.core import SYSTEM_TWO, build_metrics
from projects.p10_pareto_dashboard.dashboard import render_dashboard
from projects.p10_pareto_dashboard.report import render_markdown

HERE = Path(__file__).parent


def _build(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.results)
    if not rows:
        print("no rows in results file", file=sys.stderr)
        return 2
    metrics = build_metrics(rows, price_scale=args.price_scale, system_two=args.system_two,
                            sensitivity_scales=args.sensitivity)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_dashboard(metrics), encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(render_markdown(metrics))
    print(f"dashboard -> {out}" + (f"; metrics -> {args.json}" if args.json else ""))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p10_pareto_dashboard", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build report, metrics JSON and dashboard.html")
    b.add_argument("--results", default=str(HERE / "data" / "results.jsonl"))
    b.add_argument("--out", default="dashboard.html")
    b.add_argument("--json", default=None)
    b.add_argument("--price-scale", type=float, default=1.0, help="multiply System-Two cost by this factor")
    b.add_argument("--system-two", default=SYSTEM_TWO, help="model name whose price the scale applies to")
    b.add_argument("--sensitivity", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    b.set_defaults(fn=_build)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
