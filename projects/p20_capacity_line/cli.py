"""CLI for the scary line.

    analyse   --snapshots data/snapshots.jsonl --cap-gb 10 [--window 14] [--until-day 59] [--html report.html] [--json [out.json]]
              [--max-util 0.9]           exit 1 when utilisation >= --max-util (the line is scary)
    placement --questions data/questions.yaml --layout data/layout.yaml [--json [out.json]]
              exit 1 when any critical misplacement exists
    sampling  --rate 0.1 [--rates 0.5,0.2,0.1,0.05,0.01] --users 10000 --seed 0 [--skew 1.5] [--json [out.json]]

Exit codes: 0 ok, 1 gate blocked, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import core

HERE = Path(__file__).parent


def _emit(markdown: str, payload: object, json_opt: str | None) -> None:
    if json_opt == "-":
        print(json.dumps(payload, indent=2, default=str))
        return
    print(markdown)
    if json_opt:
        Path(json_opt).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\njson written to {json_opt}")


def _cmd_analyse(a: argparse.Namespace) -> int:
    try:
        cap = int(a.cap_gb * core.GB) if a.cap_gb else None
        snaps = core.load_snapshots(a.snapshots, cap)
        if a.until_day is not None:
            snaps = [s for s in snaps if s.day <= a.until_day] or snaps
        report = core.CapacityLine(window=a.window).analyse(snaps)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    md = report.to_markdown()
    if a.html:
        Path(a.html).write_text(core.html_report(report), encoding="utf-8")
        md += f"\n\nhtml written to {a.html}"
    _emit(md, report.to_dict(), a.json)
    return 1 if report.utilisation >= a.max_util else 0


def _cmd_placement(a: argparse.Namespace) -> int:
    try:
        questions = core.load_questions(a.questions)
        layout = core.load_layout(a.layout)
        mis = core.audit_layout(questions, layout)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    payload = {
        "recommendations": [core.recommend_placement(q).__dict__ for q in questions],
        "misplacements": [m.__dict__ for m in mis],
        "unserved": core.unserved_questions(questions, layout),
    }
    _emit(core.placement_markdown(questions, layout, mis), payload, a.json)
    return 1 if any(m.severity == "critical" for m in mis) else 0


def _cmd_sampling(a: argparse.Namespace) -> int:
    try:
        rates = [float(x) for x in a.rates.split(",")] if a.rates else [a.rate]
        if a.rate not in rates:
            rates.append(a.rate)
        rates = sorted(set(rates), reverse=True)
        sweep = core.sampling_sweep(rates, users=a.users, seed=a.seed, skew=a.skew, n_rep=a.reps)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    _emit(core.sampling_markdown(sweep, a.users, a.skew), sweep, a.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="evalkit capacity", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("analyse", help="utilisation, growth, days-to-cap, alerts")
    s.add_argument("--snapshots", default=str(HERE / "data" / "snapshots.jsonl"))
    s.add_argument("--cap-gb", type=float, default=None, help="override cap_bytes in the snapshots")
    s.add_argument("--window", type=int, default=14, help="days used for the growth fit")
    s.add_argument("--until-day", type=int, default=None, help="analyse the series as of this day")
    s.add_argument("--max-util", type=float, default=0.9, help="exit 1 at or above this utilisation")
    s.add_argument("--html", default=None, help="write a self-contained HTML report")
    s.add_argument("--json", nargs="?", const="-", default=None)
    s.set_defaults(fn=_cmd_analyse)
    p = sub.add_parser("placement", help="recommend a store class per question; flag misplacements")
    p.add_argument("--questions", default=str(HERE / "data" / "questions.yaml"))
    p.add_argument("--layout", default=str(HERE / "data" / "layout.yaml"))
    p.add_argument("--json", nargs="?", const="-", default=None)
    p.set_defaults(fn=_cmd_placement)
    m = sub.add_parser("sampling", help="distinct-user undercount from a sampled store")
    m.add_argument("--rate", type=float, default=0.1)
    m.add_argument("--rates", default="0.5,0.2,0.1,0.05,0.01", help="comma list; --rate is added")
    m.add_argument("--users", type=int, default=10_000)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--skew", type=float, default=1.5, help="pareto shape of events per user (lower = heavier tail)")
    m.add_argument("--reps", type=int, default=10)
    m.add_argument("--json", nargs="?", const="-", default=None)
    m.set_defaults(fn=_cmd_sampling)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        a = build_parser().parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
