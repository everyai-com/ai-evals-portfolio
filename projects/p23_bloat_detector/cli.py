"""CLI for the bloat detector.

    python -m projects.p23_bloat_detector.cli scan --dir data/bloated_repo [--json out.json] [--plan plan.md]
    python -m projects.p23_bloat_detector.cli scan --dir evalkit
    python -m projects.p23_bloat_detector.cli velocity --dir . [--days 30]

Exit codes: 0 no refactor needed, 1 book_the_refactor is true (or velocity threshold hit), 2 usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from projects.p23_bloat_detector.budget import RefactorBudget, load_thresholds, render_plan, render_scan, render_velocity
from projects.p23_bloat_detector.core import Analyzer
from projects.p23_bloat_detector.velocity import git_velocity

HERE = Path(__file__).parent
DATA = HERE / "data"


def cmd_scan(a) -> int:
    try:
        metrics = Analyzer().scan(a.dir)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2
    velocity = git_velocity(a.dir, a.days) if a.velocity else None
    budget = RefactorBudget(load_thresholds(a.thresholds))
    plan = budget.plan(metrics, velocity)
    print(render_scan(metrics, plan, velocity))
    if a.json:
        payload = {**metrics.to_dict(), "plan": plan.to_dict(), "velocity": velocity.to_dict() if velocity else None}
        Path(a.json).write_text(json.dumps(payload, indent=1, default=str))
        print(f"\njson -> {a.json}")
    if a.plan:
        Path(a.plan).write_text(render_plan(plan) + "\n")
        print(f"plan -> {a.plan}")
    return 1 if plan.book_the_refactor else 0


def cmd_velocity(a) -> int:
    v = git_velocity(a.dir, a.days)
    if v is None:
        print(f"{a.dir} is not a git work tree; velocity needs `git log`.", file=sys.stderr)
        return 2
    print(render_velocity(v))
    thr = load_thresholds(a.thresholds).get("commits_48h_max")
    if a.json:
        Path(a.json).write_text(json.dumps(v.to_dict(), indent=1))
    return 1 if thr is not None and v.max_commits_48h >= thr else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p23_bloat_detector", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="scan a directory and print metrics + refactor plan")
    s.add_argument("--dir", default=str(DATA / "bloated_repo"))
    s.add_argument("--thresholds", default=str(DATA / "thresholds.yaml"))
    s.add_argument("--json", default=None)
    s.add_argument("--plan", default=None, help="also write the plan as markdown to this path")
    s.add_argument("--velocity", action="store_true", help="also run git velocity on --dir")
    s.add_argument("--days", type=int, default=30)
    s.set_defaults(fn=cmd_scan)
    v = sub.add_parser("velocity", help="git commit velocity (commits/day, busiest 48h, add/delete)")
    v.add_argument("--dir", default=".")
    v.add_argument("--days", type=int, default=30)
    v.add_argument("--thresholds", default=str(DATA / "thresholds.yaml"))
    v.add_argument("--json", default=None)
    v.set_defaults(fn=cmd_velocity)
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        a = build_parser().parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
