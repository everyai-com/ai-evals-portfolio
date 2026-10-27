"""CLI for the Trajectory Grading Engine.

    python -m projects.p01_trajectory_grader.cli grade \
        --trajectories data/trajectories.jsonl --golden data/golden --tools data/tools.yaml [--json out.json]

Exit 0 when every trajectory passed, 1 when any failed, 2 on usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evalkit.io import read_jsonl
from evalkit.report import md_table
from evalkit.trajectory import Trajectory
from projects.p01_trajectory_grader.core import (
    OK,
    GoldenPath,
    GradeResult,
    ToolRegistry,
    TrajectoryGrader,
)

HERE = Path(__file__).parent


def render_markdown(results: list[GradeResult]) -> str:
    rows = []
    for r in results:
        viol = sorted({v.verdict for v in r.violations})
        rows.append([r.task_id, r.golden_id, r.score, "PASS" if r.passed else "FAIL",
                     r.edit_distance, r.precision, r.recall, ", ".join(viol) or "-"])
    out = ["## Trajectory results", "",
           md_table(["trajectory", "golden", "score", "passed", "edit_dist", "precision", "recall",
                     "violations"], rows), ""]
    fail_rows = []
    for r in results:
        for v in r.verdicts:
            if v.verdict != OK:
                fail_rows.append([r.task_id, "-" if v.step is None else v.step, v.tool,
                                  v.node or "-", v.verdict, v.detail])
    out.append("## Per-step failures")
    out.append("")
    if fail_rows:
        out.append(md_table(["trajectory", "step", "tool", "node", "verdict", "detail"], fail_rows))
    else:
        out.append("_none_")
    n_pass = sum(r.passed for r in results)
    out.append("")
    out.append(f"**{n_pass}/{len(results)} trajectories passed**; "
               f"mean score {sum(r.score for r in results) / max(1, len(results)):.3f}")
    return "\n".join(out)


def cmd_grade(args: argparse.Namespace) -> int:
    registry = ToolRegistry.load(args.tools)
    goldens = GoldenPath.load_dir(args.golden)
    trajs = [Trajectory.from_dict(d) for d in read_jsonl(args.trajectories)]
    grader = TrajectoryGrader(registry, goldens)
    results = grader.grade_all(trajs)
    print(render_markdown(results))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps([r.to_dict() for r in results], indent=2))
        print(f"\njson written to {args.json}")
    return 0 if all(r.passed for r in results) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p01_trajectory_grader", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grade", help="grade a JSONL of trajectories against golden DAGs")
    g.add_argument("--trajectories", default=str(HERE / "data" / "trajectories.jsonl"))
    g.add_argument("--golden", default=str(HERE / "data" / "golden"))
    g.add_argument("--tools", default=str(HERE / "data" / "tools.yaml"))
    g.add_argument("--json", default=None, help="also write results to this JSON file")
    g.set_defaults(func=cmd_grade)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    try:
        return int(args.func(args))
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
