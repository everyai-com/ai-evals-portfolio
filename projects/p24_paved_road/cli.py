"""CLI for the paved road.

    python -m projects.p24_paved_road.cli generate --spec data/platform.yaml --out out/
    python -m projects.p24_paved_road.cli validate --dir data/sample_app_express --spec data/platform.yaml [--json out.json]
    python -m projects.p24_paved_road.cli first-runs --log data/first_runs.jsonl [--json out.json]

Exit codes: 0 ok / deployable / ready, 1 blocking findings / not ready to call easy, 2 usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from projects.p24_paved_road.core import (CompatValidator, PlatformSpec, generate_all, load_first_runs,
                                          render_findings, render_first_runs, summarise_first_runs)

HERE = Path(__file__).parent
DATA = HERE / "data"


def cmd_generate(a) -> int:
    spec = PlatformSpec.load(a.spec)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = generate_all(spec)
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8")
    print(f"## generate ({spec.name})\n")
    for name, text in files.items():
        print(f"- {out / name} ({len(text.splitlines())} lines)")
    return 0


def cmd_validate(a) -> int:
    spec = PlatformSpec.load(a.spec)
    findings = CompatValidator(spec).validate(a.dir)
    print(render_findings(findings, spec, a.dir))
    if a.json:
        Path(a.json).write_text(json.dumps({"artifact": a.dir, "deployable": not any(f.blocking for f in findings),
                                            "findings": [f.to_dict() for f in findings]}, indent=1))
        print(f"\njson -> {a.json}")
    return 1 if any(f.blocking for f in findings) else 0


def cmd_first_runs(a) -> int:
    try:
        runs = load_first_runs(a.log)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"cannot read {a.log}: {e}", file=sys.stderr)
        return 2
    s = summarise_first_runs(runs, a.min_unaided)
    print(render_first_runs(s))
    if a.json:
        Path(a.json).write_text(json.dumps(s.to_dict(), indent=1))
        print(f"\njson -> {a.json}")
    return 0 if s.ready_to_call_easy else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p24_paved_road", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="write llms.txt, SKILL.md, reference.md and an AGENTS.md snippet")
    g.add_argument("--spec", default=str(DATA / "platform.yaml"))
    g.add_argument("--out", default="out")
    g.set_defaults(fn=cmd_generate)
    v = sub.add_parser("validate", help="deploy-time precheck of an app folder")
    v.add_argument("--dir", default=str(DATA / "sample_app_express"))
    v.add_argument("--spec", default=str(DATA / "platform.yaml"))
    v.add_argument("--json", default=None)
    v.set_defaults(fn=cmd_validate)
    f = sub.add_parser("first-runs", help="summarise a first-run usability log")
    f.add_argument("--log", default=str(DATA / "first_runs.jsonl"))
    f.add_argument("--min-unaided", type=int, default=3, help="non-engineers who must finish unaided")
    f.add_argument("--json", default=None)
    f.set_defaults(fn=cmd_first_runs)
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        a = build_parser().parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
