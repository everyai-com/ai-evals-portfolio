"""CLI for the Counterfactual Replay Debugger.

    record  [--out runs/] [--corrupt search] [--task-id golden]
    replay  --run <file> --override node=<json> [--override ...] [--model m] [--mode live|memo] [--out f]
    diff    --a <file> --b <file>
    bisect  --run <file> [--golden <file>] [--expected <answer>]

Exit 0 on success; 1 when a replay/bisect final answer does not match the golden answer;
2 on usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evalkit.report import md_table
from projects.p11_counterfactual_replay import demo
from projects.p11_counterfactual_replay.core import (
    Recorder,
    Replayer,
    bisect,
    diff,
    hops,
    load_run,
    node_id_of,
    node_output,
    save_run,
)

HERE = Path(__file__).parent
GOLDEN_RUN = HERE / "data" / "golden_run.jsonl"


def _short(v, n=72):
    s = v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)
    s = s.replace("\n", " ").replace("|", "\\|")
    return s if len(s) <= n else s[: n - 1] + "…"


def render_run(traj) -> str:
    rows = [[s.index, node_id_of(s), s.kind, s.meta.get("source", "-"), s.model or "-",
             s.meta.get("input_hash", "")[:8], _short(node_output(s))] for s in hops(traj)]
    return "\n".join([f"## Run `{traj.task_id}`", "",
                      md_table(["#", "node", "kind", "source", "model", "input_hash", "output"], rows),
                      "", f"**final answer:** {traj.final_answer}"])


def cmd_record(args) -> int:
    rec = Recorder(demo.make_llm(args.model), demo.make_tools(corrupt_search=args.corrupt == "search"),
                   task_id=args.task_id, meta={"corrupt": args.corrupt})
    traj = rec.run(demo.demo_agent)
    path = rec.save(args.out)
    print(render_run(traj))
    print(f"\nsaved to {path}")
    return 0


def _parse_overrides(items: list[str]) -> dict:
    out = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"--override expects node=<json>, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v  # bare string
    return out


def cmd_replay(args) -> int:
    recorded = load_run(args.run)
    rp = Replayer(recorded, _parse_overrides(args.override), llm=demo.make_llm(args.model),
                  tools=demo.make_tools(), mode=args.mode)
    out = rp.run(demo.demo_agent)
    print(render_run(out))
    print(f"\ncache hits: {rp.cache_hits}, live calls: {rp.live_calls}")
    print()
    print(diff(recorded, out).to_markdown())
    if args.out:
        save_run(out, args.out)
        print(f"\nsaved to {args.out}")
    if args.json:
        Path(args.json).write_text(json.dumps({"run": out.to_dict(), "diff": diff(recorded, out).to_dict()}, indent=2))
    return 0 if out.final_answer == demo.GOLDEN_ANSWER else 1


def cmd_diff(args) -> int:
    d = diff(load_run(args.a), load_run(args.b))
    print(d.to_markdown())
    if args.json:
        Path(args.json).write_text(json.dumps(d.to_dict(), indent=2))
    return 0


def cmd_bisect(args) -> int:
    bad, golden = load_run(args.run), load_run(args.golden)
    res = bisect(bad, golden, demo.demo_agent, llm=demo.make_llm(args.model), tools=demo.make_tools(),
                 expected=args.expected)
    print(res.to_markdown())
    if args.json:
        Path(args.json).write_text(json.dumps(res.to_dict(), indent=2))
    return 0 if res.culprit else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p11_counterfactual_replay", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="mock-system-one", help="mock model name for live hops")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="run the demo agent and save its trajectory")
    r.add_argument("--out", default=str(HERE / "runs"))
    r.add_argument("--task-id", default="golden")
    r.add_argument("--corrupt", choices=["none", "search"], default="none")
    r.set_defaults(func=cmd_record)

    rp = sub.add_parser("replay", help="replay a run with node overrides")
    rp.add_argument("--run", required=True)
    rp.add_argument("--override", action="append", default=[], metavar="node=<json>")
    rp.add_argument("--mode", choices=["live", "memo"], default="live")
    rp.add_argument("--out", default=None)
    rp.add_argument("--json", default=None)
    rp.set_defaults(func=cmd_replay)

    d = sub.add_parser("diff", help="diff two runs step by step")
    d.add_argument("--a", required=True)
    d.add_argument("--b", required=True)
    d.add_argument("--json", default=None)
    d.set_defaults(func=cmd_diff)

    b = sub.add_parser("bisect", help="find the earliest node whose golden override fixes the answer")
    b.add_argument("--run", required=True)
    b.add_argument("--golden", default=str(GOLDEN_RUN))
    b.add_argument("--expected", default=None)
    b.add_argument("--json", default=None)
    b.set_defaults(func=cmd_bisect)
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
