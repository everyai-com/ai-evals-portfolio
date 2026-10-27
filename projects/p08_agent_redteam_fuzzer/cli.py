"""CLI: fuzz --n 500 --seed 0 --guardrails on|off|ablate --out runs/ [--json]; repro --file; taxonomy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .fuzz import render_markdown, replay, run_matrix, to_json, write_repros
from .oracles import load_taxonomy


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p08_agent_redteam_fuzzer", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fuzz", help="run the fuzz matrix")
    f.add_argument("--n", type=int, default=500)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--guardrails", choices=["on", "off", "ablate"], default="ablate")
    f.add_argument("--out", default="runs")
    f.add_argument("--max-steps", type=int, default=8)
    f.add_argument("--repro-limit", type=int, default=50, help="max repro files to write (0 = all)")
    f.add_argument("--json", nargs="?", const="-", default=None)
    r = sub.add_parser("repro", help="replay a repro record")
    r.add_argument("--file", required=True)
    sub.add_parser("taxonomy", help="print the failure taxonomy")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "taxonomy":
        tax = load_taxonomy()
        print(yaml.safe_dump(tax, sort_keys=False))
        return 0
    if args.cmd == "repro":
        rec = json.loads(Path(args.file).read_text())
        r = replay(rec)
        f = r["failure"]
        print(f"attack {rec['attack_id']}  oracle={rec['oracle']}  step={rec['failing_step_index']}  "
              f"severity={rec['severity']}")
        print(f"replayed: failure={'none' if f is None else f.oracle}  step={None if f is None else f.failing_step_index}")
        print(f"reproduced: {r['reproduced']}")
        if r["result"] is not None:
            for s in r["result"].trajectory.steps:
                tc = s.tool_call
                print(f"  [{s.index}] {s.kind:5} " + (f"{tc.name} {json.dumps(tc.args, default=str)[:80]}"
                                                   + (f" -> ERROR {tc.error}" if tc.error else "") if tc else s.content[:80]))
        return 0 if r["reproduced"] else 1
    matrix = run_matrix(n=args.n, seed=args.seed, mode=args.guardrails, max_steps=args.max_steps)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = "off" if "off" in matrix["runs"] else "on"
    paths = write_repros(matrix, out, base, limit=args.repro_limit or None)
    if args.json == "-":
        print(json.dumps(to_json(matrix), indent=2, default=str))
    else:
        md = render_markdown(matrix)
        print(md)
        print(f"\n{len(paths)} repro records written to {out / 'repro'}")
        (out / "report.md").write_text(md)
        if args.json:
            Path(args.json).write_text(json.dumps(to_json(matrix), indent=2, default=str))
            print(f"json written to {args.json}")
    # exit 1 when the most protected configuration that was run still has failures
    tested = matrix["summaries"].get("on") or matrix["summaries"]["off"]
    return 0 if tested["failures"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
