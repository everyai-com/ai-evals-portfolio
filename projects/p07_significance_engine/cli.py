"""CLI for the significance engine.

    compare --a data/model_a.jsonl --b data/model_b.jsonl [--metric success] [--n-boot 10000] [--seed 0] [--json [out.json]] [--plot out.svg]
    multi   --baseline a.jsonl --candidates b.jsonl c.jsonl [--names B C]
    power   --baseline 0.7 --delta 0.02
    seeds   --runs data/runs.jsonl [--metric success]

Exit codes: 0 ok, 1 the comparison is not trustworthy (power warning / seed-variance warning),
2 usage error (bad format, unpaired ids with --strict).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evalkit.io import read_jsonl
from evalkit.report import md_table

from . import core

HERE = Path(__file__).parent


def _emit(markdown: str, payload: dict, json_opt: str | None) -> None:
    if json_opt == "-":
        print(json.dumps(payload, indent=2, default=float))
        return
    print(markdown)
    if json_opt:
        Path(json_opt).write_text(json.dumps(payload, indent=2, default=float))
        print(f"\njson written to {json_opt}")


def _cmd_compare(a: argparse.Namespace) -> int:
    try:
        data = core.load_pair(a.a, a.b, a.metric, strict=a.strict, strata_field=a.strata_field)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    res = core.compare_paired(
        data,
        n_boot=a.n_boot,
        seed=a.seed,
        alpha=a.alpha,
        keep_boots=bool(a.plot),
        label_a=Path(a.a).stem,
        label_b=Path(a.b).stem if a.b else "b",
    )
    _emit(res.to_markdown(), res.to_dict(), a.json)
    if a.plot:
        svg = core.svg_histogram(
            res.boots,
            res.report.ci_lo,
            res.report.ci_hi,
            res.report.delta,
            title=f"bootstrap distribution of delta on {res.report.metric} (n={res.report.n})",
        )
        Path(a.plot).write_text(svg)
        print(f"svg written to {a.plot}")
    return 1 if res.power_warning else 0


def _cmd_multi(a: argparse.Namespace) -> int:
    names = a.names or [Path(p).stem for p in a.candidates]
    if len(names) != len(a.candidates):
        print("error: --names must match --candidates", file=sys.stderr)
        return 2
    try:
        res = core.multi_compare(
            read_jsonl(a.baseline),
            {n: read_jsonl(p) for n, p in zip(names, a.candidates)},
            a.metric,
            n_boot=a.n_boot,
            seed=a.seed,
            alpha=a.alpha,
            strict=a.strict,
            baseline_name=Path(a.baseline).stem,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    _emit(res.to_markdown(), res.to_dict(), a.json)
    return 0


def _cmd_power(a: argparse.Namespace) -> int:
    deltas = sorted({a.delta, 0.01, 0.02, 0.03, 0.05, 0.10})
    table = core.power_table(a.baseline, deltas, a.power, a.alpha)
    n_req = core.required_sample_size(a.baseline, a.delta, a.power, a.alpha)
    md = (
        f"# Required sample size (baseline={a.baseline}, power={a.power}, alpha={a.alpha})\n\n"
        f"To detect a {a.delta * 100:.1f}-point change from {a.baseline:.0%} you need **n={n_req} per arm** "
        f"(unpaired two-proportion test; a paired design needs fewer).\n\n"
        + md_table(
            ["delta", "required n per arm"],
            [[f"{t['delta']:+.3f}", t["required_n_per_arm"]] for t in table],
        )
    )
    _emit(
        md, {"baseline": a.baseline, "delta": a.delta, "required_n": n_req, "table": table}, a.json
    )
    return 0


def _cmd_seeds(a: argparse.Namespace) -> int:
    rows = read_jsonl(a.runs)
    try:
        if core.detect_format(rows) != "runs":
            raise ValueError("expected runs format {id, model, seed, score|success}")
        s = core.seed_analysis(rows, a.metric, a.baseline_model)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    _emit(core.seeds_markdown(s), s, a.json)
    return 1 if s["any_warning"] else 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="evalkit stats", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--metric", default=None, help="field to compare (default: score, then success)"
        )
        p.add_argument("--n-boot", type=int, default=10_000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--alpha", type=float, default=0.05)
        p.add_argument(
            "--strict", action="store_true", help="refuse unpaired ids instead of warning"
        )
        p.add_argument(
            "--json", nargs="?", const="-", default=None, help="JSON to stdout, or to PATH"
        )

    c = sub.add_parser("compare", help="A vs B on paired per-example results")
    c.add_argument("--a", required=True, help="results for A (or a single {id,a,b} file)")
    c.add_argument("--b", default=None, help="results for B")
    c.add_argument(
        "--strata-field", default="strata", help="field (or 'tags') for the per-strata breakdown"
    )
    c.add_argument(
        "--plot", default=None, help="write an SVG histogram of the bootstrap distribution"
    )
    common(c)
    c.set_defaults(fn=_cmd_compare)

    m = sub.add_parser("multi", help="several candidates vs one baseline with Holm-Bonferroni")
    m.add_argument("--baseline", required=True)
    m.add_argument("--candidates", nargs="+", required=True)
    m.add_argument("--names", nargs="*", default=None)
    common(m)
    m.set_defaults(fn=_cmd_multi)

    p = sub.add_parser("power", help="required sample size for a delta on a proportion")
    p.add_argument("--baseline", type=float, required=True)
    p.add_argument("--delta", type=float, required=True)
    p.add_argument("--power", type=float, default=0.8)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--json", nargs="?", const="-", default=None)
    p.set_defaults(fn=_cmd_power)

    s = sub.add_parser("seeds", help="seed variance across multi-seed runs")
    s.add_argument("--runs", required=True)
    s.add_argument("--metric", default=None)
    s.add_argument("--baseline-model", default=None)
    s.add_argument("--json", nargs="?", const="-", default=None)
    s.set_defaults(fn=_cmd_seeds)
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
