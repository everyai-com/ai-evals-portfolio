"""CLI for the calibrated judge.

    python -m projects.p03_calibrated_judge.cli make-anchors --n 500 --seed 0
    python -m projects.p03_calibrated_judge.cli calibrate --anchors data/anchors.jsonl --judge biased [--json out.json]
    python -m projects.p03_calibrated_judge.cli kappa --anchors data/anchors.jsonl

Exit codes: 0 judge passes every threshold, 1 at least one bias needs debiasing, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from evalkit.io import read_jsonl
from evalkit.llm import get_client
from evalkit.report import md_table

from . import core
from .data.make_anchors import write_dataset

HERE = Path(__file__).parent


def _cmd_make_anchors(a: argparse.Namespace) -> int:
    info = write_dataset(Path(a.out), a.n, a.seed)
    print(
        f"wrote {info['n_rows']} anchors to {a.out}\nsha256: {info['sha256']}\n"
        f"annotator agreement: {info['agreement']:.3f}, kappa: {info['kappa']:.3f}"
    )
    return 0


def _cmd_calibrate(a: argparse.Namespace) -> int:
    anchors = read_jsonl(a.anchors)
    if a.sample:
        anchors = core.stratified_sample(anchors, n=a.sample, seed=a.seed)
    if os.environ.get("EVALKIT_LLM", "mock").lower() == "anthropic":
        client = get_client()
        name = f"anthropic:{client.model}"
    else:
        overrides = {
            k: v
            for k, v in {
                "position_bias": a.position_bias,
                "verbosity_bias": a.verbosity_bias,
                "self_preference": a.self_preference,
            }.items()
            if v is not None
        }
        client = core.make_mock_judge(anchors, a.judge, seed=a.seed, **overrides)
        name = f"mock:{a.judge}"
    judge = core.PairwiseJudge(client)
    if a.debias:
        judge = core.SwapDebiasedJudge(judge)
        name += "+swap-debiased"
    result = core.calibrate(judge, anchors, judge_name=name, seed=a.seed)
    print(result.to_markdown())
    if a.json:
        Path(a.json).write_text(json.dumps(result.to_dict(), indent=2, default=list))
        print(f"\njson written to {a.json}")
    if a.verdicts:
        core.write_verdicts(a.verdicts, anchors, result)
    return 1 if result.flagged else 0


def _cmd_kappa(a: argparse.Namespace) -> int:
    anchors = read_jsonl(a.anchors)
    h = core.human_agreement(anchors)
    print("# Inter-annotator agreement\n")
    print(
        md_table(
            ["measure", "value"],
            [
                ["n", h["n"]],
                ["raw agreement", h["raw_agreement"]],
                ["Cohen's kappa", h["kappa"]],
                ["band (Landis & Koch)", h["band"]],
            ],
        )
    )
    print("\n" + md_table(["strata", "kappa"], [[s, k] for s, k in h["per_strata"].items()]))
    if a.json:
        Path(a.json).write_text(json.dumps(h, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="evalkit judge", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-anchors", help="generate the golden anchor dataset + CHANGELOG")
    m.add_argument("--n", type=int, default=500)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", default=str(HERE / "data" / "anchors.jsonl"))
    m.set_defaults(fn=_cmd_make_anchors)
    c = sub.add_parser("calibrate", help="measure judge biases against the anchors")
    c.add_argument("--anchors", default=str(HERE / "data" / "anchors.jsonl"))
    c.add_argument(
        "--judge",
        choices=["biased", "unbiased"],
        default="unbiased",
        help="mock judge flavour (ignored when EVALKIT_LLM=anthropic)",
    )
    c.add_argument("--position-bias", type=float, default=None)
    c.add_argument("--verbosity-bias", type=float, default=None)
    c.add_argument("--self-preference", type=float, default=None)
    c.add_argument("--debias", action="store_true", help="wrap the judge with swap-debiasing")
    c.add_argument("--sample", type=int, default=0, help="stratified subsample size (0 = all)")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--json", default=None, help="write the report as JSON to this path")
    c.add_argument("--verdicts", default=None, help="write per-anchor verdicts JSONL")
    c.set_defaults(fn=_cmd_calibrate)
    k = sub.add_parser("kappa", help="inter-annotator agreement of the anchors")
    k.add_argument("--anchors", default=str(HERE / "data" / "anchors.jsonl"))
    k.add_argument("--json", default=None)
    k.set_defaults(fn=_cmd_kappa)
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
