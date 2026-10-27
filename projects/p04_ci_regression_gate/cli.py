"""CLI for the CI regression gate.

    python -m projects.p04_ci_regression_gate.cli run --config data/eval_config.yaml --variant pr_regressed --out results.json
    python -m projects.p04_ci_regression_gate.cli gate --results results.json --baseline baseline/main.json \
        --config data/eval_config.yaml --comment gate_comment.md --state gate_state.json
    python -m projects.p04_ci_regression_gate.cli baseline --config data/eval_config.yaml --out baseline/main.json

Exit codes: 0 pass, 1 gate blocked, 2 usage error / cost budget exceeded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from projects.p04_ci_regression_gate.core import (
    BudgetExceeded,
    EvalCache,
    EvalConfig,
    EvalRunner,
    Gate,
    GateState,
    render_comment,
    write_github_outputs,
)
from projects.p04_ci_regression_gate.demo_sut import VARIANTS, make_sut

HERE = Path(__file__).parent


def _run(config_path: str, variant: str, cache_dir: str | None) -> dict:
    cfg = EvalConfig.load(config_path)
    examples = cfg.load_examples()
    sut = make_sut(variant, examples, model=cfg.model)
    runner = EvalRunner(cfg, sut, cache=EvalCache(cache_dir), variant=variant)
    return runner.run(examples)


def cmd_run(a: argparse.Namespace) -> int:
    variant = a.variant or os.environ.get("EVAL_VARIANT", "main")
    try:
        res = _run(a.config, variant, None if a.no_cache else a.cache_dir)
    except BudgetExceeded as e:
        print(f"COST BUDGET EXCEEDED: {e}", file=sys.stderr)
        return 2
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    ag, c = res["aggregates"], res["cache"]
    print(f"## eval run `{variant}`\n")
    print(f"- task_success: {ag['task_success'] * 100:.1f}% over {ag['n_items']} items x {len(res['meta']['seeds'])} seeds")
    print(f"- latency p50/p95: {ag['latency_p50_ms']:.0f} / {ag['latency_p95_ms']:.0f} ms")
    print(f"- cost: ${ag['cost_total_usd']:.4f} (${ag['cost_per_task']:.6f} per task)")
    print(f"- cache: {c['hits']} hits / {c['misses']} misses (hit rate {c['hit_rate'] * 100:.0f}%)")
    print(f"- wrote {a.out}")
    return 0


def cmd_baseline(a: argparse.Namespace) -> int:
    a.variant, a.no_cache, a.cache_dir = "main", True, None
    return cmd_run(a)


def cmd_gate(a: argparse.Namespace) -> int:
    cfg = EvalConfig.load(a.config)
    if a.max_drop is not None:
        cfg.gate["max_drop"] = a.max_drop
    baseline = json.loads(Path(a.baseline).read_text())
    current = json.loads(Path(a.results).read_text())
    state = GateState(a.state)
    result = Gate(cfg, state).evaluate(baseline, current)
    state.save()
    comment = render_comment(result, baseline, current)
    if a.comment:
        Path(a.comment).parent.mkdir(parents=True, exist_ok=True)
        Path(a.comment).write_text(comment)
    print(comment)
    outputs = write_github_outputs(result, current, a.comment)
    if a.json:
        Path(a.json).write_text(json.dumps({**result.to_dict(), "outputs": outputs}, indent=1))
    return 1 if result.blocked else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p04_ci_regression_gate", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the eval for a variant and write results.json")
    r.add_argument("--config", default=str(HERE / "data" / "eval_config.yaml"))
    r.add_argument("--variant", choices=VARIANTS, default=None, help="defaults to $EVAL_VARIANT or main")
    r.add_argument("--out", default="results.json")
    r.add_argument("--cache-dir", default=".eval-cache")
    r.add_argument("--no-cache", action="store_true")
    r.set_defaults(fn=cmd_run)
    b = sub.add_parser("baseline", help="regenerate the committed baseline from the main variant")
    b.add_argument("--config", default=str(HERE / "data" / "eval_config.yaml"))
    b.add_argument("--out", default=str(HERE / "baseline" / "main.json"))
    b.set_defaults(fn=cmd_baseline)
    g = sub.add_parser("gate", help="compare results to the baseline; exit 1 when blocked")
    g.add_argument("--results", required=True)
    g.add_argument("--baseline", default=str(HERE / "baseline" / "main.json"))
    g.add_argument("--config", default=str(HERE / "data" / "eval_config.yaml"))
    g.add_argument("--comment", default="gate_comment.md")
    g.add_argument("--state", default="gate_state.json")
    g.add_argument("--max-drop", type=float, default=None, help="override gate.max_drop (points)")
    g.add_argument("--json", default=None, help="write the gate decision as JSON")
    g.set_defaults(fn=cmd_gate)
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        a = build_parser().parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
