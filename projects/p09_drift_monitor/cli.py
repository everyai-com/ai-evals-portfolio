"""CLI: `simulate` a drifting fortnight, `run-once` on a real traffic file, `report` the state dir."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

from evalkit.io import read_jsonl
from evalkit.llm import MockLLM, get_client
from evalkit.report import md_table, pct
from projects.p09_drift_monitor.core import (DriftDetector, JudgeScorer, LengthScorer, OfflineEvaluator,
                                             RuleScorer, TrafficSampler, default_judge_handler,
                                             load_thresholds)
from projects.p09_drift_monitor.daemon import Daemon, StdoutSink
from projects.p09_drift_monitor.synth import date_for, make_day

HERE = Path(__file__).parent
SPARK = " .:-=+*#%@"


def sparkline(values: list[float], lo: float | None = None, hi: float | None = None) -> str:
    """ASCII sparkline: each value mapped onto 10 levels between lo and hi."""
    if not values:
        return ""
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    span = (hi - lo) or 1.0
    return "".join(SPARK[min(9, max(0, int((v - lo) / span * 9.999)))] for v in values)


def build_components(args: argparse.Namespace) -> tuple[TrafficSampler, OfflineEvaluator, DriftDetector]:
    schema = yaml.safe_load(Path(args.schema).read_text(encoding="utf-8"))
    judge = get_client("mock-judge", handlers=[(r".", default_judge_handler)])
    if isinstance(judge, MockLLM):
        judge.handlers = MockLLM(handlers=[(r".", default_judge_handler)]).handlers
    sampler = TrafficSampler(rate=args.rate, seed=args.seed, stratify_by=args.stratify,
                             feedback_weight=args.feedback_weight)
    evaluator = OfflineEvaluator([RuleScorer(schema), JudgeScorer(judge), LengthScorer()])
    return sampler, evaluator, DriftDetector(load_thresholds(args.config))


def make_daemon(args: argparse.Namespace, source: Any, quiet: bool = False) -> Daemon:
    sampler, evaluator, detector = build_components(args)
    # alerts.jsonl is written by the Daemon itself; StdoutSink is the live notification channel.
    return Daemon(source, sampler, evaluator, detector, args.state, sinks=[] if quiet else [StdoutSink()])


def _simulate(args: argparse.Namespace) -> int:
    dates = {date_for(d, args.start): d for d in range(args.days)}

    def source(date: str) -> list[dict[str, Any]]:
        return make_day(dates[date], args.drift_day, args.seed, args.per_day, args.start)

    daemon = make_daemon(args, source)
    for p in (daemon.metrics_path, daemon.alerts_path):
        if p.exists():
            p.unlink()
    for date in dates:
        m, alerts = daemon.run_once(date)
        print(f"{date}: sampled {m.n}, success {m.success_rate:.3f}, halluc {m.hallucination_rate:.3f}, "
              f"alerts {len(alerts)}")
    print()
    print(render_report(daemon))
    return 0


def _run_once(args: argparse.Namespace) -> int:
    daemon = make_daemon(args, lambda date: read_jsonl(args.traffic))
    _, alerts = daemon.run_once(args.date)
    print(f"{args.date}: {len(alerts)} alert(s); state -> {args.state}")
    return 1 if any(a.severity == "critical" for a in alerts) else 0


def render_report(daemon: Daemon) -> str:
    metrics = daemon.load_metrics()
    alerts = daemon.load_alerts()
    by_date: dict[str, list] = {}
    for a in alerts:
        by_date.setdefault(a.date, []).append(a)
    rows = [[m.date, m.n, pct(m.success_rate), pct(m.hallucination_rate), f"{m.avg_steps:.2f}",
             f"{m.p95_latency_ms:.0f}", f"{m.cost_per_request:.5f}", f"{m.avg_answer_words:.0f}",
             ",".join(sorted({f"{a.detector}:{a.severity[0].upper()}" for a in by_date.get(m.date, [])})) or "-"]
            for m in metrics]
    out = ["# Drift monitor report", "",
           f"Days: {len(metrics)}; alerts: {len(alerts)} "
           f"({sum(a.severity == 'critical' for a in alerts)} critical)", "",
           "```",
           f"success rate   {sparkline([m.success_rate for m in metrics], 0.5, 1.0)}",
           f"hallucination  {sparkline([m.hallucination_rate for m in metrics], 0.0, 0.2)}",
           f"answer words   {sparkline([m.avg_answer_words for m in metrics])}",
           f"tool calls     {sparkline([m.avg_steps for m in metrics])}",
           "```", "",
           md_table(["date", "n", "success", "halluc", "avg steps", "p95 ms", "cost/req", "words", "alerts"], rows)]
    if metrics:
        last = metrics[-1]
        out += ["", f"## Per-tenant ({last.date})",
                md_table(["tenant", "n", "success", "95% CI", "halluc", "cost/req"],
                         [[t, v["n"], pct(v["success_rate"]), f"{v['success_ci_lo']:.2f}-{v['success_ci_hi']:.2f}",
                           pct(v["hallucination_rate"]), f"{v['cost_per_request']:.5f}"] for t, v in last.per_tenant.items()])]
    out += ["", "## Alerts",
            md_table(["date", "severity", "detector", "metric", "baseline", "current", "evidence", "sample ids"],
                     [[a.date, a.severity, a.detector, a.metric, f"{a.baseline:.3f}", f"{a.current:.3f}", a.evidence,
                       " ".join(a.sample_trajectory_ids[:3])] for a in alerts]) if alerts else "_none_"]
    return "\n".join(out) + "\n"


def _report(args: argparse.Namespace) -> int:
    daemon = make_daemon(args, lambda date: [], quiet=True)
    print(render_report(daemon))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p09_drift_monitor", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--state", "--out", dest="state", default="state/",
                        help="state dir holding metrics.jsonl and alerts.jsonl")
        sp.add_argument("--rate", type=float, default=0.05)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--stratify", choices=["hour", "tenant"], default="hour")
        sp.add_argument("--feedback-weight", type=float, default=1.0,
                        help="sampling weight for thumbs-down trajectories (1 = uniform)")
        sp.add_argument("--config", default=str(HERE / "data" / "thresholds.yaml"))
        sp.add_argument("--schema", default=str(HERE / "data" / "tool_schema.yaml"))

    s = sub.add_parser("simulate", help="generate drifting traffic and run the daemon day by day")
    s.add_argument("--days", type=int, default=14)
    s.add_argument("--drift-day", type=int, default=10)
    s.add_argument("--per-day", type=int, default=1000)
    s.add_argument("--start", default="2026-10-01")
    common(s)
    s.set_defaults(fn=_simulate)
    r = sub.add_parser("run-once", help="evaluate one day's traffic file against the stored baseline")
    r.add_argument("--traffic", required=True)
    r.add_argument("--date", required=True)
    common(r)
    r.set_defaults(fn=_run_once)
    q = sub.add_parser("report", help="markdown trend table + alerts from a state dir")
    common(q)
    q.set_defaults(fn=_report)
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
