"""CLI: `run` replays a traffic file through the shadow proxy; `report` summarises a store."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evalkit.io import read_jsonl
from projects.p02_shadow_router.core import JUDGES, JsonlStore, ShadowProxy
from projects.p02_shadow_router.demo import primary_client, shadow_client
from projects.p02_shadow_router.report import ReportBuilder, load_thresholds

HERE = Path(__file__).parent


def _run(args: argparse.Namespace) -> int:
    traffic = read_jsonl(args.traffic)
    store = JsonlStore(args.out)
    if not args.append:
        store.clear()
    with ShadowProxy(primary_client(), shadow_client(latency_ms=args.shadow_latency_ms,
                                                     fail_rate=args.shadow_fail_rate, seed=args.seed),
                     sample_rate=args.rate, seed=args.seed, store=store,
                     shadow_timeout_ms=args.shadow_timeout_ms) as proxy:
        for row in traffic:
            meta = {k: v for k, v in row.items() if k not in ("prompt", "request_id")}
            proxy.complete(row["prompt"], request_id=row.get("request_id"), meta=meta)
    st = proxy.stats
    print(f"replayed {st['requests']} requests; shadowed {st['shadowed']} "
          f"({st['shadowed'] / max(1, st['requests']):.1%}); shadow errors {st['shadow_errors']}, "
          f"timeouts {st['shadow_timeouts']}; store -> {args.out}")
    return 0


def _report(args: argparse.Namespace) -> int:
    thresholds = load_thresholds(args.config)
    rb = ReportBuilder(read_jsonl(args.store), thresholds=thresholds, seed=args.seed,
                       judge=JUDGES[args.judge])
    summary = rb.build()
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(rb.to_markdown(summary))
    blocked = {"reject": {"reject"}, "hold": {"reject", "hold"}}[args.fail_on]
    return 1 if summary["recommendation"] in blocked else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p02_shadow_router", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="replay traffic through the shadow proxy")
    r.add_argument("--traffic", default=str(HERE / "data" / "traffic.jsonl"))
    r.add_argument("--rate", type=float, default=0.05, help="shadow sample rate (0-1)")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", default="runs/shadow.jsonl")
    r.add_argument("--append", action="store_true", help="append to an existing store")
    r.add_argument("--shadow-latency-ms", type=float, default=2.0)
    r.add_argument("--shadow-fail-rate", type=float, default=0.02)
    r.add_argument("--shadow-timeout-ms", type=float, default=None)
    r.set_defaults(fn=_run)
    q = sub.add_parser("report", help="markdown cost/quality report from a store")
    q.add_argument("--store", default="runs/shadow.jsonl")
    q.add_argument("--config", default=str(HERE / "data" / "thresholds.yaml"))
    q.add_argument("--json", default=None, help="also write the summary as JSON")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--judge", choices=sorted(JUDGES), default="normalized",
                   help="answer-agreement judge (normalized string match, or containment)")
    q.add_argument("--fail-on", choices=["reject", "hold"], default="reject",
                   help="exit 1 when the recommendation is this bad or worse")
    q.set_defaults(fn=_report)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:  # argparse usage error -> 2
        return 0 if e.code == 0 else 2
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
