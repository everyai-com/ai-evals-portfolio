"""CLI: `tick` once, `run` the scheduler, `render` the page, `check` a real URL one-shot."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from evalkit.report import md_table
from projects.p17_status_page.core import (
    FakeProber,
    Monitor,
    Probe,
    Scheduler,
    SimClock,
    StatusStore,
    UrllibProber,
    compute_state,
    iso,
    load_probes,
    render_html,
    render_json,
)

HERE = Path(__file__).parent
DEFAULT_START = "2026-10-24T00:00:00Z"


def parse_ts(s: str | None) -> float:
    if not s:
        return time.time()
    return datetime.fromisoformat(s).astimezone(UTC).timestamp()


def build_monitor(args: argparse.Namespace) -> tuple[Monitor, float]:
    probes, settings = load_probes(args.probes)
    interval = float(getattr(args, "interval", None) or settings["interval_s"])
    scenario = getattr(args, "fake_scenario", None)
    if getattr(args, "fake", False) or scenario:
        prober = FakeProber.from_yaml(scenario or HERE / "data" / "scenario.yaml")
    else:
        prober = UrllibProber()
    return Monitor(probes, prober, StatusStore(args.state), interval_s=interval), interval


def describe(summary) -> str:
    bits = [f"{name}={status}" for name, status in summary.statuses.items()]
    extra = "".join(f"  OPENED {i.id} ({', '.join(i.regions)}: {i.cause})" for i in summary.opened)
    extra += "".join(f"  CLOSED {i.id}" for i in summary.closed)
    return f"{iso(summary.ts)}  " + "  ".join(bits) + extra


def _tick(args: argparse.Namespace) -> int:
    monitor, _ = build_monitor(args)
    summary = monitor.tick(parse_ts(args.now))
    print(describe(summary))
    return 1 if "down" in summary.statuses.values() else 0


def _run(args: argparse.Namespace) -> int:
    monitor, interval = build_monitor(args)
    if args.reset:
        monitor.store.clear()
    if args.fake or args.fake_scenario:
        clock = SimClock(parse_ts(args.start or DEFAULT_START))
        sched = Scheduler(monitor, clock=clock, sleep=clock.sleep)
    else:
        sched = Scheduler(monitor)
    n = sched.run(interval, ticks=args.ticks, on_tick=lambda s: print(describe(s)))
    state = compute_state(monitor.probes, monitor.store, sched.clock(), interval)
    print(f"\n{n} ticks; overall: {state['overall']}; open incidents: {len(state['open_incidents'])}; "
          f"history: {len(state['incident_history'])}")
    print(md_table(["probe", "status", "24h", "7d", "30d"],
                   [[p["name"], p["status"]] + [f"{p['uptime'][w] * 100:.2f}%" if p["uptime"][w] is not None else "n/a"
                                                 for w in ("24h", "7d", "30d")] for p in state["probes"]]))
    return 1 if state["open_incidents"] else 0


def _render(args: argparse.Namespace) -> int:
    probes, settings = load_probes(args.probes)
    store = StatusStore(args.state)
    state = compute_state(probes, store, parse_ts(args.now), args.interval or settings["interval_s"])
    if args.json:
        Path(args.json).write_text(render_json(state), encoding="utf-8")
    html = render_html(state)
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"wrote {args.out} ({len(html)} bytes); overall: {state['overall']}; stale: {state['stale']}; "
          f"last tick: {state['last_tick_iso']}")
    return 1 if state["stale"] or state["open_incidents"] else 0


def _check(args: argparse.Namespace) -> int:
    regions = tuple(args.regions)
    probe = Probe(name=args.name, kind=args.kind, target=args.url, regions=regions, home_region=regions[0],
                  timeout_s=args.timeout, expected_status=args.expected_status, body_regex=args.body_regex,
                  quorum=min(2, len(regions)))
    prober = UrllibProber()
    now = time.time()
    results = [prober.probe(probe, r, now) for r in regions]
    print(md_table(["region", "ok", "status", "latency ms", "error"],
                   [[r.region, r.ok, r.status if r.status is not None else "-", f"{r.latency_ms:.0f}", r.error or ""]
                    for r in results]))
    failed = {r.region for r in results if not r.ok}
    from projects.p17_status_page.core import tick_status
    status = tick_status(probe, failed)
    print(f"verdict: {status}")
    return 0 if status != "down" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p17_status_page", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--state", default="state/", help="state dir (results.jsonl, incidents.jsonl, meta.json)")
        sp.add_argument("--probes", default=str(HERE / "data" / "probes.yaml"))
        sp.add_argument("--interval", type=float, default=None, help="seconds between ticks (default: yaml)")

    t = sub.add_parser("tick", help="run every probe once and update state")
    common(t)
    t.add_argument("--fake-scenario", default=None, help="scenario.yaml for the FakeProber")
    t.add_argument("--fake", action="store_true", help="use the FakeProber with the default scenario")
    t.add_argument("--now", default=None, help="ISO timestamp for this tick (default: wall clock)")
    t.set_defaults(fn=_tick)

    r = sub.add_parser("run", help="scheduler: tick every --interval seconds for --ticks ticks")
    common(r)
    r.add_argument("--ticks", type=int, default=None)
    r.add_argument("--fake", action="store_true", help="FakeProber + simulated clock (no sleeping)")
    r.add_argument("--fake-scenario", default=None)
    r.add_argument("--start", default=None, help="simulated start time (ISO) when --fake")
    r.add_argument("--reset", action="store_true", help="clear the state dir first")
    r.set_defaults(fn=_run)

    d = sub.add_parser("render", help="write status.html (and optional JSON) from the state dir")
    common(d)
    d.add_argument("--out", default="status.html")
    d.add_argument("--json", default=None)
    d.add_argument("--now", default=None, help="ISO timestamp used for staleness / windows")
    d.set_defaults(fn=_render)

    c = sub.add_parser("check", help="one-shot real probe of a URL from the given region labels")
    c.add_argument("--url", required=True)
    c.add_argument("--regions", nargs="+", default=["local"])
    c.add_argument("--kind", choices=["http", "tcp", "function"], default="http")
    c.add_argument("--name", default="adhoc")
    c.add_argument("--timeout", type=float, default=5.0)
    c.add_argument("--expected-status", type=int, default=200)
    c.add_argument("--body-regex", default=None)
    c.set_defaults(fn=_check)
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
