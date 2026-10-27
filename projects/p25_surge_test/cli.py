"""CLI for the surge harness.

    python -m projects.p25_surge_test.cli run --scenario spike --users 10000 --window 10 --target simulated [--rate-limit] [--json out.json] [--html report.html]
    python -m projects.p25_surge_test.cli run --scenario ramp --users 5000 --target http --url http://localhost:8000/health
    python -m projects.p25_surge_test.cli headroom --scenario spike --users 10000 --target simulated [--launch-users 10000]

Exit codes: run 0 (1 if the knee is below the scenario's users, i.e. the platform fell over);
headroom 0 when the knee is at or above the launch target, 1 when short; 2 usage error.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from projects.p25_surge_test.core import (
    Scenario,
    analyse,
    build_target,
    headroom,
    load_config,
    render_html,
    render_markdown,
    result_json,
    run_scenario,
)

HERE = Path(__file__).parent
DATA = HERE / "data"


def _scenario(a, cfg) -> Scenario:
    base = dict(cfg.get("scenarios", {}).get(a.scenario, {"kind": a.scenario}))
    base.setdefault("kind", a.scenario)
    return Scenario.from_config(base, users=a.users, duration=a.duration, window=a.window, hold=a.hold,
                                start_users=a.start_users, step_seconds=a.step_seconds, seed=a.seed, name=a.scenario)


def _run(a):
    cfg = load_config(a.config)
    sc = _scenario(a, cfg)
    virtual = a.target == "simulated" and not a.real_time
    target, name = build_target(a.target, cfg, url=a.url, seed=a.seed, virtual=virtual, rate_limit=a.rate_limit)
    res = run_scenario(sc, target, virtual=virtual, max_workers=a.workers, time_scale=a.time_scale, target_name=name)
    an = analyse(res)
    launch = a.launch_users or cfg.get("launch_target_users") or sc.users
    hd = headroom(int(launch), an.knee_users, sc.users)
    return res, an, hd


def cmd_run(a) -> int:
    try:
        res, an, hd = _run(a)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(render_markdown(res, an, hd))
    if a.json:
        Path(a.json).write_text(result_json(res, an, hd))
        print(f"\njson -> {a.json}")
    if a.html:
        Path(a.html).write_text(render_html(res, an, hd))
        print(f"html -> {a.html}")
    return 1 if an.knee_users is not None and an.knee_users < res.scenario.users else 0


def cmd_headroom(a) -> int:
    try:
        res, an, hd = _run(a)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2
    print("## headroom\n")
    print(f"- scenario: {res.scenario.kind}, {res.scenario.users:,} users over {res.scenario.total_seconds} s -> {res.target_name}")
    print(f"- knee: {an.knee_users if an.knee_users is not None else 'none'} ({an.knee_reason}); max sustainable: {an.max_sustainable_users:,} users")
    print(f"- **{hd.statement}**")
    if a.json:
        Path(a.json).write_text(result_json(res, an, hd))
    return 1 if hd.short else 0


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scenario", default="spike", help="ramp | spike | soak | step (names in scenarios.yaml)")
    p.add_argument("--users", type=int, default=None)
    p.add_argument("--duration", type=int, default=None, help="ramp/soak seconds")
    p.add_argument("--window", type=int, default=None, help="spike: seconds in which everyone joins")
    p.add_argument("--hold", type=int, default=None, help="spike: seconds held after the window")
    p.add_argument("--start-users", type=int, default=None, help="step: first level")
    p.add_argument("--step-seconds", type=int, default=None)
    p.add_argument("--target", default="simulated", choices=["simulated", "http", "function"])
    p.add_argument("--url", default=None)
    p.add_argument("--rate-limit", action="store_true", help="wrap the target in the token-bucket RateLimiter")
    p.add_argument("--real-time", action="store_true", help="simulated target on the wall clock through the thread pool")
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--time-scale", type=float, default=1.0, help="thread mode: 0.1 runs a 60 s scenario in 6 s")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default=str(DATA / "scenarios.yaml"))
    p.add_argument("--launch-users", type=int, default=None)
    p.add_argument("--json", default=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p25_surge_test", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a scenario and print the report")
    _add_common(r)
    r.add_argument("--html", default=None)
    r.set_defaults(fn=cmd_run)
    h = sub.add_parser("headroom", help="run a scenario and print only the capacity headroom statement")
    _add_common(h)
    h.set_defaults(fn=cmd_headroom)
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        a = build_parser().parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
