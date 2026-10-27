"""Seeded generator for daily storage snapshots of "the closet": one database holding everything.

Writes `snapshots.jsonl`, one row per day for 90 days:

    {"day": 0, "date": "2026-01-01", "cap_bytes": 10737418240,
     "tables": [{"name": "users", "category": "platform", "rows": 1200, "bytes": 3400000}, ...]}

Story encoded in the numbers (the tests assert it):
* `platform` tables (users, apps, deployments, sessions) grow slowly and linearly.
* `analytics` tables (click / page-view / deploy events) grow fastest and *accelerate* (every
  new app adds its own event stream), so by day 59 they are roughly half the database.
* `versions` (every app version ever shipped) grows linearly and never shrinks.
* `logs` rotate after 30 days, so they plateau.
* On day 60 the analytics events are migrated to an analytics store: the analytics tables drop
  to a small residual and the line falls; growth afterwards is platform + versions only.

Usage: python -m projects.p20_capacity_line.data.make_snapshots --seed 0 --days 90
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from evalkit.io import write_jsonl

GB = 1024**3
MB = 1024**2
CAP_BYTES = 10 * GB
MIGRATION_DAY = 60

# name, category, start_bytes, bytes_per_day, bytes_per_row
TABLES = [
    ("users", "platform", 180 * MB, 1.2 * MB, 1500),
    ("apps", "platform", 240 * MB, 2.0 * MB, 4000),
    ("deployments", "platform", 420 * MB, 4.5 * MB, 2500),
    ("sessions", "platform", 300 * MB, 3.0 * MB, 800),
    ("events_click", "analytics", 500 * MB, 0.0, 120),
    ("events_pageview", "analytics", 700 * MB, 0.0, 140),
    ("events_deploy", "analytics", 120 * MB, 0.0, 300),
    ("app_versions", "versions", 600 * MB, 14.0 * MB, 250_000),
    ("request_logs", "logs", 400 * MB, 0.0, 600),
    ("misc", "other", 90 * MB, 0.1 * MB, 1000),
]


def analytics_bytes(day: int, start: float, accel: float, rng: np.random.Generator) -> float:
    """Accelerating growth: start + a*d + b*d^2 with mild noise; after migration a small residual."""
    if day >= MIGRATION_DAY:
        residual = 0.02 * start + 0.3 * MB * (day - MIGRATION_DAY)
        return residual * (1 + rng.normal(0, 0.01))
    grown = start + accel * 0.6 * day * MB + accel * 0.012 * day * day * MB
    return grown * (1 + rng.normal(0, 0.01))


def log_bytes(day: int, start: float, rng: np.random.Generator) -> float:
    """Logs rotate at 30 days: linear growth, then a plateau (with daily jitter)."""
    per_day = 20 * MB
    return (start + per_day * min(day, 30)) * (1 + rng.normal(0, 0.02))


def make_snapshots(days: int = 90, seed: int = 0, start_date: date = date(2026, 1, 1)) -> list[dict]:
    rng = np.random.default_rng(seed)
    accel = {"events_click": 15.0, "events_pageview": 21.0, "events_deploy": 4.5}
    rows_out = []
    for d in range(days):
        tables = []
        for name, cat, start, per_day, bpr in TABLES:
            if cat == "analytics":
                b = analytics_bytes(d, start, accel[name], rng)
            elif cat == "logs":
                b = log_bytes(d, start, rng)
            else:
                b = (start + per_day * d) * (1 + rng.normal(0, 0.005))
            b = max(1.0, b)
            tables.append({"name": name, "category": cat, "bytes": int(b), "rows": int(b / bpr)})
        rows_out.append(
            {
                "day": d,
                "date": (start_date + timedelta(days=d)).isoformat(),
                "cap_bytes": CAP_BYTES,
                "tables": tables,
            }
        )
    return rows_out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--out", default=str(Path(__file__).parent / "snapshots.jsonl"))
    a = ap.parse_args(argv)
    n = write_jsonl(a.out, make_snapshots(a.days, a.seed))
    print(f"wrote {n} snapshots -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
