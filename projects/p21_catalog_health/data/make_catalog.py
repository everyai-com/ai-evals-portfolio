"""Seeded generator for a community catalog with a realistic long tail.

Writes `catalog.jsonl` (~2,000 items) and `featured.yaml` (admin pins):

* ~90% of items have <= 1 distinct user in the last 30 days: to-do apps, "hello world 0..100",
  scratch and demo projects, most never used again after the day they were built.
* a middle band of team tools with a handful of users each,
* a dozen hubs with hundreds of users (deal desk, HR assistant, helpdesk agent ...), which are
  also the org-featured items,
* duplicate-name clusters ("todo app", "To-Do", "todo app v2", "hello world 7" ...),
* a few planted "rising" items whose last-7-day invocations jumped week over week.

Fields per item: id, name, owner, team, kind, created, last_used, invocations_30d,
distinct_users_30d, invocations_7d, invocations_prev_7d, stars, versions, tags, risk_tier,
users_by_team. Dates are relative to AS_OF (2026-10-24).

Usage: python -m projects.p21_catalog_health.data.make_catalog --seed 0 --n 2000
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import yaml

from evalkit.io import write_jsonl

AS_OF = date(2026, 10, 24)
TEAMS = ["sales", "support", "engineering", "hr", "finance", "marketing", "legal", "ops"]
ROLES = ["manager", "agent", "analyst", "engineer", "rep"]
KINDS = ["app", "agent", "mcp_server", "automation"]
TAIL_NAMES = [
    "todo app", "To-Do", "todo app v2", "Todo App", "to do list", "my todo", "TODO (copy)",
    "test", "test app", "my test", "untitled", "Untitled app", "scratch", "demo", "demo app",
    "my first agent", "My First Agent", "first agent (copy)", "sample", "playground",
    "quick test", "new app", "new agent", "chatbot", "Chat Bot", "chat-bot", "notes", "Notes app",
]
MID_NOUNS = ["tracker", "summariser", "digest", "lookup", "helper", "bot", "dashboard", "faq", "triage", "planner"]
HUBS = [
    ("Sales Deal Desk", "sales", "agent", 860),
    ("HR Policy Assistant", "hr", "agent", 740),
    ("IT Helpdesk Agent", "ops", "agent", 690),
    ("Expense Bot", "finance", "automation", 520),
    ("Onboarding Guide", "hr", "app", 480),
    ("Code Review Agent", "engineering", "agent", 410),
    ("Meeting Notes Summariser", "ops", "app", 390),
    ("Benefits FAQ", "hr", "app", 330),
    ("Contract Clause Finder", "legal", "mcp_server", 260),
    ("Customer Churn Signals", "support", "app", 240),
    ("Campaign Brief Writer", "marketing", "agent", 180),
    ("Ticket Triage", "support", "automation", 150),
]


def _date(rng: np.random.Generator, lo_days: int, hi_days: int) -> str:
    return (AS_OF - timedelta(days=int(rng.integers(lo_days, hi_days + 1)))).isoformat()


def _split_by_team(rng: np.random.Generator, users: int, home: str) -> dict[str, int]:
    if users <= 0:
        return {}
    weights = np.full(len(TEAMS), 1.0)
    weights[TEAMS.index(home)] = 6.0
    counts = rng.multinomial(users, weights / weights.sum())
    return {t: int(c) for t, c in zip(TEAMS, counts) if c}


def make_catalog(n: int = 2000, seed: int = 0) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    items: list[dict] = []
    owners = [f"user{int(i):04d}@example.com" for i in rng.integers(0, 3000, size=n)]

    def item(i: int, name: str, team: str, kind: str, users: int, inv_per_user: float, stars: int,
             created_lo: int, created_hi: int, idle_lo: int, idle_hi: int, tags: list[str],
             rising: bool = False) -> dict:
        inv30 = round(users * inv_per_user) if users else 0
        if inv30 == 0 and users:
            inv30 = users
        inv7 = round(inv30 * float(rng.uniform(0.15, 0.3))) if inv30 else 0
        prev7 = round(inv30 * float(rng.uniform(0.15, 0.3))) if inv30 else 0
        if rising:
            prev7 = max(5, prev7 // 4)
            inv7 = prev7 * int(rng.integers(4, 9))
            inv30 = max(inv30, inv7 + prev7)
        created = _date(rng, created_lo, created_hi)
        last_used = None if users == 0 and rng.random() < 0.5 else _date(rng, idle_lo, min(idle_hi, (AS_OF - date.fromisoformat(created)).days))
        return {
            "id": f"item-{i:04d}",
            "name": name,
            "owner": owners[i],
            "team": team,
            "kind": kind,
            "created": created,
            "last_used": last_used,
            "invocations_30d": inv30,
            "distinct_users_30d": users,
            "invocations_7d": inv7,
            "invocations_prev_7d": prev7,
            "stars": stars,
            "versions": int(rng.integers(1, 4 if users <= 1 else 30)),
            "tags": tags,
            "risk_tier": str(rng.choice(["low", "low", "low", "medium", "high"])),
            "users_by_team": _split_by_team(rng, users, team),
        }

    i = 0
    for name, team, kind, users in HUBS:
        items.append(item(i, name, team, kind, users, float(rng.uniform(4, 12)), int(rng.integers(20, 200)),
                          120, 600, 0, 2, [team, str(rng.choice(ROLES))]))
        i += 1
    # middle band: ~8% of the catalog, 2..60 users
    n_mid = int(n * 0.08)
    for _ in range(n_mid):
        team = str(rng.choice(TEAMS))
        users = int(rng.integers(2, 61))
        name = f"{team} {rng.choice(MID_NOUNS)} {rng.choice(['', 'v2', 'beta', 'v3'])}".strip()
        rising = rng.random() < 0.08
        items.append(item(i, name, team, str(rng.choice(KINDS)), users, float(rng.uniform(2, 6)),
                          int(rng.integers(0, 6)), 30, 400, 0, 20, [team, str(rng.choice(ROLES))], rising))
        i += 1
    # the long tail: everything else has 0 or 1 user
    while i < n:
        team = str(rng.choice(TEAMS))
        r = rng.random()
        if r < 0.45:
            name = str(rng.choice(TAIL_NAMES))
        elif r < 0.65:
            name = f"hello world {int(rng.integers(0, 101))}"
        elif r < 0.8:
            name = f"{team} {rng.choice(MID_NOUNS)} test"
        else:
            name = f"{owners[i].split('@')[0]}'s {rng.choice(['app', 'agent', 'bot'])}"
        users = 1 if rng.random() < 0.35 else 0
        stars = 1 if rng.random() < 0.04 else 0
        items.append(item(i, name, team, str(rng.choice(KINDS)), users, float(rng.uniform(1, 3)), stars,
                          5, 500, 0, 30 if users else 365, [team]))
        i += 1

    featured = {
        "org": [it["id"] for it in items[:6]],
        "teams": {t: [it["id"] for it in items[:len(HUBS)] if it["team"] == t] for t in TEAMS},
    }
    for t in TEAMS:
        mids = [it["id"] for it in items[len(HUBS):len(HUBS) + n_mid] if it["team"] == t][:2]
        featured["teams"][t] = featured["teams"].get(t, []) + mids
    return items, featured


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default=str(Path(__file__).parent / "catalog.jsonl"))
    ap.add_argument("--featured-out", default=str(Path(__file__).parent / "featured.yaml"))
    a = ap.parse_args(argv)
    items, featured = make_catalog(a.n, a.seed)
    k = write_jsonl(a.out, items)
    Path(a.featured_out).write_text(
        "# Admin-pinned items: org-wide and per team (ids from catalog.jsonl).\n"
        + yaml.safe_dump(featured, sort_keys=False),
        encoding="utf-8",
    )
    print(f"wrote {k} items -> {a.out}; featured -> {a.featured_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
