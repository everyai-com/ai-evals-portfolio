"""Generate data/events.jsonl: ~5k deterministic usage events over 15 days for 3 tenants.

Planted patterns
  * user u-ocr-ops (agent ocr-batch, globex) runs ~2000 OCR pages at 06:00 every day:
    the "thousands of instances every morning" scheduled burst.
  * agent research-crawler (user u-research, acme) blows its $25 daily cap on the last day
    with 150 large opus calls between 09:00 and 12:00: a one-off spike.
  * 25 duplicate deliveries (same event_id) so `ingest` has something to dedupe.

    python projects/p18_usage_meter/data/make_events.py
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
START = datetime(2026, 10, 10, tzinfo=UTC)
DAYS = 15
TENANTS = {
    "acme": {"CC-100": ["platform", "support"], "CC-110": ["research"]},
    "globex": {"CC-200": ["sales-eng", "ops"]},
    "initech": {"CC-300": ["finance"]},
}
AGENTS = ["assistant", "code-reviewer", "ticket-triage", "sql-helper", "doc-summariser", "translation"]
MODELS = ["claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5-5"]


def ts(day: int, hour: float) -> str:
    return (START + timedelta(days=day, hours=hour)).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_events(seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    users = []
    for tenant, ccs in TENANTS.items():
        for cc, teams in ccs.items():
            for team in teams:
                for i in range(4 if tenant == "acme" else 5):
                    users.append((tenant, cc, team, f"u-{team}-{i + 1}"))
    events: list[dict] = []
    n = 0

    def emit(**kw):
        nonlocal n
        n += 1
        events.append({"event_id": f"ev-{n:05d}", **kw})

    for d in range(DAYS):
        for tenant, cc, team, user in users:
            for _ in range(int(rng.poisson(11.5))):
                hour = float(np.clip(rng.normal(13, 2.5), 8, 18.99))
                agent = AGENTS[int(rng.integers(len(AGENTS)))]
                r = rng.random()
                base = dict(ts=ts(d, hour), tenant=tenant, team=team, cost_centre=cc, user=user, agent_id=agent)
                if r < 0.65:
                    model = MODELS[int(rng.choice(3, p=[0.6, 0.3, 0.1]))]
                    emit(**base, kind="model_call", model=model, input_tokens=int(rng.integers(300, 4000)),
                         output_tokens=int(rng.integers(50, 900)), units=0, success=bool(rng.random() < 0.88))
                elif r < 0.85:
                    emit(**base, kind="tool_call", model=None, input_tokens=0, output_tokens=0,
                         units=int(rng.integers(1, 6)), success=None)
                elif r < 0.95:
                    emit(**base, kind="container_seconds", model=None, input_tokens=0, output_tokens=0,
                         units=int(rng.integers(5, 240)), success=None)
                else:
                    emit(**base, kind="ocr_page", model=None, input_tokens=0, output_tokens=0,
                         units=int(rng.integers(1, 12)), success=None)
        # scheduled burst: 25 OCR jobs x ~80 pages at 06:00-06:20 every day
        for _ in range(25):
            emit(ts=ts(d, 6 + rng.random() / 3), tenant="globex", team="ops", cost_centre="CC-200", user="u-ocr-ops",
                 agent_id="ocr-batch", kind="ocr_page", model=None, input_tokens=0, output_tokens=0,
                 units=int(rng.integers(70, 90)), success=None)
    # one-off spike on the last day: research-crawler blows its daily cap
    for _ in range(150):
        emit(ts=ts(DAYS - 1, 9 + rng.random() * 3), tenant="acme", team="research", cost_centre="CC-110",
             user="u-research-1", agent_id="research-crawler", kind="model_call", model="claude-opus-5-5",
             input_tokens=int(rng.integers(7000, 9000)), output_tokens=int(rng.integers(1200, 1800)), units=0,
             success=bool(rng.random() < 0.7))
    events.sort(key=lambda e: e["ts"])
    # duplicate deliveries
    dups = [dict(events[i]) for i in rng.choice(len(events), 25, replace=False)]
    out = events + dups
    rng.shuffle(out)
    return out


if __name__ == "__main__":
    rows = make_events()
    path = HERE / "events.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    print(f"wrote {len(rows)} rows ({len({r['event_id'] for r in rows})} unique) to {path}")
