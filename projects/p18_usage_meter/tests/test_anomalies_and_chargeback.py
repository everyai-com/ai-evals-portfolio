"""Scheduled burst vs one-off spike on a hand-built timeline; chargeback totals and CSV on the demo."""
import csv
import io
import re
from pathlib import Path

import pytest

from evalkit.io import read_jsonl
from projects.p18_usage_meter.cli import main
from projects.p18_usage_meter.core import (
    Limits,
    Meter,
    UsageEvent,
    chargeback,
    chargeback_csv_rows,
    detect_anomalies,
    invoice,
    load_prices,
    render_chargeback,
    render_invoice,
    write_csv,
)

HERE = Path(__file__).resolve().parent.parent
PRICES = {"tool_call": 0.002, "container_seconds": 0.00004, "ocr_page": 0.015}


def build_timeline():
    """15 days. Three normal users spend ~$0.05/h 09-17. u-burst runs OCR at 06:00 every day.
    u-once is normal but spends $8 at 15:00 on the last day only."""
    evs, n = [], 0

    def add(day, hour, user, agent, kind="model_call", units=0, it=1500, ot=300, tenant="acme", team="t1", cc="CC-1"):
        nonlocal n
        n += 1
        evs.append(UsageEvent(f"e{n}", f"2026-10-{day:02d}T{hour:02d}:10:00Z", tenant, team, cc, user, agent, kind,
                              "claude-sonnet-5-5" if kind == "model_call" else None, it if kind == "model_call" else 0,
                              ot if kind == "model_call" else 0, units, None, True if kind == "model_call" else None))

    for day in range(10, 25):
        for user in ("u-a", "u-b", "u-c", "u-once"):
            for hour in range(9, 17):
                for _ in range(3):
                    add(day, hour, user, "assistant")                          # 3 x $0.009 = $0.027/h
        for _ in range(20):
            add(day, 6, "u-burst", "ocr-batch", kind="ocr_page", units=100, team="ops", cc="CC-2")   # $30 at 06:00
    for _ in range(40):
        add(24, 15, "u-once", "assistant", it=8000, ot=1000)                   # 40 x $0.039 = $1.56 at 15:00
    return evs


def test_scheduled_burst_and_one_off_spike_labels():
    m = Meter(PRICES)
    m.ingest(build_timeline())
    an = detect_anomalies(m, Limits().anomaly, "2026-10-24")
    by = {(a.entity_type, a.entity, a.hour): a for a in an}
    burst = by[("user", "u-burst", 6)]
    assert burst.label == "scheduled_burst" and burst.recurrence_k == 14 and burst.wilson_lo >= 0.5
    assert by[("agent_id", "ocr-batch", 6)].label == "scheduled_burst"
    once = by[("user", "u-once", 15)]
    assert once.label == "spike" and once.recurrence_k == 0
    assert {a.entity for a in an if a.entity_type == "user"} == {"u-burst", "u-once"}   # normal users are silent
    assert detect_anomalies(Meter(PRICES)) == []
    # too little history -> no verdict
    short = Meter(PRICES)
    short.ingest([e for e in build_timeline() if e.day >= "2026-10-24"])
    assert detect_anomalies(short, Limits().anomaly, "2026-10-24") == []


def test_chargeback_totals_equal_event_sum_and_csv_rows():
    m = Meter(PRICES)
    m.ingest(build_timeline())
    limits = Limits(tenants={"default": {"monthly_budget_usd": 100.0}})
    rep = chargeback(m, limits, detect_anomalies(m, Limits().anomaly))
    assert rep["total"] == pytest.approx(m.total())
    assert sum(cc["spend"] for cc in rep["cost_centres"].values()) == pytest.approx(m.total())
    for cc in rep["cost_centres"].values():
        assert sum(t["spend"] for t in cc["teams"].values()) == pytest.approx(cc["spend"])
        for t in cc["teams"].values():
            assert sum(t["by_kind"].values()) == pytest.approx(t["spend"])
    assert sum(cc["share"] for cc in rep["cost_centres"].values()) == pytest.approx(1.0)
    t1 = rep["cost_centres"]["CC-1"]["teams"]["t1"]
    assert t1["cost_per_success"] == pytest.approx(t1["spend"] / t1["successes"])
    assert rep["cost_centres"]["CC-2"]["teams"]["ops"]["cost_per_success"] is None   # no success flags on OCR
    assert any(a["label"] == "scheduled_burst" for a in rep["cost_centres"]["CC-2"]["teams"]["ops"]["anomalies"])
    assert rep["budgets"]["acme 2026-10"]["status"] == "over"
    md = render_chargeback(rep)
    assert "scheduled_burst:u-burst@06h" in md and "Tenant budgets" in md
    rows = chargeback_csv_rows(m)
    combos = {(e.cost_centre, e.team, e.kind, e.model) for e in m.events}
    assert len(rows) == len(combos) == 2
    assert sum(r["spend_usd"] for r in rows) == pytest.approx(m.total(), abs=1e-4)
    parsed = list(csv.DictReader(io.StringIO(write_csv(rows))))
    assert len(parsed) == len(rows)
    inv = invoice(m, limits, "acme", "2026-10")
    assert inv["subtotal"] == pytest.approx(m.total()) and "Subtotal" in render_invoice(inv)


def test_demo_events_end_to_end(tmp_path, capsys):
    state = str(tmp_path / "state")
    rc = main(["ingest", "--events", str(HERE / "data" / "events.jsonl"), "--state", state])
    out = capsys.readouterr().out
    assert rc == 0 and "25 duplicates skipped" in out
    rows = read_jsonl(HERE / "data" / "events.jsonl")
    assert len({r["event_id"] for r in rows}) == len(Meter.load(state, load_prices(HERE / "data" / "prices.yaml")).events)
    # the planted patterns
    rc = main(["anomalies", "--state", state])
    out = capsys.readouterr().out
    assert rc == 1
    assert re.search(r"scheduled_burst \| user=u-ocr-ops \| 2026-10-24 \| 06:00", out)
    assert re.search(r"spike \| agent_id=research-crawler", out)
    # the cap blower is blocked for the rest of the day, another agent is not
    rc = main(["decide", "--state", state, "--tenant", "acme", "--user", "u-research-1", "--agent", "research-crawler",
               "--tokens", "2000", "--now", "2026-10-24T12:30:00Z"])
    out = capsys.readouterr().out
    assert rc == 1 and "agent_daily_cap" in out
    rc = main(["decide", "--state", state, "--tenant", "globex", "--user", "u-ops-1", "--agent", "assistant",
               "--tokens", "2000", "--now", "2026-10-24T12:30:00Z"])
    out = capsys.readouterr().out
    assert rc == 0 and "budget_80pct" in out                      # globex is at 94% of its $500 budget
    rc = main(["decide", "--state", state, "--tenant", "globex", "--user", "u-ops-1", "--agent", "assistant",
               "--tokens", "2000", "--now", "2026-10-24T12:31:00Z"])
    assert rc == 0 and "budget_80pct" not in capsys.readouterr().out   # fired once, persisted in enforcer.json
    csv_path = tmp_path / "out.csv"
    rc = main(["report", "--state", state, "--csv", str(csv_path), "--json", str(tmp_path / "r.json")])
    out = capsys.readouterr().out
    assert rc == 0 and "# Chargeback report" in out and csv_path.exists()
    n_rows = len(list(csv.DictReader(csv_path.open())))
    m = Meter.load(state, load_prices(HERE / "data" / "prices.yaml"))
    assert n_rows == len({(e.cost_centre, e.team, e.kind, e.model) for e in m.events})
    assert main(["invoice", "--state", state, "--tenant", "initech"]) == 0
    assert main(["nope"]) == 2
