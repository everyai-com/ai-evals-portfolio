"""Dedup, pricing, daily caps with clock reset, token bucket refill, budget alerts firing once each."""
from datetime import UTC, datetime, timedelta

import pytest

from evalkit.llm import estimate_cost
from projects.p18_usage_meter.core import (
    Enforcer,
    Limits,
    Meter,
    Request,
    TokenBucket,
    UsageEvent,
    price_event,
)

PRICES = {"tool_call": 0.002, "container_seconds": 0.00004, "ocr_page": 0.015}
T0 = datetime(2026, 10, 24, 9, 0, tzinfo=UTC)


def ev(i, kind="model_call", model="claude-sonnet-5-5", it=1000, ot=200, units=0, ts="2026-10-24T09:00:00Z", **kw):
    base = dict(event_id=f"e{i}", ts=ts, tenant="acme", team="t", cost_centre="cc", user="u1", agent_id="a1",
                kind=kind, model=model, input_tokens=it, output_tokens=ot, units=units)
    base.update(kw)
    return UsageEvent(**base)


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t += timedelta(**kw)


def test_dedup_by_event_id():
    m = Meter(PRICES)
    r = m.ingest([ev(1), ev(2), ev(1), {"event_id": "e2", "ts": "2026-10-24T09:00:00Z", "tenant": "acme", "team": "t",
                                        "cost_centre": "cc", "user": "u1", "agent_id": "a1", "kind": "model_call"}])
    assert (r.added, r.duplicates) == (2, 2)
    assert m.ingest([ev(1)]).duplicates == 1 and len(m.events) == 2
    assert m.ingest([{"event_id": "bad", "kind": "nope"}]).rejected == 1


def test_cost_computation_by_kind():
    m = Meter(PRICES)
    m.ingest([ev(1, it=1000, ot=200), ev(2, kind="ocr_page", model=None, it=0, ot=0, units=80),
              ev(3, kind="tool_call", model=None, it=0, ot=0, units=3),
              ev(4, kind="container_seconds", model=None, it=0, ot=0, units=120),
              ev(5, cost_usd=9.0)])
    costs = {e.event_id: e.cost_usd for e in m.events}
    assert costs["e1"] == pytest.approx(estimate_cost("claude-sonnet-5-5", 1000, 200)) == pytest.approx((1000 * 3 + 200 * 15) / 1e6)
    assert costs["e2"] == pytest.approx(80 * 0.015)
    assert costs["e3"] == pytest.approx(0.006)
    assert costs["e4"] == pytest.approx(0.0048)
    assert costs["e5"] == 9.0                                   # a cost supplied upstream is kept
    with pytest.raises(KeyError):
        price_event(ev(9, kind="ocr_page", model=None, units=1), {})
    assert m.total() == pytest.approx(sum(costs.values()))


def test_daily_cap_blocks_next_request_and_resets_next_day():
    limits = Limits(agents={"default": {"daily_cap_usd": 1.0}}, users={"default": {"daily_cap_usd": 100.0}},
                    burst=1000, requests_per_minute=60_000)
    clock = Clock()
    enf = Enforcer(Meter(PRICES), limits, clock)
    req = Request("acme", "u1", "a1", input_tokens=100_000 // 10, output_tokens=0, model="claude-sonnet-5-5")  # $0.03
    n = 0
    while True:
        d = enf.decide(req)
        if not d.allow:
            break
        enf.record(req)
        n += 1
        assert n < 100
    assert n == 33 and d.reason.startswith("agent_daily_cap")
    assert d.retry_after_s == pytest.approx((T0 + timedelta(days=1)).replace(hour=0).timestamp() - T0.timestamp())
    assert d.remaining["agent_daily_usd"] == pytest.approx(1.0 - 33 * 0.03)
    clock.advance(seconds=d.retry_after_s + 1)                 # just past midnight UTC
    d2 = enf.decide(req)
    assert d2.allow and d2.remaining["agent_daily_usd"] == pytest.approx(1.0)


def test_per_request_and_user_caps():
    limits = Limits(max_tokens=1000, max_units=10, users={"default": {"daily_cap_usd": 0.01}}, agents={})
    enf = Enforcer(Meter(PRICES), limits, Clock())
    assert not enf.decide(Request("acme", "u1", "a1", input_tokens=900, output_tokens=200)).allow
    assert not enf.decide(Request("acme", "u1", "a1", kind="ocr_page", units=11)).allow
    d = enf.decide(Request("acme", "u1", "a1", kind="ocr_page", units=1))      # $0.015 > user cap $0.01
    assert not d.allow and d.reason.startswith("user_daily_cap")
    assert enf.decide(Request("acme", "u1", "a1", kind="tool_call", units=1)).allow


def test_token_bucket_refill():
    b = TokenBucket(capacity=3, rate=1.0)                       # 3 burst, 1 token / s
    assert [b.take(0.0)[0] for _ in range(3)] == [True, True, True]
    ok, retry = b.take(0.0)
    assert not ok and retry == pytest.approx(1.0)
    ok, _ = b.take(0.5)
    assert not ok
    assert b.take(1.0)[0]                                        # refilled one token
    assert not b.take(1.0)[0]
    assert all(b.take(100.0)[0] for _ in range(3)) and not b.take(100.0)[0]   # capped at capacity
    # through the enforcer, per agent, with the injected clock
    limits = Limits(burst=2, requests_per_minute=60, agents={}, users={})
    clock = Clock()
    enf = Enforcer(Meter(PRICES), limits, clock)
    req = Request("acme", "u1", "a1", input_tokens=10)
    assert enf.decide(req).allow and enf.decide(req).allow
    d = enf.decide(req)
    assert not d.allow and d.reason.startswith("rate_limit") and d.retry_after_s == pytest.approx(1.0)
    assert enf.decide(Request("acme", "u1", "other-agent", input_tokens=10)).allow   # separate bucket
    clock.advance(seconds=1)
    assert enf.decide(req).allow


def test_budget_alerts_fire_once_each_at_80_and_100():
    limits = Limits(tenants={"default": {"monthly_budget_usd": 1.0}}, agents={}, users={}, burst=10_000,
                    requests_per_minute=600_000)
    clock = Clock()
    enf = Enforcer(Meter(PRICES), limits, clock)
    req = Request("acme", "u1", "a1", kind="ocr_page", units=10)   # $0.15 per request
    fired = []
    for _ in range(10):
        d = enf.decide(req)
        assert d.allow                                              # soft budget never blocks
        enf.record(req)
        fired.extend(d.warnings)
    assert [w.split(":")[0] for w in fired] == ["budget_80pct", "budget_100pct"]
    assert [a.threshold for a in enf.alerts] == [0.8, 1.0]
    assert enf.alerts[0].spent == pytest.approx(0.9) and enf.alerts[1].spent == pytest.approx(1.05)
    # state round-trips so a restarted enforcer does not re-fire
    enf2 = Enforcer(enf.meter, limits, clock)
    enf2.load_state(enf.to_state())
    assert enf2.decide(req).warnings == []
    clock.advance(days=31)                                          # new month: alerts can fire again
    assert enf2.decide(req).warnings == []                          # ...but spend restarted at 0, so nothing yet
