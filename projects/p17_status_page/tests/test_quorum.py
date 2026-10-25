"""Quorum, debounce, recovery and uptime rules on hand-built timelines."""
from projects.p17_status_page.core import (
    DAY,
    FakeProber,
    Monitor,
    Probe,
    StatusStore,
    is_down,
    tick_status,
    uptime,
)

REGIONS = ("us-east", "eu-west", "ap-southeast")


def probe(name="api", quorum=2, debounce=2, regions=REGIONS):
    return Probe(name=name, kind="http", target="https://x.internal/h", regions=regions, home_region="us-east",
                 quorum=quorum, debounce_ticks=debounce)


def outage(lo, hi, regions, probes=("*",), error="HTTP 503"):
    return {"ticks": [lo, hi], "regions": list(regions), "probes": list(probes), "error": error}


def run(tmp_path, outages, ticks, p=None, interval=300.0):
    p = p or probe()
    store = StatusStore(tmp_path / "state")
    mon = Monitor([p], FakeProber({"latency_ms": {r: 100 for r in REGIONS}, "outages": outages}), store, interval)
    summaries = [mon.tick(1_000_000 + i * interval) for i in range(ticks)]
    return mon, store, summaries


def test_quorum_rule_pure():
    p = probe()
    assert not is_down(p, set())
    assert not is_down(p, {"ap-southeast"})                       # single far region alone
    assert tick_status(p, {"ap-southeast"}) == "degraded-remote"
    assert is_down(p, {"us-east"})                                # home region alone
    assert is_down(p, {"eu-west", "ap-southeast"})                # 2 of 3 remote
    assert tick_status(p, {"eu-west", "ap-southeast"}) == "down"


def test_single_far_region_never_opens_an_incident(tmp_path):
    # ap-southeast fails on every single tick for 20 ticks: degraded, never an incident.
    _, store, summaries = run(tmp_path, [outage(0, 19, ["ap-southeast"])], 20)
    assert all(s.statuses["api"] == "degraded-remote" for s in summaries)
    assert store.load_incidents() == []


def test_home_region_failure_opens_incident_after_debounce(tmp_path):
    _, store, summaries = run(tmp_path, [outage(3, 10, ["us-east"])], 8)
    assert summaries[3].statuses["api"] == "down" and summaries[3].opened == []   # first failed tick: debounced
    assert [i.probe for i in summaries[4].opened] == ["api"]                      # second consecutive: opens
    incs = store.load_incidents()
    assert len(incs) == 1 and incs[0].open and incs[0].regions == ["us-east"]


def test_two_of_three_regions_opens_incident(tmp_path):
    _, store, summaries = run(tmp_path, [outage(2, 6, ["eu-west", "ap-southeast"])], 6)
    assert summaries[2].statuses["api"] == "down"
    assert len(store.load_incidents()) == 1
    assert store.load_incidents()[0].regions == ["ap-southeast", "eu-west"]


def test_debounce_blip_shorter_than_k_never_opens(tmp_path):
    # single-tick home failures at ticks 1, 3, 5: each resets the streak -> no incident.
    outs = [outage(t, t, ["us-east"]) for t in (1, 3, 5)]
    _, store, summaries = run(tmp_path, outs, 7, probe(debounce=2))
    assert [s.statuses["api"] for s in summaries] == ["up", "down", "up", "down", "up", "down", "up"]
    assert store.load_incidents() == []


def test_debounce_persists_across_monitor_restart(tmp_path):
    p = probe(debounce=3)
    store = StatusStore(tmp_path / "s")
    prober = FakeProber({"outages": [outage(0, 5, ["us-east"])]})
    Monitor([p], prober, store).tick(0)
    Monitor([p], prober, store).tick(300)
    assert store.load_incidents() == []
    s = Monitor([p], prober, store).tick(600)          # third consecutive, new Monitor object each time
    assert len(s.opened) == 1


def test_recovery_closes_incident_and_uptime_math(tmp_path):
    # 10 ticks, down on ticks 2-4 (3 ticks) -> uptime 7/10; incident opens at tick 3, closes at tick 5.
    mon, store, summaries = run(tmp_path, [outage(2, 4, ["us-east"])], 10)
    opened = [i for s in summaries for i in s.opened]
    closed = [i for s in summaries for i in s.closed]
    assert len(opened) == 1 and len(closed) == 1
    assert opened[0].opened_at == 1_000_000 + 3 * 300 and closed[0].closed_at == 1_000_000 + 5 * 300
    incs = store.load_incidents()
    assert len(incs) == 1 and not incs[0].open
    p = mon.probes[0]
    results = store.load_results()
    assert uptime(p, results, 0, 1e12) == 7 / 10
    # uptime uses the raw quorum rule: a single far-region failure does not reduce it.
    _, store2, _ = run(tmp_path / "b", [outage(0, 9, ["ap-southeast"])], 10)
    assert uptime(p, store2.load_results(), 0, 1e12) == 1.0
    assert uptime(p, [], 0, 1e12) is None


def test_uptime_windows_on_hand_built_timeline(tmp_path):
    # 48 ticks at 1h spacing: down for 6 ticks within the last 24h, none before -> 24h=18/24, 48h=42/48.
    p = probe()
    store = StatusStore(tmp_path / "s")
    mon = Monitor([p], FakeProber({"outages": [outage(30, 35, ["us-east"])]}), store, 3600)
    t0 = 1_000_000
    for i in range(48):
        mon.tick(t0 + i * 3600)
    now = t0 + 47 * 3600
    res = store.load_results()
    assert uptime(p, res, now - DAY, now) == 18 / 24
    assert uptime(p, res, now - 2 * DAY, now) == 42 / 48
