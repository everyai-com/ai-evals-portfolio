"""Staleness self-check, HTML rendering, scheduler with injected sleep, CLI on the committed scenario."""
import io
import re
import urllib.error
from pathlib import Path

from projects.p17_status_page.cli import main
from projects.p17_status_page.core import (
    FakeProber,
    Monitor,
    Probe,
    Scheduler,
    SimClock,
    StatusStore,
    UrllibProber,
    compute_state,
    load_probes,
    render_html,
    render_json,
)

HERE = Path(__file__).resolve().parent.parent


def test_staleness_self_check(tmp_path):
    probes, _ = load_probes(HERE / "data" / "probes.yaml")
    store = StatusStore(tmp_path / "s")
    mon = Monitor(probes, FakeProber(), store, interval_s=300)
    mon.tick(1000.0)
    fresh = compute_state(probes, store, 1000.0 + 599, 300)
    assert not fresh["stale"] and fresh["overall"] == "all systems operational"
    stale = compute_state(probes, store, 1000.0 + 601, 300)
    assert stale["stale"] and stale["overall"] == "monitor stale"
    html = render_html(stale)
    assert "Monitor stale" in html
    assert "monitor stale" in render_json(stale)
    empty = compute_state(probes, StatusStore(tmp_path / "empty"), 5000.0, 300)
    assert empty["stale"] and empty["last_tick"] is None


def test_html_one_row_per_probe_and_no_external_urls(tmp_path):
    probes, _ = load_probes(HERE / "data" / "probes.yaml")
    store = StatusStore(tmp_path / "s")
    prober = FakeProber.from_yaml(HERE / "data" / "scenario.yaml")
    mon = Monitor(probes, prober, store, 300)
    for i in range(12):
        mon.tick(1_700_000_000 + i * 300)
    state = compute_state(probes, store, 1_700_000_000 + 11 * 300, 300)
    html = render_html(state)
    assert html.count('<tr class="probe"') == len(probes) == 4
    for p in probes:
        assert f'data-probe="{p.name}"' in html
    assert not re.search(r'(src|href)\s*=\s*"\s*(https?:)?//', html)
    assert "http://" not in html and "https://" not in html      # not even the probe targets leak
    assert "<svg" in html and "--bg" in html and "prefers-color-scheme" in html
    assert "Incident history" in html and "inc-" in html         # the scenario's incident is closed by tick 11


def test_scheduler_uses_injected_sleep_and_clock(tmp_path):
    probes, _ = load_probes(HERE / "data" / "probes.yaml")
    clock = SimClock(10_000.0)
    mon = Monitor(probes, FakeProber(), StatusStore(tmp_path / "s"), 300)
    seen = []
    n = Scheduler(mon, clock=clock, sleep=clock.sleep).run(300, ticks=5, on_tick=lambda s: seen.append(s.ts))
    assert n == 5
    assert clock.sleeps == [300] * 4                      # sleeps between ticks only, never after the last
    assert seen == [10_000 + i * 300 for i in range(5)]
    assert mon.store.load_meta()["ticks"] == 5


def test_urllib_prober_uses_injected_opener():
    calls = []

    class Resp:
        status = 200

        def __init__(self, body):
            self.body = body

        def read(self, n=-1):
            return self.body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout):
        calls.append((req.full_url, timeout))
        if "bad" in req.full_url:
            raise urllib.error.HTTPError(req.full_url, 503, "down", {}, io.BytesIO(b""))
        return Resp(b'{"ok": true}')

    p = Probe("g", "http", "https://x.internal/healthz", ("us-east",), "us-east", body_regex=r'"ok"\s*:\s*true')
    r = UrllibProber(opener=opener).probe(p, "us-east", 1.0)
    assert r.ok and r.status == 200 and calls == [("https://x.internal/healthz", 5.0)]
    bad = Probe("b", "http", "https://x.internal/bad", ("us-east",), "us-east")
    r2 = UrllibProber(opener=opener).probe(bad, "us-east", 1.0)
    assert not r2.ok and r2.status == 503 and "503" in r2.error
    fn = Probe("f", "function", "projects.p17_status_page.core:sandbox_selfcheck", ("us-east",), "us-east")
    assert UrllibProber().probe(fn, "us-east", 1.0).ok


def test_cli_demo_scenario_end_to_end(tmp_path, capsys):
    state = str(tmp_path / "state")
    rc = main(["run", "--fake", "--ticks", "12", "--interval", "300", "--state", state, "--reset"])
    out = capsys.readouterr().out
    assert rc == 0                                   # the incident has closed by the last tick
    assert "OPENED inc-" in out and "CLOSED inc-" in out
    assert out.count("agent-api=down") == 3           # ticks 8, 9, 10
    assert "degraded-remote" in out
    assert "open incidents: 0; history: 1" in out
    rc = main(["render", "--state", state, "--out", str(tmp_path / "status.html"), "--json", str(tmp_path / "s.json"),
               "--now", "2026-10-24T00:55:00Z"])
    assert rc == 0 and (tmp_path / "status.html").exists() and (tmp_path / "s.json").exists()
    rc = main(["tick", "--state", state, "--fake-scenario", str(HERE / "data" / "scenario.yaml"),
               "--now", "2026-10-24T01:00:00Z"])
    assert rc == 0
    assert main(["bogus"]) == 2
