"""Independent status page engine: probes, regional quorum, debounce, incidents, uptime, staleness.

Rules (so the README claims are checkable)
------------------------------------------
Regional quorum   A probe is DOWN on a tick iff it fails in its `home_region` OR it fails in at
                  least `quorum` regions. Fewer remote-only failures are "degraded-remote": shown
                  on the page, never an incident. This is the fix for the far-region false alarm.
Debounce          An incident opens only after `debounce_ticks` consecutive DOWN ticks; it closes
                  on the first tick where the probe is not DOWN. Streaks are persisted with the
                  state so the rule survives process restarts.
Uptime            uptime(window) = (# ticks in window where probe not DOWN) / (# ticks in window).
                  DOWN uses the raw quorum rule per tick (not the debounced incident), so a
                  single bad tick costs its share of uptime even if it never became an incident.
Latency           p50 / p95 per (probe, region) over successful results in the last 24 h, via
                  numpy.percentile with linear interpolation.
Staleness         The page is "monitor stale" if now - last_tick > 2 * interval_s: the monitor
                  monitors itself, because a status page that stops updating looks like "all green".
"""
from __future__ import annotations

import importlib
import json
import math
import re
import socket
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from evalkit.io import append_jsonl, read_jsonl, write_jsonl

DAY = 86_400.0
WINDOWS = {"24h": DAY, "7d": 7 * DAY, "30d": 30 * DAY}


# --------------------------------------------------------------------------- definitions

@dataclass(frozen=True)
class Probe:
    name: str
    kind: str                       # http | tcp | function
    target: str
    regions: tuple[str, ...]
    home_region: str
    timeout_s: float = 5.0
    expected_status: int = 200
    body_regex: str | None = None
    quorum: int = 2
    debounce_ticks: int = 2

    def __post_init__(self) -> None:
        if self.kind not in {"http", "tcp", "function"}:
            raise ValueError(f"probe {self.name}: unknown kind {self.kind!r}")
        if self.home_region not in self.regions:
            raise ValueError(f"probe {self.name}: home_region {self.home_region!r} not in regions")
        if self.quorum < 1:
            raise ValueError(f"probe {self.name}: quorum must be >= 1")


@dataclass
class ProbeResult:
    probe: str
    region: str
    ts: float
    ok: bool
    latency_ms: float
    status: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Incident:
    id: str
    probe: str
    opened_at: float
    regions: list[str]
    cause: str
    closed_at: float | None = None

    @property
    def open(self) -> bool:
        return self.closed_at is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_probes(path: str | Path) -> tuple[list[Probe], dict[str, Any]]:
    """Load `probes.yaml`. Top-level regions/home_region/quorum/debounce_ticks are defaults
    each probe may override. Returns (probes, settings) where settings carries interval_s."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    regions = tuple(cfg.get("regions", ["us-east"]))
    probes = []
    for p in cfg.get("probes", []):
        pr = tuple(p.get("regions", regions))
        probes.append(Probe(
            name=p["name"], kind=p["kind"], target=p["target"], regions=pr,
            home_region=p.get("home_region", cfg.get("home_region", pr[0])),
            timeout_s=float(p.get("timeout_s", 5.0)),
            expected_status=int(p.get("expected_status", 200)),
            body_regex=p.get("body_regex"),
            quorum=int(p.get("quorum", cfg.get("quorum", 2))),
            debounce_ticks=int(p.get("debounce_ticks", cfg.get("debounce_ticks", 2))),
        ))
    settings = {"interval_s": float(cfg.get("interval_s", 300)), "regions": list(regions)}
    return probes, settings


# --------------------------------------------------------------------------- probers

class Prober(Protocol):
    def probe(self, probe: Probe, region: str, now: float) -> ProbeResult: ...


def sandbox_selfcheck() -> bool:
    """Default `function` probe target used by the committed fixture: always healthy."""
    return True


def _resolve_callable(target: str) -> Callable[[], Any]:
    mod, _, attr = target.partition(":")
    if not attr:
        raise ValueError(f"function target must be 'module:callable', got {target!r}")
    return getattr(importlib.import_module(mod), attr)


class UrllibProber:
    """Real probes: http via urllib, tcp via socket.create_connection, function via import.

    Regions are advisory here (one process cannot be in three regions); the region label is
    recorded so a deployment that runs one worker per region shares the same code path.
    `opener` is injectable so the http path is testable without a network."""

    def __init__(self, opener: Callable[..., Any] | None = None,
                 connect: Callable[..., Any] | None = None):
        self.opener = opener or urllib.request.urlopen
        self.connect = connect or socket.create_connection

    def probe(self, probe: Probe, region: str, now: float) -> ProbeResult:
        t0 = time.perf_counter()
        ok, status, error = False, None, None
        try:
            if probe.kind == "http":
                req = urllib.request.Request(probe.target, headers={"User-Agent": "p17-status-probe"})
                try:
                    with self.opener(req, timeout=probe.timeout_s) as resp:
                        status = int(getattr(resp, "status", 200))
                        body = resp.read(65536).decode("utf-8", "replace")
                except urllib.error.HTTPError as e:
                    status, body = e.code, ""
                ok = status == probe.expected_status
                if ok and probe.body_regex:
                    ok = re.search(probe.body_regex, body) is not None
                    if not ok:
                        error = "body regex did not match"
                elif not ok:
                    error = f"HTTP {status} != {probe.expected_status}"
            elif probe.kind == "tcp":
                host, _, port = probe.target.rpartition(":")
                with self.connect((host, int(port)), timeout=probe.timeout_s):
                    ok = True
            else:
                ok = bool(_resolve_callable(probe.target)())
                if not ok:
                    error = "function returned falsy"
        except Exception as e:  # noqa: BLE001 - any failure is a failed probe, never a crash
            ok, error = False, f"{type(e).__name__}: {e}"[:200]
        return ProbeResult(probe.name, region, now, ok, (time.perf_counter() - t0) * 1000, status, error)


class FakeProber:
    """Scripted prober for tests and the demo.

    `scenario` = {latency_ms: {region: ms}, jitter_ms, seed,
                  outages: [{ticks: [from, to], regions: [...], probes: [...] | ["*"], error}]}.
    Tick n for a (probe, region) pair is the n-th time that pair is probed (0-based), which is
    exactly the Monitor tick index because every tick probes every pair once."""

    def __init__(self, scenario: dict[str, Any] | None = None):
        sc = scenario or {}
        self.latency = {k: float(v) for k, v in (sc.get("latency_ms") or {}).items()}
        self.jitter = float(sc.get("jitter_ms", 0.0))
        self.outages = list(sc.get("outages") or [])
        self.rng = np.random.default_rng(int(sc.get("seed", 0)))
        self.calls: dict[tuple[str, str], int] = {}

    @classmethod
    def from_yaml(cls, path: str | Path) -> FakeProber:
        return cls(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    def outage_for(self, probe: str, region: str, tick: int) -> dict[str, Any] | None:
        for o in self.outages:
            lo, hi = o.get("ticks", [0, 0])
            probes = o.get("probes", ["*"])
            if lo <= tick <= hi and region in o.get("regions", []) and ("*" in probes or probe in probes):
                return o
        return None

    def probe(self, probe: Probe, region: str, now: float) -> ProbeResult:
        key = (probe.name, region)
        tick = self.calls.get(key, 0)
        self.calls[key] = tick + 1
        base = self.latency.get(region, 100.0)
        latency = max(1.0, base + (self.rng.normal(0, self.jitter) if self.jitter else 0.0))
        o = self.outage_for(probe.name, region, tick)
        if o:
            err = str(o.get("error", "failed"))
            status = int(err.split()[1]) if err.startswith("HTTP ") else None
            return ProbeResult(probe.name, region, now, False, probe.timeout_s * 1000 if status is None else latency,
                               status, err)
        return ProbeResult(probe.name, region, now, True, float(latency), 200 if probe.kind == "http" else None)


# --------------------------------------------------------------------------- store

class StatusStore:
    """JSONL state: results.jsonl (append-only), incidents.jsonl (rewritten), meta.json."""

    def __init__(self, state_dir: str | Path):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.dir / "results.jsonl"
        self.incidents_path = self.dir / "incidents.jsonl"
        self.meta_path = self.dir / "meta.json"

    def append_results(self, results: Iterable[ProbeResult]) -> int:
        return append_jsonl(self.results_path, (r.to_dict() for r in results))

    def load_results(self, since: float | None = None) -> list[ProbeResult]:
        if not self.results_path.exists():
            return []
        rows = [ProbeResult(**r) for r in read_jsonl(self.results_path)]
        return [r for r in rows if since is None or r.ts >= since]

    def load_incidents(self) -> list[Incident]:
        if not self.incidents_path.exists():
            return []
        return [Incident(**r) for r in read_jsonl(self.incidents_path)]

    def save_incidents(self, incidents: Sequence[Incident]) -> None:
        write_jsonl(self.incidents_path, (i.to_dict() for i in incidents))

    def load_meta(self) -> dict[str, Any]:
        if not self.meta_path.exists():
            return {"last_tick": None, "ticks": 0, "streaks": {}}
        return json.loads(self.meta_path.read_text(encoding="utf-8"))

    def save_meta(self, meta: dict[str, Any]) -> None:
        self.meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    def clear(self) -> None:
        for p in (self.results_path, self.incidents_path, self.meta_path):
            if p.exists():
                p.unlink()


# --------------------------------------------------------------------------- rules

def is_down(probe: Probe, failed_regions: set[str]) -> bool:
    """Regional quorum rule: DOWN iff home region failed or >= quorum regions failed."""
    return probe.home_region in failed_regions or len(failed_regions) >= probe.quorum


def tick_status(probe: Probe, failed_regions: set[str]) -> str:
    if is_down(probe, failed_regions):
        return "down"
    return "degraded-remote" if failed_regions else "up"


@dataclass
class TickSummary:
    ts: float
    results: list[ProbeResult]
    statuses: dict[str, str]                 # probe -> up | degraded-remote | down
    opened: list[Incident] = field(default_factory=list)
    closed: list[Incident] = field(default_factory=list)


class Monitor:
    def __init__(self, probes: Sequence[Probe], prober: Prober, store: StatusStore,
                 interval_s: float = 300.0):
        self.probes = list(probes)
        self.prober = prober
        self.store = store
        self.interval_s = float(interval_s)

    def tick(self, now: float) -> TickSummary:
        """Run every probe in every region once, apply quorum + debounce, update incidents."""
        meta = self.store.load_meta()
        streaks: dict[str, int] = dict(meta.get("streaks", {}))
        incidents = self.store.load_incidents()
        open_by_probe = {i.probe: i for i in incidents if i.open}
        results: list[ProbeResult] = []
        statuses: dict[str, str] = {}
        opened: list[Incident] = []
        closed: list[Incident] = []
        for probe in self.probes:
            rs = [self.prober.probe(probe, region, now) for region in probe.regions]
            results.extend(rs)
            failed = {r.region for r in rs if not r.ok}
            status = tick_status(probe, failed)
            statuses[probe.name] = status
            if status == "down":
                streaks[probe.name] = streaks.get(probe.name, 0) + 1
                if streaks[probe.name] >= probe.debounce_ticks and probe.name not in open_by_probe:
                    cause = "; ".join(f"{r.region}: {r.error or 'failed'}" for r in rs if not r.ok)
                    inc = Incident(id=f"inc-{int(now)}-{probe.name}", probe=probe.name, opened_at=now,
                                   regions=sorted(failed), cause=cause)
                    incidents.append(inc)
                    open_by_probe[probe.name] = inc
                    opened.append(inc)
            else:
                streaks[probe.name] = 0
                if probe.name in open_by_probe:
                    inc = open_by_probe.pop(probe.name)
                    inc.closed_at = now
                    closed.append(inc)
        self.store.append_results(results)
        self.store.save_incidents(incidents)
        meta.update({"last_tick": now, "ticks": int(meta.get("ticks", 0)) + 1, "streaks": streaks,
                     "interval_s": self.interval_s})
        self.store.save_meta(meta)
        return TickSummary(now, results, statuses, opened, closed)


class Scheduler:
    """Cron stand-in: tick, sleep interval, repeat. clock/sleep are injected (tests never sleep)."""

    def __init__(self, monitor: Monitor, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep):
        self.monitor = monitor
        self.clock = clock
        self.sleep = sleep

    def run(self, interval_s: float, ticks: int | None = None,
            on_tick: Callable[[TickSummary], None] | None = None) -> int:
        n = 0
        while ticks is None or n < ticks:
            summary = self.monitor.tick(self.clock())
            n += 1
            if on_tick:
                on_tick(summary)
            if ticks is not None and n >= ticks:
                break
            self.sleep(interval_s)
        return n


class SimClock:
    """Deterministic clock whose sleep advances time instead of waiting."""

    def __init__(self, start: float):
        self.now = float(start)
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.now += s


# --------------------------------------------------------------------------- state computation

def iso(ts: float | None) -> str:
    if ts is None:
        return "never"
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def uptime(probe: Probe, results: Sequence[ProbeResult], since: float, until: float) -> float | None:
    """Share of ticks in (since, until] where the probe was not DOWN (quorum rule). None = no data.

    The window is open on the left so "last 24 h" at a 1 h cadence holds exactly 24 ticks."""
    by_tick: dict[float, set[str]] = {}
    for r in results:
        if r.probe == probe.name and since < r.ts <= until:
            by_tick.setdefault(r.ts, set())
            if not r.ok:
                by_tick[r.ts].add(r.region)
    if not by_tick:
        return None
    good = sum(not is_down(probe, failed) for failed in by_tick.values())
    return good / len(by_tick)


def latency_stats(results: Sequence[ProbeResult], probe: str, region: str, since: float) -> dict[str, float | None]:
    vals = [r.latency_ms for r in results if r.probe == probe and r.region == region and r.ok and r.ts >= since]
    if not vals:
        return {"p50": None, "p95": None, "n": 0}
    arr = np.asarray(vals, float)
    return {"p50": float(np.percentile(arr, 50)), "p95": float(np.percentile(arr, 95)), "n": len(vals)}


def compute_state(probes: Sequence[Probe], store: StatusStore, now: float,
                  interval_s: float | None = None) -> dict[str, Any]:
    """Everything the page needs, as plain JSON-able data."""
    meta = store.load_meta()
    interval = float(interval_s or meta.get("interval_s") or 300.0)
    last_tick = meta.get("last_tick")
    stale = last_tick is None or (now - last_tick) > 2 * interval
    results = store.load_results(since=now - 31 * DAY)
    incidents = store.load_incidents()
    open_inc = [i for i in incidents if i.open]
    rows = []
    for probe in probes:
        latest_ts = max((r.ts for r in results if r.probe == probe.name), default=None)
        latest = {r.region: r for r in results if r.probe == probe.name and r.ts == latest_ts}
        failed = {reg for reg, r in latest.items() if not r.ok}
        status = tick_status(probe, failed) if latest else "unknown"
        if probe.name in {i.probe for i in open_inc}:
            status = "down"
        days = []
        day0 = math.floor(now / DAY) * DAY
        for d in range(29, -1, -1):
            start = day0 - d * DAY
            days.append({"date": datetime.fromtimestamp(start, tz=UTC).strftime("%Y-%m-%d"),
                         "uptime": uptime(probe, results, start - 1e-6, start + DAY - 1e-6)})  # [00:00, 24:00)
        rows.append({
            "name": probe.name, "kind": probe.kind, "status": status, "home_region": probe.home_region,
            "quorum": probe.quorum, "debounce_ticks": probe.debounce_ticks,
            "regions": [{"region": reg, "ok": (latest[reg].ok if reg in latest else None),
                         "latency_ms": (latest[reg].latency_ms if reg in latest else None),
                         "error": (latest[reg].error if reg in latest else None),
                         **latency_stats(results, probe.name, reg, now - DAY)} for reg in probe.regions],
            "uptime": {w: uptime(probe, results, now - span, now) for w, span in WINDOWS.items()},
            "days": days,
        })
    statuses = [r["status"] for r in rows]
    if stale:
        overall = "monitor stale"
    elif "down" in statuses:
        overall = "major outage" if statuses.count("down") > 1 else "partial outage"
    elif "degraded-remote" in statuses:
        overall = "degraded (remote region)"
    elif statuses and all(s == "up" for s in statuses):
        overall = "all systems operational"
    else:
        overall = "no data yet"
    return {
        "generated_at": now, "generated_at_iso": iso(now), "last_tick": last_tick,
        "last_tick_iso": iso(last_tick), "interval_s": interval, "stale": stale,
        "stale_after_s": 2 * interval, "ticks": meta.get("ticks", 0), "overall": overall,
        "probes": rows,
        "open_incidents": [i.to_dict() for i in open_inc],
        "incident_history": [i.to_dict() for i in sorted(incidents, key=lambda i: -i.opened_at) if not i.open][:50],
    }


# --------------------------------------------------------------------------- rendering

def render_json(state: dict[str, Any]) -> str:
    return json.dumps(state, indent=2, sort_keys=True)


_COLORS = {"up": "var(--ok)", "degraded-remote": "var(--warn)", "down": "var(--bad)", "unknown": "var(--muted)"}


def _esc(s: Any) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;"))


def _bar_svg(days: list[dict[str, Any]]) -> str:
    w, h, gap = 8, 28, 2
    parts = [f'<svg class="bars" viewBox="0 0 {30 * (w + gap)} {h}" width="{30 * (w + gap)}" height="{h}" '
             f'role="img" aria-label="30-day uptime">']
    for i, d in enumerate(days):
        u = d["uptime"]
        if u is None:
            fill = "var(--muted)"
        elif u >= 0.999:
            fill = "var(--ok)"
        elif u >= 0.95:
            fill = "var(--warn)"
        else:
            fill = "var(--bad)"
        label = f"{d['date']}: " + ("no data" if u is None else f"{u * 100:.2f}% uptime")
        parts.append(f'<rect x="{i * (w + gap)}" y="0" width="{w}" height="{h}" rx="1.5" fill="{fill}">'
                     f'<title>{_esc(label)}</title></rect>')
    parts.append("</svg>")
    return "".join(parts)


def _fmt_uptime(u: float | None) -> str:
    return "n/a" if u is None else f"{u * 100:.2f}%"


def _fmt_ms(v: float | None) -> str:
    return "–" if v is None else f"{v:.0f} ms"


def render_html(state: dict[str, Any]) -> str:
    """Self-contained status page: inline CSS with variables (light/dark), inline SVG, no URLs."""
    banner_class = {"all systems operational": "ok", "monitor stale": "stale", "no data yet": "muted"}.get(
        state["overall"], "bad" if "outage" in state["overall"] else "warn")
    stale_note = ""
    if state["stale"]:
        stale_note = (f'<p class="note">Monitor stale: last tick {_esc(state["last_tick_iso"])}, older than '
                      f'{state["stale_after_s"]:.0f}s (2 × interval). The checks are not running; '
                      f'this page says nothing about the service right now.</p>')
    rows = []
    for p in state["probes"]:
        chips = "".join(
            f'<span class="chip {"ok" if r["ok"] else ("muted" if r["ok"] is None else "bad")}" '
            f'title="{_esc(r["error"] or "")}">{_esc(r["region"])}'
            f'{" ⌂" if r["region"] == p["home_region"] else ""} {_fmt_ms(r["latency_ms"])}'
            f' <small>p50 {_fmt_ms(r["p50"])} · p95 {_fmt_ms(r["p95"])}</small></span>'
            for r in p["regions"])
        rows.append(
            f'<tr class="probe" data-probe="{_esc(p["name"])}"><td><strong>{_esc(p["name"])}</strong>'
            f'<br><small>{_esc(p["kind"])} · quorum {p["quorum"]} · home {_esc(p["home_region"])}</small></td>'
            f'<td><span class="status" style="color:{_COLORS[p["status"]]}">●</span> {_esc(p["status"])}</td>'
            f'<td class="chips">{chips}</td>'
            f'<td>{_bar_svg(p["days"])}<br><small>24h {_fmt_uptime(p["uptime"]["24h"])} · '
            f'7d {_fmt_uptime(p["uptime"]["7d"])} · 30d {_fmt_uptime(p["uptime"]["30d"])}</small></td></tr>')
    open_items = "".join(
        f'<li class="incident open"><code>{_esc(i["id"])}</code> <strong>{_esc(i["probe"])}</strong> since {_esc(iso(i["opened_at"]))} '
        f'in {_esc(", ".join(i["regions"]))}<br><small>{_esc(i["cause"])}</small></li>'
        for i in state["open_incidents"]) or "<li>No open incidents.</li>"
    hist_items = "".join(
        f'<li class="incident"><code>{_esc(i["id"])}</code> <strong>{_esc(i["probe"])}</strong> {_esc(iso(i["opened_at"]))} → '
        f'{_esc(iso(i["closed_at"]))} ({(i["closed_at"] - i["opened_at"]) / 60:.0f} min) '
        f'in {_esc(", ".join(i["regions"]))}<br><small>{_esc(i["cause"])}</small></li>'
        for i in state["incident_history"]) or "<li>No past incidents.</li>"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gizmos status</title>
<style>
:root {{ --bg:#ffffff; --fg:#1b1f24; --muted:#9aa4b2; --ok:#1f9d55; --warn:#d7a000; --bad:#d13c3c;
        --card:#f4f6f8; --line:#e1e5ea; }}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{ --bg:#0f1317; --fg:#e6e9ee; --muted:#5b6673; --ok:#3ccf7a;
        --warn:#f0c040; --bad:#ff6b6b; --card:#171c22; --line:#2a313a; }} }}
:root[data-theme="dark"] {{ --bg:#0f1317; --fg:#e6e9ee; --muted:#5b6673; --ok:#3ccf7a; --warn:#f0c040;
        --bad:#ff6b6b; --card:#171c22; --line:#2a313a; }}
body {{ margin:0; padding:24px 16px; background:var(--bg); color:var(--fg);
       font:15px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; max-width:1100px; margin-inline:auto; }}
.banner {{ padding:18px 20px; border-radius:10px; font-size:20px; font-weight:600; color:#fff; }}
.banner.ok {{ background:var(--ok); }} .banner.warn {{ background:var(--warn); color:#1b1f24; }}
.banner.bad {{ background:var(--bad); }} .banner.stale {{ background:var(--muted); color:#1b1f24; }}
.banner.muted {{ background:var(--muted); }}
.note {{ background:var(--card); border-left:4px solid var(--warn); padding:10px 14px; border-radius:6px; }}
table {{ width:100%; border-collapse:collapse; margin-top:16px; }}
td, th {{ text-align:left; vertical-align:top; padding:10px 8px; border-top:1px solid var(--line); }}
.chip {{ display:inline-block; padding:3px 8px; margin:2px 4px 2px 0; border-radius:999px; background:var(--card);
        border:1px solid var(--line); white-space:nowrap; }}
.chip.ok {{ border-color:var(--ok); }} .chip.bad {{ border-color:var(--bad); color:var(--bad); }}
.chip small {{ color:var(--muted); }}
.bars {{ display:block; max-width:100%; }}
ul {{ padding-left:18px; }} .incident.open {{ color:var(--bad); }}
footer {{ color:var(--muted); margin-top:24px; font-size:13px; }}
@media (max-width: 700px) {{ td, th {{ display:block; border-top:none; }} tr.probe {{ display:block; border-top:1px solid var(--line); }} }}
</style>
</head>
<body>
<div class="banner {banner_class}">{_esc(state["overall"].capitalize())}</div>
{stale_note}
<p>Last checked: <strong>{_esc(state["last_tick_iso"])}</strong> · page generated {_esc(state["generated_at_iso"])}
 · checks every {state["interval_s"]:.0f}s from {len(state["probes"][0]["regions"]) if state["probes"] else 0} regions
 · {state["ticks"]} ticks recorded</p>
<table>
<thead><tr><th>Component</th><th>Status</th><th>Regions (⌂ = home) · latest / p50 / p95 (24h)</th><th>30-day uptime</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
<h2>Open incidents</h2>
<ul>{open_items}</ul>
<h2>Incident history</h2>
<ul>{hist_items}</ul>
<footer>A probe is down only if its home region fails or at least <em>quorum</em> regions fail together;
a single far region failing alone shows as degraded-remote and never opens an incident. Incidents open after
consecutive failed ticks (debounce) and close on the first healthy tick. This page is static, has no
external dependencies, and is generated by a worker independent of the platform it watches.</footer>
</body>
</html>
"""
