"""Harness, scenarios, analysis and reports.

Scenarios produce a per-second schedule of users; the harness turns it into seeded arrivals, drives
the target (sequentially on a virtual clock, or through a ThreadPoolExecutor on the wall clock),
and buckets the responses into a per-second series:

    offered      arrivals scheduled in [s, s+1)
    throughput   2xx responses *completed* in [s, s+1)  (plateaus at capacity under overload)
    p50/p95/p99  latency of requests that *arrived* in the second
    error_rate   5xx / offered;  rate_limited = 429 / offered
    in_flight    time-averaged requests in flight during the second (from arrival/completion times)

Analysis:
    knee          first load level (users) whose p95 >= 2 x baseline p95 or error_rate > 1%
    baseline p95  median p95 over the low-load points (users <= 10% of max, first 3 warm-up seconds dropped: cold-start window + backlog drain)
    sustainable   largest `users` seen before the knee with p95 < 2x baseline and errors <= 1%
    Little's law  L = lambda * W per second: in_flight vs offered x mean latency; median relative error
    headroom      launch target / knee: "3.1x short" or "1.4x headroom"
"""
from __future__ import annotations

import html
import json
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evalkit.report import md_table, pct
from projects.p25_surge_test.targets import RateLimiter, Request, SimulatedTarget, Target

HERE = Path(__file__).parent
DEFAULT_CONFIG = HERE / "data" / "scenarios.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------- scenarios


@dataclass
class Scenario:
    """kind: ramp (1 -> users over duration) | spike (users join within window, then hold) |
    soak (users for duration) | step (start_users, x2 ... up to users, each for step_seconds)."""
    kind: str
    users: int
    duration: int = 60           # ramp / soak
    window: int = 10             # spike: seconds in which everyone joins
    hold: int = 20               # spike: seconds to hold after the join window
    start_users: int = 500       # step
    step_seconds: int = 10       # step
    rps_per_user: float = 1.0
    join_burst: int = 0          # extra requests a user fires the second they join (spike: page load)
    seed: int = 0
    name: str = ""

    @classmethod
    def from_config(cls, cfg: dict[str, Any], **overrides: Any) -> Scenario:
        d = {k: v for k, v in cfg.items() if k in cls.__dataclass_fields__}
        d.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**d)

    def levels(self) -> list[int]:
        if self.kind == "step":
            out, u = [], max(1, self.start_users)
            while u < self.users:
                out.append(u)
                u *= 2
            out.append(self.users)
            return out
        return [self.users]

    @property
    def total_seconds(self) -> int:
        if self.kind == "spike":
            return self.window + self.hold
        if self.kind == "step":
            return len(self.levels()) * self.step_seconds
        return self.duration

    def users_at(self, s: int) -> int:
        if self.kind == "ramp":
            return max(1, round(self.users * (s + 1) / max(1, self.duration)))
        if self.kind == "spike":
            return self.users if s >= self.window else round(self.users * (s + 1) / max(1, self.window))
        if self.kind == "step":
            return self.levels()[min(s // self.step_seconds, len(self.levels()) - 1)]
        if self.kind == "soak":
            return self.users
        raise ValueError(f"unknown scenario kind {self.kind!r}")

    def schedule(self) -> list[Request]:
        """Seeded arrivals: each active user fires rps_per_user requests per second at uniform
        offsets; users who join this second also fire `join_burst` extra requests."""
        rng = np.random.default_rng(self.seed)
        out: list[Request] = []
        prev, seq = 0, 0
        for s in range(self.total_seconds):
            users = self.users_at(s)
            n_steady = int(round(users * self.rps_per_user))
            joins = max(0, users - prev)
            n_burst = joins * self.join_burst
            n = n_steady + n_burst
            offsets = rng.random(n)
            ids = np.concatenate([rng.integers(0, users, n_steady), np.repeat(np.arange(prev, users), self.join_burst)]) if n else np.zeros(0, int)
            for k in range(n):
                out.append(Request(int(ids[k]), s + float(offsets[k]), "/", seq))
                seq += 1
            prev = users
        out.sort(key=lambda r: r.t)
        return out


# ---------------------------------------------------------------- harness


@dataclass
class Sample:
    t_arr: float
    t_done: float
    status: int
    latency_ms: float
    nbytes: int
    user_id: int


@dataclass
class SeriesPoint:
    second: int
    users: int
    offered: int
    throughput: int
    p50: float
    p95: float
    p99: float
    error_rate: float
    rate_limited: float
    in_flight: float
    mean_latency_ms: float


@dataclass
class RunResult:
    scenario: Scenario
    series: list[SeriesPoint]
    n_requests: int
    n_ok: int
    n_5xx: int
    n_429: int
    bytes_total: int
    wall_seconds: float
    mode: str                     # virtual | threads
    target_name: str
    samples: list[Sample] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {"scenario": asdict(self.scenario), "series": [asdict(p) for p in self.series],
                "n_requests": self.n_requests, "n_ok": self.n_ok, "n_5xx": self.n_5xx, "n_429": self.n_429,
                "bytes_total": self.bytes_total, "wall_seconds": round(self.wall_seconds, 3), "mode": self.mode,
                "target": self.target_name}


def _bucket(samples: list[Sample], scenario: Scenario) -> list[SeriesPoint]:
    T = scenario.total_seconds
    arr = [[] for _ in range(T)]
    done_ok = [0] * T
    inflight = [0.0] * T
    for s in samples:
        a = min(T - 1, int(s.t_arr))
        arr[a].append(s)
        if 200 <= s.status < 400:
            d = int(s.t_done)
            if 0 <= d < T:
                done_ok[d] += 1
        lo, hi = int(s.t_arr), min(T - 1, int(s.t_done))
        for sec in range(max(0, lo), hi + 1):
            inflight[sec] += min(s.t_done, sec + 1) - max(s.t_arr, sec)
    out = []
    for sec in range(T):
        rows = arr[sec]
        lat = np.array([r.latency_ms for r in rows if r.status != 429]) if rows else np.zeros(0)
        n = len(rows)
        e5 = sum(1 for r in rows if r.status >= 500)
        e4 = sum(1 for r in rows if r.status == 429)
        p50, p95, p99 = (float(np.percentile(lat, 50)), float(np.percentile(lat, 95)), float(np.percentile(lat, 99))) if len(lat) else (0.0, 0.0, 0.0)
        out.append(SeriesPoint(sec, scenario.users_at(sec), n, done_ok[sec], p50, p95, p99,
                               e5 / n if n else 0.0, e4 / n if n else 0.0, inflight[sec],
                               float(lat.mean()) if len(lat) else 0.0))
    return out


def run_scenario(scenario: Scenario, target: Target, *, virtual: bool = True, max_workers: int = 64,
                 time_scale: float = 1.0, target_name: str = "") -> RunResult:
    """Drive `target` with the scenario's schedule.

    virtual=True : calls are made sequentially in arrival order; the target reads `request.t`.
    virtual=False: a ThreadPoolExecutor issues each request at its scheduled wall-clock offset
                   (scaled by `time_scale`, e.g. 0.01 to run a 60 s scenario in 0.6 s)."""
    sched = scenario.schedule()
    t_wall = time.perf_counter()
    samples: list[Sample] = []
    if virtual:
        for req in sched:
            resp = target.call(req)
            samples.append(Sample(req.t, req.t + resp.latency_ms / 1000.0, resp.status, resp.latency_ms, resp.bytes, req.user_id))
        mode = "virtual"
    else:
        t0 = time.perf_counter()

        def fire(req: Request) -> Sample:
            delay = req.t * time_scale - (time.perf_counter() - t0)
            if delay > 0:
                time.sleep(delay)
            start = time.perf_counter()
            resp = target.call(req)
            # bucket by the *scheduled* second (scheduling jitter at time_scale=0.01 would otherwise move
            # a request a whole second); latency and completion are wall clock
            return Sample(req.t, req.t + (time.perf_counter() - start), resp.status, resp.latency_ms, resp.bytes, req.user_id)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            samples = list(ex.map(fire, sched))
        mode = "threads"
    series = _bucket(samples, scenario)
    return RunResult(scenario, series, len(samples), sum(1 for s in samples if 200 <= s.status < 400),
                     sum(1 for s in samples if s.status >= 500), sum(1 for s in samples if s.status == 429),
                     sum(s.nbytes for s in samples), time.perf_counter() - t_wall, mode,
                     target_name or type(target).__name__, samples)


# ---------------------------------------------------------------- analysis


@dataclass
class Analysis:
    baseline_p95_ms: float
    knee_users: int | None
    knee_second: int | None
    knee_reason: str
    max_sustainable_users: int
    peak_throughput: int
    p95_max_ms: float
    error_rate_max: float
    rate_limited_max: float
    littles_median_rel_error: float | None
    littles_law_ok: bool
    littles_tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_knee(series: list[SeriesPoint], p95_factor: float = 2.0, error_threshold: float = 0.01,
              min_responses: int = 5, warmup: int = 3) -> tuple[float, SeriesPoint | None, str]:
    """(baseline_p95, knee point or None, reason). Points are walked in increasing load; the knee is
    the first point whose p95 >= factor x baseline or whose 5xx rate > threshold."""
    pts = [p for p in series if p.offered >= min_responses]
    if not pts:
        return 0.0, None, "no traffic"
    # warm-up seconds (cold starts, connection setup, backlog drain) are neither baseline nor knee
    # candidates; scenarios that need the first seconds measured (a spike) should start lower
    skip = {p.second for p in pts[:warmup]} if len(pts) >= 5 * warmup else set()
    pts = [p for p in pts if p.second not in skip]
    max_users = max(p.users for p in pts)
    low = [p for p in pts if p.users <= max(1, 0.1 * max_users)] or pts[:max(1, len(pts) // 10)]
    baseline = statistics.median(p.p95 for p in low)
    for p in sorted(pts, key=lambda p: (p.users, p.second)):
        if p.error_rate > error_threshold:
            return baseline, p, f"error rate {pct(p.error_rate)} > {pct(error_threshold)}"
        if baseline > 0 and p.p95 >= p95_factor * baseline:
            return baseline, p, f"p95 {p.p95:.0f} ms >= {p95_factor:g} x baseline {baseline:.0f} ms"
    return baseline, None, "no knee within the tested load"


def littles_law(series: list[SeriesPoint], min_responses: int = 20) -> float | None:
    """Median over seconds of |offered x mean_latency_s - in_flight| / in_flight.
    L = lambda W (Little, 1961) holds for any stable system; a large error means the harness
    mis-measures or the system is far from steady state (e.g. inside a spike's join window)."""
    errs = []
    for p in series:
        if p.offered >= min_responses and p.in_flight > 0:
            est = p.offered * (1 - p.rate_limited) * p.mean_latency_ms / 1000.0
            errs.append(abs(est - p.in_flight) / p.in_flight)
    return float(statistics.median(errs)) if errs else None


def analyse(result: RunResult, littles_tolerance: float = 0.25) -> Analysis:
    series = result.series
    baseline, knee, reason = find_knee(series)
    served = [p for p in series if p.offered >= 5]
    if knee is not None:
        ok_pts = [p for p in served if p.users < knee.users and p.error_rate <= 0.01 and (baseline <= 0 or p.p95 < 2 * baseline)]
    else:
        ok_pts = [p for p in served if p.error_rate <= 0.01 and (baseline <= 0 or p.p95 < 2 * baseline)]
    sustainable = max((p.users for p in ok_pts), default=0)
    lerr = littles_law(series)
    return Analysis(round(baseline, 2), knee.users if knee else None, knee.second if knee else None, reason, sustainable,
                    max((p.throughput for p in series), default=0), round(max((p.p95 for p in series), default=0.0), 2),
                    max((p.error_rate for p in series), default=0.0), max((p.rate_limited for p in series), default=0.0),
                    round(lerr, 4) if lerr is not None else None, lerr is not None and lerr <= littles_tolerance, littles_tolerance)


@dataclass
class Headroom:
    launch_users: int
    knee_users: int | None
    tested_users: int
    ratio: float            # launch / knee (> 1 means short)
    short: bool
    statement: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def headroom(launch_users: int, knee_users: int | None, tested_users: int, context: str = "launch demo") -> Headroom:
    """ratio = launch / knee. Short when the knee is below the launch target. With no knee found,
    the best claim is 'at least tested/launch x headroom'."""
    if knee_users is None:
        ratio = launch_users / max(1, tested_users)
        short = tested_users < launch_users
        stmt = (f"projected {launch_users:,} users at {context}; harness found no knee up to {tested_users:,} users"
                + (f" -> at least {tested_users / launch_users:.1f}x headroom" if not short
                   else f" -> untested above {tested_users:,}; run a bigger scenario before claiming headroom"))
        return Headroom(launch_users, None, tested_users, round(ratio, 3), short, stmt)
    ratio = launch_users / max(1, knee_users)
    short = knee_users < launch_users
    stmt = (f"projected {launch_users:,} users at {context}; harness shows knee at {knee_users:,} -> "
            + (f"{ratio:.1f}x short" if short else f"{1 / ratio:.1f}x headroom"))
    return Headroom(launch_users, knee_users, tested_users, round(ratio, 3), short, stmt)


# ---------------------------------------------------------------- reports


def ascii_chart(values: list[float], labels: list[str], width: int = 40, unit: str = "", max_rows: int = 20) -> str:
    """Horizontal bars, sampled to <= max_rows rows, scaled to the max value."""
    if not values:
        return "(no data)"
    step = max(1, math.ceil(len(values) / max_rows))
    idx = list(range(0, len(values), step))
    vmax = max(values) or 1.0
    rows = []
    for i in idx:
        bar = "#" * int(round(values[i] / vmax * width))
        rows.append(f"{labels[i]:>6} |{bar:<{width}}| {values[i]:.4g}{unit}")
    return "\n".join(rows)


def render_markdown(result: RunResult, analysis: Analysis, head: Headroom | None = None, max_rows: int = 16) -> str:
    sc = result.scenario
    s = result.series
    out = [f"## surge test: {sc.kind} ({sc.name or sc.kind}) -> {result.target_name} [{result.mode}]", "",
           md_table(["users", "seconds", "requests", "ok", "5xx", "429", "peak rps", "p95 max", "err max", "wall s"],
                    [[sc.users, sc.total_seconds, result.n_requests, result.n_ok, result.n_5xx, result.n_429,
                      analysis.peak_throughput, f"{analysis.p95_max_ms:.0f} ms", pct(analysis.error_rate_max),
                      f"{result.wall_seconds:.2f}"]]), "",
           md_table(["baseline p95", "knee (users)", "knee reason", "max sustainable users", "Little's law median error", "little ok"],
                    [[f"{analysis.baseline_p95_ms:.0f} ms", analysis.knee_users if analysis.knee_users is not None else "none",
                      analysis.knee_reason, analysis.max_sustainable_users,
                      pct(analysis.littles_median_rel_error) if analysis.littles_median_rel_error is not None else "-",
                      "yes" if analysis.littles_law_ok else "no"]])]
    if head is not None:
        out += ["", f"**headroom:** {head.statement}"]
    labels = [f"s={p.second}" for p in s]
    out += ["", "### p95 latency per second (ms)", "", "```text", ascii_chart([p.p95 for p in s], labels, unit=" ms"), "```",
            "", "### 5xx error rate per second", "", "```text", ascii_chart([p.error_rate * 100 for p in s], labels, unit="%"), "```"]
    if result.n_429:
        out += ["", "### 429 (rate limited) per second", "", "```text", ascii_chart([p.rate_limited * 100 for p in s], labels, unit="%"), "```"]
    step = max(1, math.ceil(len(s) / max_rows))
    rows = [[p.second, p.users, p.offered, p.throughput, f"{p.p50:.0f}", f"{p.p95:.0f}", f"{p.p99:.0f}",
             pct(p.error_rate), pct(p.rate_limited), f"{p.in_flight:.1f}"] for p in s[::step]]
    out += ["", "### Series (sampled)", "", md_table(["s", "users", "offered", "rps ok", "p50", "p95", "p99", "5xx", "429", "in-flight"], rows)]
    return "\n".join(out)


def _svg_line_chart(title: str, xs: list[float], series: dict[str, list[float]], unit: str = "",
                    width: int = 720, height: int = 220, marker_x: float | None = None) -> str:
    """Inline SVG polylines; no xmlns, no external resources."""
    pad_l, pad_r, pad_t, pad_b = 56, 16, 28, 28
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    xmax = max(xs) if xs else 1.0
    ymax = max((max(v) for v in series.values() if v), default=1.0) or 1.0
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    def X(x: float) -> float:
        return pad_l + (x / xmax) * w if xmax else pad_l

    def Y(y: float) -> float:
        return pad_t + h - (y / ymax) * h

    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{html.escape(title)}">',
             f'<text x="{pad_l}" y="18" class="t">{html.escape(title)}</text>',
             f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + h}" class="ax"/>',
             f'<line x1="{pad_l}" y1="{pad_t + h}" x2="{pad_l + w}" y2="{pad_t + h}" class="ax"/>']
    for k in range(5):
        yv = ymax * k / 4
        parts.append(f'<text x="{pad_l - 6}" y="{Y(yv) + 4:.1f}" class="lbl" text-anchor="end">{yv:.3g}{unit}</text>')
        parts.append(f'<line x1="{pad_l}" y1="{Y(yv):.1f}" x2="{pad_l + w}" y2="{Y(yv):.1f}" class="grid"/>')
    for k in range(5):
        xv = xmax * k / 4
        parts.append(f'<text x="{X(xv):.1f}" y="{height - 8}" class="lbl" text-anchor="middle">{xv:.3g}s</text>')
    for i, (name, vals) in enumerate(series.items()):
        pts = " ".join(f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(xs, vals))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[i % len(colors)]}" stroke-width="2"/>')
        parts.append(f'<text x="{pad_l + 8 + i * 150}" y="{pad_t + 12}" fill="{colors[i % len(colors)]}" class="lbl">{html.escape(name)}</text>')
    if marker_x is not None:
        parts.append(f'<line x1="{X(marker_x):.1f}" y1="{pad_t}" x2="{X(marker_x):.1f}" y2="{pad_t + h}" stroke="#888" stroke-dasharray="4 3"/>')
        parts.append(f'<text x="{X(marker_x) + 4:.1f}" y="{pad_t + h - 6}" class="lbl">knee</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def render_html(result: RunResult, analysis: Analysis, head: Headroom | None = None) -> str:
    s = result.series
    xs = [float(p.second) for p in s]
    knee_x = float(analysis.knee_second) if analysis.knee_second is not None else None
    charts = [
        _svg_line_chart("offered load vs ok throughput (req/s)", xs, {"offered": [float(p.offered) for p in s], "ok throughput": [float(p.throughput) for p in s]}, marker_x=knee_x),
        _svg_line_chart("latency p50 / p95 / p99 (ms)", xs, {"p50": [p.p50 for p in s], "p95": [p.p95 for p in s], "p99": [p.p99 for p in s]}, " ms", marker_x=knee_x),
        _svg_line_chart("5xx and 429 rate (%)", xs, {"5xx": [p.error_rate * 100 for p in s], "429": [p.rate_limited * 100 for p in s]}, "%", marker_x=knee_x),
        _svg_line_chart("in flight (time-averaged) vs users/100", xs, {"in flight": [p.in_flight for p in s], "users/100": [p.users / 100 for p in s]}, marker_x=knee_x),
    ]
    sc = result.scenario
    rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in [
        ("scenario", f"{sc.kind} users={sc.users} seconds={sc.total_seconds}"), ("target", result.target_name),
        ("requests", f"{result.n_requests} (ok {result.n_ok}, 5xx {result.n_5xx}, 429 {result.n_429})"),
        ("baseline p95", f"{analysis.baseline_p95_ms:.0f} ms"), ("knee", f"{analysis.knee_users} users ({analysis.knee_reason})" if analysis.knee_users else analysis.knee_reason),
        ("max sustainable users", analysis.max_sustainable_users), ("peak ok throughput", f"{analysis.peak_throughput} rps"),
        ("Little's law", f"median error {pct(analysis.littles_median_rel_error)}" if analysis.littles_median_rel_error is not None else "-"),
        ("headroom", head.statement if head else "-")])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>surge test: {html.escape(sc.kind)}</title>
<style>body{{font:14px system-ui,sans-serif;margin:24px;color:#222}}table{{border-collapse:collapse;margin-bottom:16px}}th,td{{text-align:left;padding:4px 10px;border-bottom:1px solid #ddd}}svg{{display:block;margin:12px 0}}.t{{font-size:14px;font-weight:600}}.lbl{{font-size:11px;fill:#555}}.ax{{stroke:#444}}.grid{{stroke:#eee}}</style>
</head><body>
<h1>surge test: {html.escape(sc.kind)} -> {html.escape(result.target_name)}</h1>
<table>{rows}</table>
{chr(10).join(charts)}
</body></html>
"""


def result_json(result: RunResult, analysis: Analysis, head: Headroom | None = None) -> str:
    return json.dumps({"run": result.to_dict(), "analysis": analysis.to_dict(), "headroom": head.to_dict() if head else None}, indent=1)


# ---------------------------------------------------------------- convenience


def build_target(kind: str, cfg: dict[str, Any], *, url: str | None = None, seed: int = 0, virtual: bool = True,
                 rate_limit: bool = False, fn=None) -> tuple[Target, str]:
    from projects.p25_surge_test.targets import FunctionTarget, HttpTarget
    if kind == "simulated":
        t: Target = SimulatedTarget.from_config(cfg.get("target", {}).get("simulated", {}), seed=seed, virtual_clock=virtual)
        name = f"simulated(c={t.concurrency_limit}, service={t.service_ms:g}ms)"
    elif kind == "http":
        if not url:
            raise ValueError("--url is required for --target http")
        t, name = HttpTarget(url), f"http({url})"
    elif kind == "function":
        t, name = FunctionTarget(fn or (lambda r: 200)), "function"
    else:
        raise ValueError(f"unknown target {kind!r}")
    if rate_limit:
        rl = cfg.get("rate_limiter", {})
        t = RateLimiter(t, rl.get("per_user_rps", 1.0), rl.get("burst", 2), rl.get("global_rps"), rl.get("global_burst"), virtual_clock=virtual)
        name += f" + ratelimit(user={rl.get('per_user_rps', 1.0)}rps/burst{rl.get('burst', 2)}, global={rl.get('global_rps')})"
    return t, name
