"""The scary line: capacity analysis, placement advisor and sampling audit for a shared store.

Three engines, all pure functions over in-memory data:

CapacityLine
    utilisation      total_bytes / cap_bytes for the latest snapshot
    composition      bytes per category and its share of the total
    growth           Theil-Sen slope (median of pairwise slopes, bytes/day) over the last
                     `window` days per category and in total: robust to one-off spikes and
                     to the migration-day cliff, unlike least squares
    p90 growth       90th percentile of day-over-day deltas in the window: the pessimistic rate
    days_to_cap      (cap - current) / slope, infinite when slope <= 0
    biggest          the single table with the most bytes, and the projection if its whole
                     category were moved out of the store ("what if we moved it out")
    alerts           thresholds (50/75/90 %) already crossed, with the first day they were
                     crossed, plus the projected day at which the uncrossed ones will be hit
    sparkline / svg  ASCII and inline-SVG views of the utilisation history

Placement advisor
    recommend_placement(question) maps a question's requirements to a store class:
        retention == cold                                -> cold_archive
        exactness == exact and write_rate == high        -> exact_counter
        exactness == approximate                         -> analytics_sampled
        otherwise                                        -> transactional
    audit_layout() compares the store class that actually serves each question with the
    recommendation; the severity matrix in SEVERITY says whether a mismatch is wrong
    (critical: a sampled store answering an exact question), wasteful (warning: cold or
    approximate data in the hot transactional store) or merely odd (info).

Sampling audit
    A sampled store keeps each event with probability r. For a user with k events,
    P(user appears in the sample) = 1 - (1 - r)^k, so distinct-user counts taken straight from
    the sample undercount by the population average of (1 - r)^k, which is large whenever most
    users have few events (the long tail). Event counts scaled by 1/r are unbiased:
    E[sampled_events / r] = events. bootstrap_ci over repeated seeded simulations gives the
    interval on both statistics.
"""

from __future__ import annotations

import html
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evalkit.io import read_jsonl
from evalkit.metrics import bootstrap_ci
from evalkit.report import md_table, pct

GB = 1024**3
MB = 1024**2
CATEGORIES = ("platform", "analytics", "versions", "logs", "other")
STORE_CLASSES = ("transactional", "analytics_sampled", "exact_counter", "cold_archive")
DEFAULT_THRESHOLDS = (0.5, 0.75, 0.9)
BLOCKS = "▁▂▃▄▅▆▇█"


# ---------------------------------------------------------------- snapshots


@dataclass(frozen=True)
class TableStat:
    name: str
    category: str
    rows: int
    bytes: int


@dataclass
class StorageSnapshot:
    day: int
    date: str
    cap_bytes: int
    tables: list[TableStat]

    @property
    def total_bytes(self) -> int:
        return sum(t.bytes for t in self.tables)

    @property
    def utilisation(self) -> float:
        return self.total_bytes / self.cap_bytes if self.cap_bytes else 0.0

    def by_category(self) -> dict[str, int]:
        out: dict[str, int] = {c: 0 for c in CATEGORIES}
        for t in self.tables:
            out[t.category] = out.get(t.category, 0) + t.bytes
        return out

    @classmethod
    def from_dict(cls, row: dict[str, Any], cap_bytes: int | None = None) -> StorageSnapshot:
        tables = [
            TableStat(t["name"], t.get("category", "other"), int(t.get("rows", 0)), int(t["bytes"]))
            for t in row["tables"]
        ]
        cap = int(cap_bytes if cap_bytes is not None else row.get("cap_bytes", 0))
        if cap <= 0:
            raise ValueError("cap_bytes must be positive (set it in the snapshot or pass --cap-gb)")
        return cls(int(row["day"]), str(row.get("date", row["day"])), cap, tables)


def load_snapshots(path: str | Path, cap_bytes: int | None = None) -> list[StorageSnapshot]:
    snaps = [StorageSnapshot.from_dict(r, cap_bytes) for r in read_jsonl(path)]
    snaps.sort(key=lambda s: s.day)
    if not snaps:
        raise ValueError(f"{path}: no snapshots")
    return snaps


# ---------------------------------------------------------------- growth


def theil_sen(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Median of pairwise slopes (y_j - y_i) / (x_j - x_i), i < j. Robust to ~29% outliers."""
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if x.size < 2:
        return 0.0
    i, j = np.triu_indices(x.size, k=1)
    dx = x[j] - x[i]
    ok = dx != 0
    if not ok.any():
        return 0.0
    return float(np.median((y[j] - y[i])[ok] / dx[ok]))


def p90_growth(ys: Sequence[float]) -> float:
    """90th percentile of consecutive day-over-day deltas (pessimistic daily growth)."""
    y = np.asarray(ys, float)
    if y.size < 2:
        return 0.0
    return float(np.percentile(np.diff(y), 90))


def days_to_cap(current: float, cap: float, slope: float) -> float:
    """(cap - current) / slope; inf when not growing; 0 when already over."""
    if current >= cap:
        return 0.0
    if slope <= 0:
        return math.inf
    return (cap - current) / slope


def sparkline(values: Sequence[float], lo: float = 0.0, hi: float | None = None) -> str:
    v = np.asarray(values, float)
    if v.size == 0:
        return ""
    hi = max(v.max(), lo + 1e-12) if hi is None else hi
    idx = np.clip(((v - lo) / (hi - lo) * (len(BLOCKS) - 1)).round(), 0, len(BLOCKS) - 1)
    return "".join(BLOCKS[int(i)] for i in idx)


# ---------------------------------------------------------------- capacity report


@dataclass
class CategoryGrowth:
    category: str
    bytes: int
    share: float
    slope_bytes_per_day: float
    p90_bytes_per_day: float


@dataclass
class Alert:
    threshold: float
    crossed: bool
    first_day: int | None  # day the line first crossed the threshold (if crossed)
    projected_days: float | None  # days from now until crossing at current growth (if not crossed)


@dataclass
class CapacityReport:
    day: int
    date: str
    cap_bytes: int
    total_bytes: int
    utilisation: float
    window: int
    composition: list[CategoryGrowth]
    slope_bytes_per_day: float
    p90_bytes_per_day: float
    days_to_cap: float
    days_to_cap_p90: float
    biggest_table: TableStat
    biggest_category: str
    what_if_moved: dict[str, float]
    alerts: list[Alert]
    history_utilisation: list[float]
    history_days: list[int]
    sparkline: str
    fastest_category: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["days_to_cap"] = _jsonable(self.days_to_cap)
        d["days_to_cap_p90"] = _jsonable(self.days_to_cap_p90)
        d["what_if_moved"] = {k: (_jsonable(v) if isinstance(v, float) else v) for k, v in self.what_if_moved.items()}
        for a in d["alerts"]:
            a["projected_days"] = _jsonable(a["projected_days"])
        return d

    def to_markdown(self) -> str:
        lines = [
            f"# The line: day {self.day} ({self.date})",
            "",
            (f"**Utilisation {pct(self.utilisation)}** ({_gb(self.total_bytes)} of {_gb(self.cap_bytes)}). "
            f"Growth {_mb(self.slope_bytes_per_day)}/day (Theil-Sen over last {self.window} days), "
            f"p90 {_mb(self.p90_bytes_per_day)}/day. "
            f"**Days to cap: {_days(self.days_to_cap)}** (p90 growth: {_days(self.days_to_cap_p90)})."),
            "",
            f"`{self.sparkline}` utilisation, day {self.history_days[0]} -> {self.day}",
            "",
            "## Composition",
            "",
            md_table(
                ["category", "GB", "share", "growth MB/day", "p90 MB/day"],
                [
                    [c.category, _gbf(c.bytes), pct(c.share), _mbf(c.slope_bytes_per_day), _mbf(c.p90_bytes_per_day)]
                    for c in self.composition
                ],
            ),
            "",
            (f"Fastest-growing category: **{self.fastest_category}**. Biggest table: "
            f"**{self.biggest_table.name}** ({self.biggest_table.category}, {_gb(self.biggest_table.bytes)}, "
            f"{self.biggest_table.rows:,} rows)."),
            "",
            (f"What if **{self.biggest_category}** moved out: utilisation {pct(self.what_if_moved['utilisation'])}, "
            f"growth {_mb(self.what_if_moved['slope_bytes_per_day'])}/day, "
            f"days to cap {_days(self.what_if_moved['days_to_cap'])}."),
            "",
            "## Alerts",
            "",
            md_table(
                ["threshold", "status", "detail"],
                [
                    [
                        pct(a.threshold, 0),
                        "CROSSED" if a.crossed else "ok",
                        f"first crossed day {a.first_day}" if a.crossed else f"projected in {_days(a.projected_days)}",
                    ]
                    for a in self.alerts
                ],
            ),
        ]
        return "\n".join(lines)


class CapacityLine:
    """Analyse a time series of StorageSnapshot into a CapacityReport."""

    def __init__(self, window: int = 14, thresholds: Sequence[float] = DEFAULT_THRESHOLDS):
        if window < 2:
            raise ValueError("window must be >= 2")
        self.window = window
        self.thresholds = tuple(sorted(thresholds))

    def analyse(self, snapshots: Sequence[StorageSnapshot]) -> CapacityReport:
        if not snapshots:
            raise ValueError("no snapshots")
        snaps = sorted(snapshots, key=lambda s: s.day)
        last = snaps[-1]
        cap = last.cap_bytes
        recent = snaps[-self.window :]
        days = [s.day for s in recent]
        totals = [s.total_bytes for s in recent]
        slope = theil_sen(days, totals)
        p90 = p90_growth(totals)

        cats = sorted({t.category for s in snaps for t in s.tables}, key=lambda c: (c not in CATEGORIES, c))
        total = last.total_bytes or 1
        comp: list[CategoryGrowth] = []
        for c in cats:
            series = [s.by_category().get(c, 0) for s in recent]
            comp.append(
                CategoryGrowth(
                    category=c,
                    bytes=int(series[-1]),
                    share=series[-1] / total,
                    slope_bytes_per_day=theil_sen(days, series),
                    p90_bytes_per_day=p90_growth(series),
                )
            )
        comp.sort(key=lambda g: -g.bytes)
        fastest = max(comp, key=lambda g: g.slope_bytes_per_day).category if comp else "-"

        biggest = max(last.tables, key=lambda t: t.bytes)
        big_cat = biggest.category
        moved_slope = slope - next(g.slope_bytes_per_day for g in comp if g.category == big_cat)
        moved_bytes = last.total_bytes - last.by_category().get(big_cat, 0)
        what_if = {
            "category": big_cat,
            "bytes": float(moved_bytes),
            "utilisation": moved_bytes / cap,
            "slope_bytes_per_day": moved_slope,
            "days_to_cap": days_to_cap(moved_bytes, cap, moved_slope),
        }

        hist_u = [s.utilisation for s in snaps]
        alerts = []
        for th in self.thresholds:
            crossed_days = [s.day for s in snaps if s.utilisation >= th]
            crossed = last.utilisation >= th
            first = min(crossed_days) if crossed_days else None
            proj = None if crossed else days_to_cap(last.total_bytes, th * cap, slope)
            alerts.append(Alert(th, crossed, first if crossed else None, proj))

        return CapacityReport(
            day=last.day,
            date=last.date,
            cap_bytes=cap,
            total_bytes=last.total_bytes,
            utilisation=last.utilisation,
            window=len(recent),
            composition=comp,
            slope_bytes_per_day=slope,
            p90_bytes_per_day=p90,
            days_to_cap=days_to_cap(last.total_bytes, cap, slope),
            days_to_cap_p90=days_to_cap(last.total_bytes, cap, p90),
            biggest_table=biggest,
            biggest_category=big_cat,
            what_if_moved=what_if,
            alerts=alerts,
            history_utilisation=hist_u,
            history_days=[s.day for s in snaps],
            sparkline=sparkline(hist_u, 0.0, 1.0),
            fastest_category=fastest,
        )


# ---------------------------------------------------------------- html / svg


def svg_chart(days: Sequence[int], utilisation: Sequence[float], thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
              width: int = 720, height: int = 260) -> str:
    """Inline SVG line chart of utilisation (0..1) with threshold rules. No external refs."""
    pad_l, pad_r, pad_t, pad_b = 48, 16, 16, 28
    w, h = width - pad_l - pad_r, height - pad_t - pad_b
    d0, d1 = (days[0], days[-1]) if days else (0, 1)
    span = max(1, d1 - d0)

    def px(d: float) -> float:
        return pad_l + (d - d0) / span * w

    def py(u: float) -> float:
        return pad_t + (1 - min(max(u, 0.0), 1.2) / 1.2) * h

    pts = " ".join(f"{px(d):.1f},{py(u):.1f}" for d, u in zip(days, utilisation))
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="utilisation over time">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>',
        f'<line x1="{pad_l}" y1="{py(0):.1f}" x2="{width - pad_r}" y2="{py(0):.1f}" stroke="#999"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{py(0):.1f}" stroke="#999"/>',
    ]
    colours = {0.5: "#c9a227", 0.75: "#d2691e", 0.9: "#b22222"}
    for th in thresholds:
        parts.append(
            f'<line x1="{pad_l}" y1="{py(th):.1f}" x2="{width - pad_r}" y2="{py(th):.1f}" '
            f'stroke="{colours.get(th, "#888")}" stroke-dasharray="6 4"/>'
            f'<text x="{width - pad_r - 2}" y="{py(th) - 3:.1f}" font-size="11" text-anchor="end" '
            f'fill="{colours.get(th, "#888")}">{int(th * 100)}%</text>'
        )
    parts.append(f'<line x1="{pad_l}" y1="{py(1.0):.1f}" x2="{width - pad_r}" y2="{py(1.0):.1f}" stroke="#000" stroke-width="2"/>')
    parts.append(f'<text x="{pad_l + 4}" y="{py(1.0) - 4:.1f}" font-size="12" font-weight="bold">cap (100%)</text>')
    parts.append(f'<polyline fill="none" stroke="#1f5fbf" stroke-width="2" points="{pts}"/>')
    for u in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<text x="{pad_l - 6}" y="{py(u) + 4:.1f}" font-size="11" text-anchor="end">{int(u * 100)}%</text>')
    for d in np.linspace(d0, d1, 7):
        parts.append(f'<text x="{px(d):.1f}" y="{height - 8}" font-size="11" text-anchor="middle">d{int(d)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def html_report(report: CapacityReport) -> str:
    """Self-contained HTML: inline CSS, inline SVG, no scripts, no external URLs."""
    rows = "".join(
        f"<tr><td>{html.escape(c.category)}</td><td>{_gbf(c.bytes):.2f}</td><td>{pct(c.share)}</td>"
        f"<td>{_mbf(c.slope_bytes_per_day):.1f}</td><td>{_mbf(c.p90_bytes_per_day):.1f}</td></tr>"
        for c in report.composition
    )
    alerts = "".join(
        f"<li class='{'bad' if a.crossed else 'ok'}'>{int(a.threshold * 100)}%: "
        + (f"crossed on day {a.first_day}" if a.crossed else f"projected in {_days(a.projected_days)}")
        + "</li>"
        for a in report.alerts
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>The scary line - day {report.day}</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;color:#222}}
h1{{font-size:20px}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px 10px;text-align:right}}
td:first-child,th:first-child{{text-align:left}} .bad{{color:#b22222;font-weight:bold}} .ok{{color:#2e7d32}}
code{{font-size:18px}}
</style></head><body>
<h1>The scary line: {pct(report.utilisation)} of {_gb(report.cap_bytes)} on day {report.day} ({html.escape(report.date)})</h1>
<p>Growth {_mb(report.slope_bytes_per_day)}/day (Theil-Sen, {report.window}-day window), p90 {_mb(report.p90_bytes_per_day)}/day.
<b>Days to cap: {_days(report.days_to_cap)}</b> (p90: {_days(report.days_to_cap_p90)}).</p>
<p><code>{html.escape(report.sparkline)}</code></p>
{svg_chart(report.history_days, report.history_utilisation)}
<h2>Composition</h2>
<table><tr><th>category</th><th>GB</th><th>share</th><th>growth MB/day</th><th>p90 MB/day</th></tr>{rows}</table>
<p>Biggest table: <b>{html.escape(report.biggest_table.name)}</b> ({html.escape(report.biggest_category)}, {_gb(report.biggest_table.bytes)}).
If <b>{html.escape(report.biggest_category)}</b> moved out: utilisation {pct(report.what_if_moved['utilisation'])},
days to cap {_days(report.what_if_moved['days_to_cap'])}.</p>
<h2>Alerts</h2><ul>{alerts}</ul>
</body></html>"""


# ---------------------------------------------------------------- placement advisor


@dataclass
class Question:
    id: str
    text: str
    exactness: str  # exact | approximate
    retention: str  # hot | warm | cold
    query: str  # point | aggregate | distinct | scan
    write_rate: str  # low | medium | high

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Question:
        needs = d.get("needs", d)
        q = cls(
            id=str(d["id"]),
            text=str(d.get("text", d["id"])),
            exactness=str(needs.get("exactness", "exact")),
            retention=str(needs.get("retention", "hot")),
            query=str(needs.get("query", "point")),
            write_rate=str(needs.get("write_rate", "low")),
        )
        if q.exactness not in ("exact", "approximate"):
            raise ValueError(f"{q.id}: exactness must be exact|approximate")
        if q.retention not in ("hot", "warm", "cold"):
            raise ValueError(f"{q.id}: retention must be hot|warm|cold")
        if q.write_rate not in ("low", "medium", "high"):
            raise ValueError(f"{q.id}: write_rate must be low|medium|high")
        return q


@dataclass
class Placement:
    question_id: str
    store_class: str
    reason: str


def recommend_placement(question: Question | dict[str, Any]) -> Placement:
    """Decision table (first rule wins); see module docstring."""
    q = question if isinstance(question, Question) else Question.from_dict(question)
    if q.retention == "cold":
        return Placement(q.id, "cold_archive", "long retention, rarely read: keep it out of the hot store")
    if q.exactness == "exact" and q.write_rate == "high":
        return Placement(q.id, "exact_counter",
                         "exact answer over a high write rate: a dedicated exact counter (HLL-free, no sampling)")
    if q.exactness == "approximate":
        return Placement(q.id, "analytics_sampled", "approximate is fine: sampled analytics store is cheapest")
    return Placement(q.id, "transactional", "exact, low write rate: the platform's transactional store")


# severity of serving a question that *should* live in `recommended` from `actual`
SEVERITY: dict[tuple[str, str], tuple[str, str]] = {
    ("exact_counter", "analytics_sampled"): ("critical", "exact question answered from a sampled store: counts will be wrong"),
    ("exact_counter", "cold_archive"): ("warning", "exact counter served from cold archive: too slow for a live answer"),
    ("exact_counter", "transactional"): ("warning", "high write rate into the transactional store: this is the closet"),
    ("transactional", "analytics_sampled"): ("critical", "transactional question answered from a sampled store"),
    ("transactional", "cold_archive"): ("critical", "hot transactional question served from cold archive"),
    ("transactional", "exact_counter"): ("info", "counter store used for a point lookup"),
    ("analytics_sampled", "transactional"): ("warning", "approximate telemetry stored in the transactional store: eats capacity"),
    ("analytics_sampled", "exact_counter"): ("info", "exact counter where sampled would do: over-provisioned"),
    ("analytics_sampled", "cold_archive"): ("warning", "live approximate query served from cold archive"),
    ("cold_archive", "transactional"): ("warning", "cold data in the hot transactional store: eats capacity"),
    ("cold_archive", "analytics_sampled"): ("critical", "archive served from a sampled store: records are lost"),
    ("cold_archive", "exact_counter"): ("warning", "archive served from a counter store: no row-level history"),
}


@dataclass
class Misplacement:
    question_id: str
    table: str
    actual: str
    recommended: str
    severity: str
    why: str


def load_questions(path: str | Path) -> list[Question]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [Question.from_dict(q) for q in doc.get("questions", doc if isinstance(doc, list) else [])]


def load_layout(path: str | Path) -> list[dict[str, Any]]:
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    tables = doc.get("tables", doc if isinstance(doc, list) else [])
    for t in tables:
        if t.get("store_class") not in STORE_CLASSES:
            raise ValueError(f"table {t.get('name')}: store_class must be one of {STORE_CLASSES}")
    return tables


def audit_layout(questions: Sequence[Question], layout: Sequence[dict[str, Any]]) -> list[Misplacement]:
    """For every (table, question) pair in the layout compare actual vs recommended store class."""
    by_id = {q.id: q for q in questions}
    out: list[Misplacement] = []
    for t in layout:
        for qid in t.get("serves", []):
            if qid not in by_id:
                raise ValueError(f"layout table {t['name']} serves unknown question {qid!r}")
            rec = recommend_placement(by_id[qid])
            actual = t["store_class"]
            if actual == rec.store_class:
                continue
            sev, why = SEVERITY.get((rec.store_class, actual), ("info", "store class differs from recommendation"))
            out.append(Misplacement(qid, t["name"], actual, rec.store_class, sev, why))
    order = {"critical": 0, "warning": 1, "info": 2}
    out.sort(key=lambda m: (order[m.severity], m.question_id))
    return out


def unserved_questions(questions: Sequence[Question], layout: Sequence[dict[str, Any]]) -> list[str]:
    served = {q for t in layout for q in t.get("serves", [])}
    return [q.id for q in questions if q.id not in served]


def placement_markdown(questions: Sequence[Question], layout: Sequence[dict[str, Any]],
                       misplacements: Sequence[Misplacement]) -> str:
    recs = [recommend_placement(q) for q in questions]
    lines = [
        "# Placement advisor",
        "",
        md_table(["question", "exactness", "retention", "write rate", "recommended store"],
                 [[q.id, q.exactness, q.retention, q.write_rate, r.store_class] for q, r in zip(questions, recs)]),
        "",
        f"## Misplacements in current layout ({len(misplacements)})",
        "",
    ]
    if misplacements:
        lines.append(md_table(["severity", "question", "table", "actual", "recommended", "why"],
                              [[m.severity, m.question_id, m.table, m.actual, m.recommended, m.why] for m in misplacements]))
    else:
        lines.append("none")
    un = unserved_questions(questions, layout)
    if un:
        lines += ["", "Unserved questions: " + ", ".join(un)]
    return "\n".join(lines)


# ---------------------------------------------------------------- sampling audit


def generate_population(users: int, seed: int = 0, skew: float = 1.5, mean_events: float = 3.0) -> np.ndarray:
    """Return an array of user ids, one per event. Events per user ~ 1 + Pareto-ish tail.

    skew controls the tail: with the Zipf-like draw `1 + floor(pareto(skew) * (mean_events - 1))`
    most users produce one or two events and a few produce hundreds.
    """
    rng = np.random.default_rng(seed)
    per_user = 1 + np.floor(rng.pareto(skew, size=users) * max(mean_events - 1, 0.0)).astype(int)
    return np.repeat(np.arange(users), per_user)


def sampling_undercount(population_events: Sequence[int], sample_rate: float, seed: int = 0) -> dict[str, float]:
    """Simulate one sampled store over `population_events` (user id per event).

    Returns exact vs sampled distinct users (the sampled count is what a dashboard reads
    straight off the sampled table), the undercount factor exact/sampled, and the scaled
    event-count estimate sampled_events / rate with its relative error.
    """
    if not 0 < sample_rate <= 1:
        raise ValueError("sample_rate must be in (0, 1]")
    ev = np.asarray(population_events)
    rng = np.random.default_rng(seed)
    keep = rng.random(ev.size) < sample_rate
    exact_users = int(np.unique(ev).size)
    sampled_users = int(np.unique(ev[keep]).size)
    exact_events = int(ev.size)
    est_events = float(keep.sum() / sample_rate)
    return {
        "sample_rate": sample_rate,
        "exact_users": exact_users,
        "sampled_users": sampled_users,
        "undercount_factor": exact_users / max(sampled_users, 1),
        "users_missed_share": 1 - sampled_users / max(exact_users, 1),
        "exact_events": exact_events,
        "est_events": est_events,
        "event_rel_error": (est_events - exact_events) / max(exact_events, 1),
    }


def expected_undercount(population_events: Sequence[int], sample_rate: float) -> float:
    """Closed form: share of users missed = mean over users of (1 - r)^k_user."""
    counts = np.asarray(list(Counter(np.asarray(population_events).tolist()).values()), float)
    return float(np.mean((1 - sample_rate) ** counts))


def sampling_sweep(rates: Sequence[float], users: int = 10_000, seed: int = 0, skew: float = 1.5,
                   n_rep: int = 10) -> list[dict[str, Any]]:
    """Repeat the simulation n_rep times per rate; bootstrap_ci over the repetitions."""
    pop = generate_population(users, seed, skew)
    out = []
    for r in rates:
        runs = [sampling_undercount(pop, r, seed=seed * 1000 + k) for k in range(n_rep)]
        uc = [x["undercount_factor"] for x in runs]
        ee = [x["event_rel_error"] for x in runs]
        uc_pt, uc_lo, uc_hi = bootstrap_ci(uc, n_boot=500, seed=seed)
        ee_pt, ee_lo, ee_hi = bootstrap_ci(ee, n_boot=500, seed=seed)
        out.append({
            "sample_rate": r,
            "exact_users": runs[0]["exact_users"],
            "sampled_users_mean": float(np.mean([x["sampled_users"] for x in runs])),
            "undercount_factor": uc_pt, "undercount_ci": [uc_lo, uc_hi],
            "expected_missed_share": expected_undercount(pop, r),
            "event_rel_error": ee_pt, "event_rel_error_ci": [ee_lo, ee_hi],
        })
    return out


def sampling_markdown(sweep: Sequence[dict[str, Any]], users: int, skew: float) -> str:
    rows = [
        [s["sample_rate"], s["exact_users"], f"{s['sampled_users_mean']:.0f}",
         f"{s['undercount_factor']:.2f}x [{s['undercount_ci'][0]:.2f}, {s['undercount_ci'][1]:.2f}]",
         pct(s["expected_missed_share"]),
         f"{s['event_rel_error'] * 100:+.2f}% [{s['event_rel_error_ci'][0] * 100:+.2f}, {s['event_rel_error_ci'][1] * 100:+.2f}]"]
        for s in sweep
    ]
    return "\n".join([
        f"# Sampling audit: {users:,} users, event skew {skew}",
        "",
        md_table(["rate", "exact users", "sampled users", "undercount (bootstrap 95%)", "expected missed", "event estimate error (95%)"], rows),
        "",
        ("Counting *events* from the sampled store and scaling by 1/rate is unbiased. Counting *people* from it "
        "is not: a user with k events survives sampling with probability 1 - (1 - r)^k, so the long tail of "
        "one-event users vanishes and nothing scales it back."),
    ])


# ---------------------------------------------------------------- formatting helpers


def _jsonable(x: float | None) -> float | None:
    if x is None:
        return None
    return None if math.isinf(x) or math.isnan(x) else float(x)


def _gb(b: float) -> str:
    return f"{b / GB:.2f} GB"


def _gbf(b: float) -> float:
    return b / GB


def _mb(b: float) -> str:
    return f"{b / MB:.1f} MB"


def _mbf(b: float) -> float:
    return b / MB


def _days(d: float | None) -> str:
    if d is None or math.isinf(d):
        return "never (not growing)"
    return f"{d:.0f} days"
