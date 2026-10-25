"""Usage metering, limits, anomaly detection and chargeback for an agent platform.

Rules and statistics (so the README claims are checkable)
--------------------------------------------------------
Pricing      model_call -> evalkit.llm.estimate_cost(model, input_tokens, output_tokens);
             other kinds -> prices.yaml[kind] * units. Events arriving with a cost keep it.
Dedup        Meter.ingest drops any event whose event_id was already ingested (at-least-once
             delivery from the gateway must not double-bill).
Hard caps    per-request max tokens / units; per-agent and per-user daily spend caps (UTC day):
             block iff spent_today + estimated_cost > cap; retry_after_s = seconds to midnight UTC.
Token bucket rate limit per agent: capacity = burst, refill = requests_per_minute / 60 per second,
             one token per allowed request; when empty, retry_after_s = (1 - tokens) / refill.
Budgets      per-tenant monthly budget is soft: the request is allowed, but crossing 80 % / 100 %
             raises an alert once per (tenant, month, threshold).
Anomalies    For each user and each agent on day D: hourly spend x_h for h in 0..23 is compared
             to a baseline of non-zero hourly spends over the previous `baseline_days` days.
             The baseline is the *population* of entities of the same type (every user, or every
             agent), not the entity alone: a user whose entire history is a 06:00 burst would
             otherwise hide inside their own baseline. With m = median and MAD = median(|x - m|),
             the robust sigma is 1.4826 * MAD (consistent with sigma for a normal). Hour h is
             anomalous iff
                 x_h > m + mad_k * 1.4826 * MAD  and  x_h - m >= min_abs_usd.
             Recurrence: the entity's own spend at the same hour-of-day exceeded that threshold
             on k of the n baseline days. If the Wilson 95 % lower bound
             (evalkit.metrics.wilson_ci) of k/n is >= recurrence_floor the label is
             `scheduled_burst` ("thousands of instances every morning"), otherwise `spike`
             (one-off). With fewer than `min_history` non-zero baseline hours in the population
             there is no verdict.
Chargeback   share = spend / total; budget consumption = spent_month / monthly_budget;
             cost per successful outcome = sum(cost of events carrying `success`) /
             count(success == True).
"""
from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evalkit.io import read_jsonl, write_jsonl
from evalkit.llm import estimate_cost
from evalkit.metrics import wilson_ci
from evalkit.report import md_table, pct

KINDS = ("model_call", "tool_call", "container_seconds", "ocr_page")


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)


def fmt_ts(d: datetime) -> str:
    return d.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- events

@dataclass
class UsageEvent:
    event_id: str
    ts: str                 # ISO-8601 UTC, e.g. 2026-10-24T06:12:00Z
    tenant: str
    team: str
    cost_centre: str
    user: str
    agent_id: str
    kind: str
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    units: float = 0.0
    cost_usd: float | None = None
    success: bool | None = None

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"event {self.event_id}: unknown kind {self.kind!r}")

    @property
    def day(self) -> str:
        return self.ts[:10]

    @property
    def hour(self) -> str:
        return self.ts[:13]

    @property
    def month(self) -> str:
        return self.ts[:7]

    @property
    def hour_of_day(self) -> int:
        return int(self.ts[11:13])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_prices(path: str | Path) -> dict[str, float]:
    return {k: float(v) for k, v in (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).items()}


def price_event(ev: UsageEvent, prices: dict[str, float]) -> float:
    """model_call -> evalkit price table; everything else -> prices[kind] * units."""
    if ev.kind == "model_call":
        return estimate_cost(ev.model or "mock", int(ev.input_tokens), int(ev.output_tokens))
    if ev.kind not in prices:
        raise KeyError(f"no price for kind {ev.kind!r}")
    return prices[ev.kind] * float(ev.units)


@dataclass
class Agg:
    cost: float = 0.0
    events: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    units: float = 0.0
    outcomes: int = 0       # events carrying a success flag
    successes: int = 0
    outcome_cost: float = 0.0   # cost of the events carrying a success flag

    def add(self, ev: UsageEvent) -> None:
        self.cost += ev.cost_usd or 0.0
        self.events += 1
        self.input_tokens += ev.input_tokens
        self.output_tokens += ev.output_tokens
        self.units += ev.units
        if ev.success is not None:
            self.outcomes += 1
            self.successes += int(bool(ev.success))
            self.outcome_cost += ev.cost_usd or 0.0

    @property
    def cost_per_success(self) -> float | None:
        """Cost of outcome-bearing events per successful outcome (events without a flag excluded)."""
        return None if self.successes == 0 else self.outcome_cost / self.successes


@dataclass
class IngestResult:
    added: int
    duplicates: int
    rejected: int
    cost_added: float


class Meter:
    """In-memory ledger of priced, deduplicated events with grouped aggregates."""

    def __init__(self, prices: dict[str, float] | None = None):
        self.prices = prices or {}
        self.events: list[UsageEvent] = []
        self.seen: set[str] = set()

    def ingest(self, rows: Iterable[UsageEvent | dict[str, Any]]) -> IngestResult:
        added = dup = rejected = 0
        cost = 0.0
        for row in rows:
            try:
                ev = row if isinstance(row, UsageEvent) else UsageEvent(**{k: v for k, v in row.items()
                                                                           if k in UsageEvent.__dataclass_fields__})
            except (TypeError, ValueError):
                rejected += 1
                continue
            if ev.event_id in self.seen:
                dup += 1
                continue
            if ev.cost_usd is None:
                ev.cost_usd = price_event(ev, self.prices)
            self.seen.add(ev.event_id)
            self.events.append(ev)
            added += 1
            cost += ev.cost_usd
        return IngestResult(added, dup, rejected, cost)

    def aggregate(self, *keys: str, where: Callable[[UsageEvent], bool] | None = None) -> dict[tuple, Agg]:
        """Group by any of: tenant team cost_centre user agent_id kind model day hour month."""
        out: dict[tuple, Agg] = defaultdict(Agg)
        for ev in self.events:
            if where and not where(ev):
                continue
            out[tuple(getattr(ev, k) for k in keys)].add(ev)
        return dict(out)

    def spend(self, **filters: Any) -> float:
        total = 0.0
        for ev in self.events:
            if all(getattr(ev, k) == v for k, v in filters.items()):
                total += ev.cost_usd or 0.0
        return total

    def total(self) -> float:
        return sum(ev.cost_usd or 0.0 for ev in self.events)

    def days(self) -> list[str]:
        return sorted({ev.day for ev in self.events})

    # persistence
    def save(self, state_dir: str | Path) -> Path:
        p = Path(state_dir) / "events.jsonl"
        write_jsonl(p, (ev.to_dict() for ev in self.events))
        return p

    @classmethod
    def load(cls, state_dir: str | Path, prices: dict[str, float] | None = None) -> Meter:
        m = cls(prices)
        p = Path(state_dir) / "events.jsonl"
        if p.exists():
            m.ingest(read_jsonl(p))
        return m


# --------------------------------------------------------------------------- limits

@dataclass
class Limits:
    max_tokens: int = 16_000
    max_units: float = 200.0
    requests_per_minute: float = 60.0
    burst: int = 20
    agents: dict[str, dict[str, float]] = field(default_factory=dict)
    users: dict[str, dict[str, float]] = field(default_factory=dict)
    tenants: dict[str, dict[str, float]] = field(default_factory=dict)
    budget_thresholds: tuple[float, ...] = (0.8, 1.0)
    anomaly: dict[str, float] = field(default_factory=lambda: {"baseline_days": 14, "mad_k": 6.0,
                                                               "min_abs_usd": 0.5, "recurrence_floor": 0.5})

    @classmethod
    def from_yaml(cls, path: str | Path) -> Limits:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        pr, rl = cfg.get("per_request", {}), cfg.get("rate_limit", {})
        an = dict(cls().anomaly)
        an.update(cfg.get("anomaly", {}))
        return cls(max_tokens=int(pr.get("max_tokens", 16_000)), max_units=float(pr.get("max_units", 200)),
                   requests_per_minute=float(rl.get("requests_per_minute", 60)), burst=int(rl.get("burst", 20)),
                   agents=cfg.get("agents", {}), users=cfg.get("users", {}), tenants=cfg.get("tenants", {}),
                   budget_thresholds=tuple(cfg.get("alerts", {}).get("budget_thresholds", [0.8, 1.0])),
                   anomaly=an)

    def _lookup(self, table: dict, key: str, fld: str) -> float | None:
        v = table.get(key, table.get("default", {})).get(fld)
        return None if v is None else float(v)

    def agent_daily_cap(self, agent: str) -> float | None:
        return self._lookup(self.agents, agent, "daily_cap_usd")

    def user_daily_cap(self, user: str) -> float | None:
        return self._lookup(self.users, user, "daily_cap_usd")

    def tenant_monthly_budget(self, tenant: str) -> float | None:
        return self._lookup(self.tenants, tenant, "monthly_budget_usd")


class TokenBucket:
    """Classic token bucket: `capacity` tokens, refilled at `rate` tokens/second."""

    def __init__(self, capacity: int, rate: float, tokens: float | None = None, last: float = 0.0):
        self.capacity = float(capacity)
        self.rate = float(rate)
        self.tokens = self.capacity if tokens is None else float(tokens)
        self.last = float(last)

    def _refill(self, now: float) -> None:
        if now > self.last:
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = max(self.last, now)

    def take(self, now: float) -> tuple[bool, float]:
        """Returns (allowed, retry_after_s)."""
        self._refill(now)
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True, 0.0
        return False, (1.0 - self.tokens) / self.rate if self.rate > 0 else math.inf

    def to_dict(self) -> dict[str, float]:
        return {"tokens": self.tokens, "last": self.last}


@dataclass
class Request:
    tenant: str
    user: str
    agent_id: str
    kind: str = "model_call"
    model: str = "claude-sonnet-5-5"
    input_tokens: int = 0
    output_tokens: int = 0
    units: float = 0.0
    team: str = ""
    cost_centre: str = ""

    @property
    def tokens(self) -> int:
        return int(self.input_tokens) + int(self.output_tokens)


@dataclass
class Decision:
    allow: bool
    reason: str
    remaining: dict[str, float | None]
    retry_after_s: float | None = None
    warnings: list[str] = field(default_factory=list)
    estimated_cost_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BudgetAlert:
    tenant: str
    month: str
    threshold: float
    spent: float
    budget: float
    at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def seconds_to_midnight(now: datetime) -> float:
    nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (nxt - now).total_seconds()


class Enforcer:
    """Evaluates a request against the meter's current state. Deterministic given `clock`."""

    def __init__(self, meter: Meter, limits: Limits, clock: Callable[[], datetime]):
        self.meter = meter
        self.limits = limits
        self.clock = clock
        self.buckets: dict[str, TokenBucket] = {}
        self.fired: set[tuple[str, str, float]] = set()
        self.alerts: list[BudgetAlert] = []

    def bucket(self, agent: str) -> TokenBucket:
        if agent not in self.buckets:
            self.buckets[agent] = TokenBucket(self.limits.burst, self.limits.requests_per_minute / 60.0)
        return self.buckets[agent]

    def estimate(self, req: Request) -> float:
        ev = UsageEvent("est", fmt_ts(self.clock()), req.tenant, req.team, req.cost_centre, req.user, req.agent_id,
                        req.kind, req.model, req.input_tokens, req.output_tokens, req.units)
        return price_event(ev, self.meter.prices)

    def decide(self, req: Request) -> Decision:
        now = self.clock()
        day, month = now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")
        est = self.estimate(req)
        lim = self.limits
        agent_cap, user_cap = lim.agent_daily_cap(req.agent_id), lim.user_daily_cap(req.user)
        budget = lim.tenant_monthly_budget(req.tenant)
        agent_spent = self.meter.spend(agent_id=req.agent_id, day=day)
        user_spent = self.meter.spend(user=req.user, day=day)
        tenant_spent = self.meter.spend(tenant=req.tenant, month=month)
        remaining = {
            "agent_daily_usd": None if agent_cap is None else max(0.0, agent_cap - agent_spent),
            "user_daily_usd": None if user_cap is None else max(0.0, user_cap - user_spent),
            "tenant_monthly_usd": None if budget is None else budget - tenant_spent,
            "bucket_tokens": self.bucket(req.agent_id).tokens,
        }
        warnings: list[str] = []

        def block(reason: str, retry: float | None = None) -> Decision:
            return Decision(False, reason, remaining, retry, warnings, est)

        # 1. per-request hard limits
        if req.kind == "model_call" and req.tokens > lim.max_tokens:
            return block(f"per_request.max_tokens: {req.tokens} > {lim.max_tokens}")
        if req.kind != "model_call" and req.units > lim.max_units:
            return block(f"per_request.max_units: {req.units} > {lim.max_units}")
        # 2. daily caps (hard)
        if agent_cap is not None and agent_spent + est > agent_cap:
            return block(f"agent_daily_cap: {req.agent_id} spent ${agent_spent:.2f} + ${est:.4f} > ${agent_cap:.2f}",
                         seconds_to_midnight(now))
        if user_cap is not None and user_spent + est > user_cap:
            return block(f"user_daily_cap: {req.user} spent ${user_spent:.2f} + ${est:.4f} > ${user_cap:.2f}",
                         seconds_to_midnight(now))
        # 3. token bucket (consumes a token only when the request is otherwise allowed)
        ok, retry = self.bucket(req.agent_id).take(now.timestamp())
        remaining["bucket_tokens"] = self.bucket(req.agent_id).tokens
        if not ok:
            return block(f"rate_limit: {req.agent_id} bucket empty ({lim.requests_per_minute:.0f}/min, burst {lim.burst})",
                         retry)
        # 4. soft monthly budget: warn once per threshold
        if budget is not None and budget > 0:
            ratio = (tenant_spent + est) / budget
            for thr in sorted(lim.budget_thresholds):
                key = (req.tenant, month, float(thr))
                if ratio >= thr and key not in self.fired:
                    self.fired.add(key)
                    self.alerts.append(BudgetAlert(req.tenant, month, float(thr), tenant_spent + est, budget, fmt_ts(now)))
                    warnings.append(f"budget_{int(thr * 100)}pct: {req.tenant} at {ratio * 100:.1f}% of ${budget:.0f} for {month}")
        return Decision(True, "ok", remaining, None, warnings, est)

    def record(self, req: Request, success: bool | None = None, event_id: str | None = None) -> UsageEvent:
        """Ingest the event a request would produce (used by tests / simulations)."""
        now = self.clock()
        eid = event_id or f"ev-{req.agent_id}-{req.user}-{len(self.meter.events)}"
        ev = UsageEvent(eid, fmt_ts(now), req.tenant, req.team, req.cost_centre, req.user, req.agent_id, req.kind,
                        req.model, req.input_tokens, req.output_tokens, req.units, None, success)
        self.meter.ingest([ev])
        return ev

    # persistence of the small mutable state (bucket levels, fired alerts)
    def to_state(self) -> dict[str, Any]:
        return {"buckets": {k: b.to_dict() for k, b in self.buckets.items()},
                "fired": [list(k) for k in sorted(self.fired)],
                "alerts": [a.to_dict() for a in self.alerts]}

    def load_state(self, st: dict[str, Any]) -> None:
        for k, b in st.get("buckets", {}).items():
            self.buckets[k] = TokenBucket(self.limits.burst, self.limits.requests_per_minute / 60.0, b["tokens"], b["last"])
        self.fired = {(t, m, float(th)) for t, m, th in st.get("fired", [])}
        self.alerts = [BudgetAlert(**a) for a in st.get("alerts", [])]


# --------------------------------------------------------------------------- anomalies

@dataclass
class Anomaly:
    entity_type: str        # user | agent_id
    entity: str
    day: str
    hour: int
    spend: float
    baseline_median: float
    threshold: float
    label: str              # spike | scheduled_burst
    recurrence_k: int
    recurrence_n: int
    wilson_lo: float
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _hourly(meter: Meter, key: str, value: str, days: Sequence[str]) -> np.ndarray:
    """Matrix [len(days), 24] of spend for one entity."""
    idx = {d: i for i, d in enumerate(days)}
    m = np.zeros((len(days), 24))
    for ev in meter.events:
        if getattr(ev, key) == value and ev.day in idx:
            m[idx[ev.day], ev.hour_of_day] += ev.cost_usd or 0.0
    return m


def detect_anomalies(meter: Meter, cfg: dict[str, float] | None = None, day: str | None = None,
                     min_history: int = 8) -> list[Anomaly]:
    """Median + MAD hourly spike detection per user and per agent for `day` (default: last day).

    Threshold comes from the population baseline (see module docstring); recurrence from the
    entity's own history at the same hour-of-day."""
    cfg = {**Limits().anomaly, **(cfg or {})}
    if not meter.events:
        return []
    day = day or meter.days()[-1]
    d0 = datetime.strptime(day, "%Y-%m-%d")
    n_base = int(cfg["baseline_days"])
    base_days = [(d0 - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(n_base, 0, -1)]
    out: list[Anomaly] = []
    for key in ("user", "agent_id"):
        entities = sorted({getattr(ev, key) for ev in meter.events if ev.day == day})
        history = {ent: _hourly(meter, key, ent, base_days)
                   for ent in sorted({getattr(ev, key) for ev in meter.events if ev.day in base_days} | set(entities))}
        population = np.concatenate([m[m > 0] for m in history.values()]) if history else np.zeros(0)
        if population.size < min_history:
            continue
        med = float(np.median(population))
        mad = float(np.median(np.abs(population - med)))
        sigma = 1.4826 * mad
        thr = max(med + cfg["mad_k"] * sigma, med + cfg["min_abs_usd"])
        for ent in entities:
            base = history[ent]
            today = _hourly(meter, key, ent, [day])[0]
            for h in range(24):
                x = float(today[h])
                if x <= thr:
                    continue
                k = int((base[:, h] > thr).sum())
                lo, _ = wilson_ci(k, n_base)
                label = "scheduled_burst" if lo >= cfg["recurrence_floor"] else "spike"
                out.append(Anomaly(key, ent, day, h, x, med, thr, label, k, n_base, lo,
                                   f"${x:.2f} at {h:02d}:00 vs baseline median ${med:.2f} (MAD ${mad:.2f}, "
                                   f"threshold ${thr:.2f}); same hour exceeded on {k}/{n_base} baseline days "
                                   f"(Wilson lo {lo:.2f})"))
    return out


# --------------------------------------------------------------------------- chargeback

def _top(agg: dict[tuple, Agg], n: int = 3) -> list[tuple[str, float]]:
    return [(k[0], a.cost) for k, a in sorted(agg.items(), key=lambda kv: -kv[1].cost)[:n]]


def chargeback(meter: Meter, limits: Limits, anomalies: Sequence[Anomaly] = (), month: str | None = None) -> dict[str, Any]:
    """Structured chargeback: per cost centre -> per team, plus tenant budgets. Totals equal the event sum."""
    evs = meter.events
    if month:
        evs = [e for e in evs if e.month == month]
    sub = Meter(meter.prices)
    sub.ingest(evs)
    total = sub.total()
    anom_by_user = defaultdict(list)
    anom_by_agent = defaultdict(list)
    for a in anomalies:
        (anom_by_user if a.entity_type == "user" else anom_by_agent)[a.entity].append(a)
    centres: dict[str, Any] = {}
    for (cc,), cagg in sorted(sub.aggregate("cost_centre").items()):
        teams: dict[str, Any] = {}
        for (team,), tagg in sorted(sub.aggregate("team", where=lambda e, cc=cc: e.cost_centre == cc).items()):
            w = lambda e, cc=cc, team=team: e.cost_centre == cc and e.team == team
            by_kind = {k[0]: a.cost for k, a in sub.aggregate("kind", where=w).items()}
            by_model = {k[0] or "-": a.cost for k, a in sub.aggregate("model", where=w).items() if k[0]}
            users = [u for (u,) in sub.aggregate("user", where=w)]
            agents = [a for (a,) in sub.aggregate("agent_id", where=w)]
            teams[team] = {
                "spend": tagg.cost, "events": tagg.events, "share": tagg.cost / total if total else 0.0,
                "by_kind": by_kind, "by_model": by_model,
                "top_agents": _top(sub.aggregate("agent_id", where=w)),
                "top_users": _top(sub.aggregate("user", where=w)),
                "outcomes": tagg.outcomes, "successes": tagg.successes, "cost_per_success": tagg.cost_per_success,
                "anomalies": [a.to_dict() for u in users for a in anom_by_user[u]]
                             + [a.to_dict() for g in agents for a in anom_by_agent[g]],
            }
        centres[cc] = {"spend": cagg.cost, "share": cagg.cost / total if total else 0.0, "events": cagg.events,
                       "cost_per_success": cagg.cost_per_success, "teams": teams}
    budgets: dict[str, Any] = {}
    for (tenant, m), agg in sorted(sub.aggregate("tenant", "month").items()):
        b = limits.tenant_monthly_budget(tenant)
        budgets[f"{tenant} {m}"] = {"tenant": tenant, "month": m, "spent": agg.cost, "budget": b,
                                    "consumed": (agg.cost / b) if b else None,
                                    "status": "over" if b and agg.cost > b else ("warn" if b and agg.cost >= 0.8 * b else "ok")}
    return {"month": month, "total": total, "events": len(evs), "cost_centres": centres, "budgets": budgets,
            "anomalies": [a.to_dict() for a in anomalies]}


def _usd(x: float | None) -> str:
    return "n/a" if x is None else f"${x:,.2f}"


def render_chargeback(report: dict[str, Any]) -> str:
    out = [f"# Chargeback report{(' — ' + report['month']) if report['month'] else ''}", "",
           f"Total: **{_usd(report['total'])}** across {report['events']} events, "
           f"{len(report['cost_centres'])} cost centres.", ""]
    out += ["## Cost centres", md_table(["cost centre", "spend", "share", "events", "cost / success"],
                                        [[cc, _usd(v["spend"]), pct(v["share"]), v["events"], _usd(v["cost_per_success"])]
                                         for cc, v in report["cost_centres"].items()]), ""]
    for cc, v in report["cost_centres"].items():
        out.append(f"### {cc}")
        out.append(md_table(["team", "spend", "share", "by kind", "by model", "top agents", "top users", "cost / success", "anomalies"],
                            [[team, _usd(t["spend"]), pct(t["share"]),
                              ", ".join(f"{k} {_usd(c)}" for k, c in sorted(t["by_kind"].items(), key=lambda kv: -kv[1])),
                              ", ".join(f"{k} {_usd(c)}" for k, c in sorted(t["by_model"].items(), key=lambda kv: -kv[1])) or "-",
                              ", ".join(f"{a} {_usd(c)}" for a, c in t["top_agents"]),
                              ", ".join(f"{u} {_usd(c)}" for u, c in t["top_users"]),
                              _usd(t["cost_per_success"]) + (f" ({t['successes']}/{t['outcomes']})" if t["outcomes"] else ""),
                              ", ".join(f"{a['label']}:{a['entity']}@{a['hour']:02d}h" for a in t["anomalies"]) or "-"]
                             for team, t in v["teams"].items()]))
        out.append("")
    out += ["## Tenant budgets", md_table(["tenant", "month", "spent", "budget", "consumed", "status"],
                                          [[b["tenant"], b["month"], _usd(b["spent"]), _usd(b["budget"]),
                                            pct(b["consumed"]) if b["consumed"] is not None else "n/a", b["status"]]
                                           for b in report["budgets"].values()]), ""]
    out += ["## Anomalies", md_table(["entity", "day", "hour", "spend", "baseline median", "threshold", "label", "recurrence"],
                                     [[f"{a['entity_type']}={a['entity']}", a["day"], f"{a['hour']:02d}:00", _usd(a["spend"]),
                                       _usd(a["baseline_median"]), _usd(a["threshold"]), a["label"],
                                       f"{a['recurrence_k']}/{a['recurrence_n']} (lo {a['wilson_lo']:.2f})"]
                                      for a in report["anomalies"]]) if report["anomalies"] else "_none_"]
    return "\n".join(out) + "\n"


def chargeback_csv_rows(meter: Meter, month: str | None = None) -> list[dict[str, Any]]:
    """One row per (cost_centre, team, kind, model); spend sums to the event total."""
    evs = [e for e in meter.events if not month or e.month == month]
    sub = Meter(meter.prices)
    sub.ingest(evs)
    total = sub.total()
    rows = []
    for (cc, team, kind, model), a in sorted(sub.aggregate("cost_centre", "team", "kind", "model").items(),
                                             key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], str(kv[0][3]))):
        rows.append({"cost_centre": cc, "team": team, "kind": kind, "model": model or "", "events": a.events,
                     "input_tokens": a.input_tokens, "output_tokens": a.output_tokens, "units": round(a.units, 3),
                     "spend_usd": round(a.cost, 6), "share": round(a.cost / total, 6) if total else 0.0,
                     "successes": a.successes, "outcomes": a.outcomes})
    return rows


def write_csv(rows: Sequence[dict[str, Any]], path: str | Path | None = None) -> str:
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    if path:
        Path(path).write_text(buf.getvalue(), encoding="utf-8")
    return buf.getvalue()


def invoice(meter: Meter, limits: Limits, tenant: str, month: str) -> dict[str, Any]:
    """Invoice-style statement: line items per (kind, model), subtotal, budget status."""
    w = lambda e: e.tenant == tenant and e.month == month
    lines = []
    for (kind, model), a in sorted(meter.aggregate("kind", "model", where=w).items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        qty = a.input_tokens + a.output_tokens if kind == "model_call" else a.units
        unit = "tokens" if kind == "model_call" else kind.replace("_", " ")
        lines.append({"item": f"{kind}" + (f" / {model}" if model else ""), "quantity": qty, "unit": unit,
                      "events": a.events, "amount": a.cost})
    subtotal = sum(line["amount"] for line in lines)
    budget = limits.tenant_monthly_budget(tenant)
    teams = {k[0]: a.cost for k, a in meter.aggregate("team", where=w).items()}
    return {"tenant": tenant, "month": month, "lines": lines, "subtotal": subtotal, "budget": budget,
            "consumed": subtotal / budget if budget else None, "by_team": teams}


def render_invoice(inv: dict[str, Any]) -> str:
    out = [f"# Statement — tenant `{inv['tenant']}` — {inv['month']}", "",
           md_table(["item", "quantity", "unit", "events", "amount"],
                    [[line["item"], f"{line['quantity']:,.0f}", line["unit"], line["events"], _usd(line["amount"])]
                     for line in inv["lines"]]), "",
           f"**Subtotal: {_usd(inv['subtotal'])}**" + (f" — {pct(inv['consumed'])} of ${inv['budget']:,.0f} monthly budget"
                                                      if inv["budget"] else ""), "",
           "Chargeback by team: " + ", ".join(f"{t} {_usd(c)}" for t, c in sorted(inv["by_team"].items(), key=lambda kv: -kv[1]))]
    return "\n".join(out) + "\n"


def dump_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, default=str)
