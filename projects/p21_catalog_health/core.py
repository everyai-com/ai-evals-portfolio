"""Marketplace / registry health: long tail, duplicates, dormancy, scoping and curation.

Statistics (all over `distinct_users_30d` unless stated)
-------------------------------------------------------
long tail        share of items with 0 / <= 1 / <= 5 distinct users in the last 30 days
Gini             G = (2 * sum_i i * x_(i)) / (n * sum x) - (n + 1) / n over the ascending sorted
                 values x_(1..n); 0 = every item used equally, -> 1 = one item carries everything
Pareto           share of total usage carried by the top `top_frac` of items, and the smallest
                 share of items that carries `target` (80%) of usage
duplicates       items grouped by a normalised name: lowercase, "to-do"/"to do" -> "todo",
                 "(copy)", "v2", bare numbers, "beta", "test", "app", "agent", "list" tokens stripped,
                 non-alphanumerics dropped; a cluster is any normalised key with >= 2 members
dormancy         an item is dormant when it has not been used for >= `days` days (or never) AND
                 has fewer than `min_stars` stars (stars are a human "keep this" signal)
scoping          "isolate what you see to what you need": an item is visible to a user when it
                 is org-featured, featured for their team, owned by their team AND used by someone
                 on it this month, used by >= `min_team_users` people on their team in the last
                 30 days, tagged with their role, or in the global top-K by distinct users.
                 Ownership alone earns nothing: a dead to-do app is hidden from its own team.
                 Visible items are ranked by
                 score = 1000*featured_org + 800*featured_team + 100*team_users
                       + 20*ln(1 + distinct_users) + 10*stars + 50*role_match + 5*used_this_week
rising           week-over-week growth g = (inv_7d - inv_prev_7d) / max(inv_prev_7d, 1), reported
                 when inv_7d >= min_inv and g >= min_growth
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evalkit.io import read_jsonl, write_jsonl
from evalkit.report import md_table, pct

AS_OF = date(2026, 10, 24)
KINDS = ("app", "agent", "mcp_server", "automation")


# ---------------------------------------------------------------- items


@dataclass
class CatalogItem:
    id: str
    name: str
    owner: str
    team: str
    kind: str
    created: date
    last_used: date | None
    invocations_30d: int
    distinct_users_30d: int
    stars: int = 0
    versions: int = 1
    tags: list[str] = field(default_factory=list)
    risk_tier: str = "low"
    invocations_7d: int = 0
    invocations_prev_7d: int = 0
    users_by_team: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CatalogItem:
        if d.get("kind") not in KINDS:
            raise ValueError(f"{d.get('id')}: kind must be one of {KINDS}, got {d.get('kind')!r}")
        return cls(
            id=str(d["id"]), name=str(d["name"]), owner=str(d.get("owner", "")), team=str(d.get("team", "")),
            kind=str(d["kind"]), created=_parse_date(d.get("created")) or AS_OF,
            last_used=_parse_date(d.get("last_used")),
            invocations_30d=int(d.get("invocations_30d", 0)), distinct_users_30d=int(d.get("distinct_users_30d", 0)),
            stars=int(d.get("stars", 0)), versions=int(d.get("versions", 1)), tags=list(d.get("tags", [])),
            risk_tier=str(d.get("risk_tier", "low")), invocations_7d=int(d.get("invocations_7d", 0)),
            invocations_prev_7d=int(d.get("invocations_prev_7d", 0)),
            users_by_team={str(k): int(v) for k, v in (d.get("users_by_team") or {}).items()},
        )

    def days_idle(self, today: date = AS_OF) -> int | None:
        return None if self.last_used is None else (today - self.last_used).days


def _parse_date(v: Any) -> date | None:
    if v in (None, ""):
        return None
    return v if isinstance(v, date) else date.fromisoformat(str(v))


def load_catalog(path: str | Path) -> list[CatalogItem]:
    items = [CatalogItem.from_dict(r) for r in read_jsonl(path)]
    ids = [i.id for i in items]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate item ids")
    return items


# ---------------------------------------------------------------- long tail


def gini(values: Sequence[float]) -> float:
    """Gini coefficient of non-negative values (0 equal .. 1 concentrated)."""
    x = np.sort(np.asarray(values, float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return float(2 * (i * x).sum() / (n * x.sum()) - (n + 1) / n)


def pareto_share(values: Sequence[float], top_frac: float = 0.1) -> float:
    """Share of the total carried by the top `top_frac` of items."""
    x = np.sort(np.asarray(values, float))[::-1]
    if x.size == 0 or x.sum() == 0:
        return 0.0
    k = max(1, math.ceil(top_frac * x.size))
    return float(x[:k].sum() / x.sum())


def items_for_share(values: Sequence[float], target: float = 0.8) -> float:
    """Smallest fraction of items (largest first) whose usage sums to >= target of the total."""
    x = np.sort(np.asarray(values, float))[::-1]
    if x.size == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x) / x.sum()
    k = int(np.searchsorted(cum, target - 1e-12) + 1)
    return min(1.0, k / x.size)


def long_tail_stats(items: Sequence[CatalogItem]) -> dict[str, float]:
    users = [i.distinct_users_30d for i in items]
    inv = [i.invocations_30d for i in items]
    n = max(1, len(items))
    return {
        "n_items": len(items),
        "share_0_users": sum(u == 0 for u in users) / n,
        "share_le1_users": sum(u <= 1 for u in users) / n,
        "share_le5_users": sum(u <= 5 for u in users) / n,
        "gini_users": gini(users),
        "gini_invocations": gini(inv),
        "top10pct_items_usage_share": pareto_share(inv, 0.1),
        "top1pct_items_usage_share": pareto_share(inv, 0.01),
        "items_share_for_80pct_usage": items_for_share(inv, 0.8),
    }


# ---------------------------------------------------------------- duplicates

_SUFFIXES = re.compile(r"\b(copy|v\d+|\d+|beta|alpha|test|app|agent|bot|new|my|final|draft|list)\b")


def normalise_name(name: str) -> str:
    s = name.lower()
    s = re.sub(r"to[\s\-_]*do", "todo", s)
    s = re.sub(r"hello[\s\-_]*world", "helloworld", s)
    s = s.replace("(", " ").replace(")", " ")
    s = _SUFFIXES.sub(" ", s)
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def duplicate_clusters(items: Sequence[CatalogItem], min_size: int = 2) -> list[dict[str, Any]]:
    groups: dict[str, list[CatalogItem]] = defaultdict(list)
    for it in items:
        key = normalise_name(it.name)
        if key:
            groups[key].append(it)
    out = []
    for key, members in groups.items():
        if len(members) >= min_size:
            out.append({
                "key": key,
                "size": len(members),
                "names": sorted({m.name for m in members}),
                "ids": [m.id for m in members],
                "total_users": sum(m.distinct_users_30d for m in members),
                "teams": sorted({m.team for m in members}),
            })
    out.sort(key=lambda c: (-c["size"], c["key"]))
    return out


# ---------------------------------------------------------------- dormancy


@dataclass
class DormancyRule:
    days: int = 60
    min_stars: int = 1  # items with at least this many stars are never retired automatically

    def is_dormant(self, item: CatalogItem, today: date = AS_OF) -> bool:
        if item.stars >= self.min_stars:
            return False
        idle = item.days_idle(today)
        return idle is None or idle >= self.days

    def reason(self, item: CatalogItem, today: date = AS_OF) -> str:
        idle = item.days_idle(today)
        return "never used" if idle is None else f"idle {idle} days (>= {self.days})"


def retirement_queue(items: Sequence[CatalogItem], rule: DormancyRule, today: date = AS_OF) -> list[dict[str, Any]]:
    """One row per dormant item with the owner notification text."""
    rows = []
    for it in items:
        if not rule.is_dormant(it, today):
            continue
        rows.append({
            "id": it.id, "name": it.name, "kind": it.kind, "owner": it.owner, "team": it.team,
            "last_used": it.last_used.isoformat() if it.last_used else None,
            "days_idle": it.days_idle(today), "stars": it.stars, "reason": rule.reason(it, today),
            "notification": (
                f"Hi {it.owner}: your {it.kind} '{it.name}' ({it.id}) is {rule.reason(it, today)} and has no stars. "
                f"It will be archived in 14 days unless you use it or star it."
            ),
        })
    rows.sort(key=lambda r: (-(r["days_idle"] if r["days_idle"] is not None else 10**9), r["id"]))
    return rows


# ---------------------------------------------------------------- featured / curation


@dataclass
class Featured:
    org: list[str] = field(default_factory=list)
    teams: dict[str, list[str]] = field(default_factory=dict)

    def for_team(self, team: str) -> set[str]:
        return set(self.teams.get(team, []))


def load_featured(path: str | Path | None) -> Featured:
    if path is None or not Path(path).exists():
        return Featured()
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Featured(org=[str(x) for x in doc.get("org", [])],
                    teams={str(t): [str(x) for x in ids or []] for t, ids in (doc.get("teams") or {}).items()})


def rank_usage(items: Sequence[CatalogItem], n: int = 10) -> list[CatalogItem]:
    """Usage-weighted ranking: distinct users first, then invocations, then stars."""
    return sorted(items, key=lambda i: (-i.distinct_users_30d, -i.invocations_30d, -i.stars, i.id))[:n]


def rising_items(items: Sequence[CatalogItem], min_inv: int = 10, min_growth: float = 1.0, n: int = 10) -> list[dict[str, Any]]:
    out = []
    for it in items:
        if it.invocations_7d < min_inv:
            continue
        g = (it.invocations_7d - it.invocations_prev_7d) / max(it.invocations_prev_7d, 1)
        if g >= min_growth:
            out.append({"id": it.id, "name": it.name, "team": it.team, "invocations_7d": it.invocations_7d,
                        "invocations_prev_7d": it.invocations_prev_7d, "wow_growth": g})
    out.sort(key=lambda r: (-r["wow_growth"], -r["invocations_7d"], r["id"]))
    return out[:n]


# ---------------------------------------------------------------- scoping


@dataclass
class UserContext:
    team: str
    role: str = ""
    org: str = "default"


@dataclass
class ScopedView:
    user: UserContext
    before: int
    after: int
    visible: list[dict[str, Any]]

    @property
    def reduction(self) -> float:
        return 1 - self.after / self.before if self.before else 0.0

    def to_markdown(self, limit: int = 20) -> str:
        head = (f"# Scoped view for team={self.user.team} role={self.user.role or '-'}: "
                f"{self.before} items -> {self.after} ({pct(self.reduction)} hidden)")
        rows = [[v["rank"], v["id"], v["name"], v["kind"], v["team_users"], v["distinct_users"], v["stars"],
                 f"{v['score']:.0f}", ",".join(v["reasons"])] for v in self.visible[:limit]]
        return head + "\n\n" + md_table(["#", "id", "name", "kind", "team users", "users", "stars", "score", "why"], rows)


def scoped_view(items: Sequence[CatalogItem], user: UserContext, featured: Featured | None = None,
                top_global: int = 10, min_team_users: int = 2, today: date = AS_OF) -> ScopedView:
    featured = featured or Featured()
    org_ids, team_ids = set(featured.org), featured.for_team(user.team)
    top_ids = {i.id for i in rank_usage(items, top_global)}
    visible = []
    for it in items:
        team_users = it.users_by_team.get(user.team, 0)
        reasons = []
        if it.id in org_ids:
            reasons.append("org-featured")
        if it.id in team_ids:
            reasons.append("team-featured")
        if it.team == user.team and team_users > 0:
            reasons.append("team-owned")
        if team_users >= min_team_users:
            reasons.append("team-used")
        if user.role and user.role in it.tags:
            reasons.append("role")
        if it.id in top_ids:
            reasons.append("top-global")
        if not reasons:
            continue
        idle = it.days_idle(today)
        score = (1000 * ("org-featured" in reasons) + 800 * ("team-featured" in reasons) + 100 * team_users
                 + 20 * math.log1p(it.distinct_users_30d) + 10 * it.stars + 50 * ("role" in reasons)
                 + 5 * (idle is not None and idle <= 7))
        visible.append({"id": it.id, "name": it.name, "kind": it.kind, "team": it.team, "team_users": team_users,
                        "distinct_users": it.distinct_users_30d, "stars": it.stars, "score": score, "reasons": reasons})
    visible.sort(key=lambda v: (-v["score"], v["id"]))
    for r, v in enumerate(visible, 1):
        v["rank"] = r
    return ScopedView(user, len(items), len(visible), visible)


# ---------------------------------------------------------------- health report


@dataclass
class HealthReport:
    as_of: str
    n_items: int
    by_kind: dict[str, int]
    long_tail: dict[str, float]
    duplicates: list[dict[str, Any]]
    dormancy: DormancyRule
    n_dormant: int
    dormant_share: float
    top: list[dict[str, Any]]
    rising: list[dict[str, Any]]
    scoping: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["dormancy"] = asdict(self.dormancy)
        return d

    def to_markdown(self) -> str:
        lt = self.long_tail
        lines = [
            f"# Catalog health: {self.n_items:,} items as of {self.as_of}",
            "",
            md_table(["kind", "items"], [[k, v] for k, v in sorted(self.by_kind.items())]),
            "",
            "## Long tail (distinct users, last 30 days)",
            "",
            md_table(["stat", "value"], [
                ["items with 0 users", pct(lt["share_0_users"])],
                ["items with <= 1 user", pct(lt["share_le1_users"])],
                ["items with <= 5 users", pct(lt["share_le5_users"])],
                ["Gini (users)", f"{lt['gini_users']:.3f}"],
                ["Gini (invocations)", f"{lt['gini_invocations']:.3f}"],
                ["top 1% of items carry", pct(lt["top1pct_items_usage_share"]) + " of invocations"],
                ["top 10% of items carry", pct(lt["top10pct_items_usage_share"]) + " of invocations"],
                ["items needed for 80% of usage", pct(lt["items_share_for_80pct_usage"])],
            ]),
            "",
            f"## Duplicate-name clusters ({len(self.duplicates)}; {sum(c['size'] for c in self.duplicates):,} items)",
            "",
            md_table(["normalised", "items", "users", "names"],
                     [[c["key"], c["size"], c["total_users"], "; ".join(c["names"][:5]) + (" ..." if len(c["names"]) > 5 else "")]
                      for c in self.duplicates[:8]]),
            "",
            f"## Dormancy: {self.n_dormant:,} items ({pct(self.dormant_share)}) idle >= {self.dormancy.days} days with < {self.dormancy.min_stars} star(s)",
            "",
            "## Top by usage",
            "",
            md_table(["id", "name", "team", "users", "invocations", "stars"],
                     [[t["id"], t["name"], t["team"], t["distinct_users"], t["invocations"], t["stars"]] for t in self.top]),
            "",
            "## Rising (week over week)",
            "",
            md_table(["id", "name", "team", "7d", "prev 7d", "growth"],
                     [[r["id"], r["name"], r["team"], r["invocations_7d"], r["invocations_prev_7d"], f"{r['wow_growth']:+.0%}"]
                      for r in self.rising]) if self.rising else "none",
            "",
            "## What each team sees before vs after scoping",
            "",
            md_table(["team", "before", "after", "hidden"],
                     [[s["team"], s["before"], s["after"], pct(s["reduction"])] for s in self.scoping]),
        ]
        return "\n".join(lines)


def health_report(items: Sequence[CatalogItem], featured: Featured | None = None,
                  rule: DormancyRule | None = None, today: date = AS_OF) -> HealthReport:
    rule = rule or DormancyRule()
    queue = retirement_queue(items, rule, today)
    by_kind: dict[str, int] = defaultdict(int)
    for it in items:
        by_kind[it.kind] += 1
    teams = sorted({it.team for it in items})
    scoping = []
    for t in teams:
        v = scoped_view(items, UserContext(team=t), featured, today=today)
        scoping.append({"team": t, "before": v.before, "after": v.after, "reduction": v.reduction})
    return HealthReport(
        as_of=today.isoformat(), n_items=len(items), by_kind=dict(by_kind), long_tail=long_tail_stats(items),
        duplicates=duplicate_clusters(items), dormancy=rule, n_dormant=len(queue),
        dormant_share=len(queue) / max(1, len(items)),
        top=[{"id": i.id, "name": i.name, "team": i.team, "distinct_users": i.distinct_users_30d,
              "invocations": i.invocations_30d, "stars": i.stars} for i in rank_usage(items, 10)],
        rising=rising_items(items), scoping=scoping,
    )


def write_retire_queue(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    return write_jsonl(path, rows)
