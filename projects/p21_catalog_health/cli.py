"""CLI for catalog health.

    report     --catalog data/catalog.jsonl [--featured data/featured.yaml] [--days 60] [--json [out.json]]
               [--max-untouched 0.95]   exit 1 when the share of items with <= 1 user exceeds it
    dormant    --catalog ... --days 60 [--min-stars 1] --out retire_queue.jsonl [--json [out.json]]
    view       --catalog ... --team sales [--role manager] [--featured ...] [--limit 20] [--json [out.json]]
    duplicates --catalog ... [--min-size 2] [--json [out.json]]

Exit codes: 0 ok, 1 gate blocked, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from evalkit.report import md_table

from . import core

HERE = Path(__file__).parent


def _emit(markdown: str, payload: object, json_opt: str | None) -> None:
    if json_opt == "-":
        print(json.dumps(payload, indent=2, default=str))
        return
    print(markdown)
    if json_opt:
        Path(json_opt).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\njson written to {json_opt}")


def _load(a: argparse.Namespace):
    items = core.load_catalog(a.catalog)
    featured = core.load_featured(getattr(a, "featured", None))
    today = date.fromisoformat(a.today) if a.today else core.AS_OF
    return items, featured, today


def _cmd_report(a: argparse.Namespace) -> int:
    try:
        items, featured, today = _load(a)
        rep = core.health_report(items, featured, core.DormancyRule(a.days, a.min_stars), today)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    _emit(rep.to_markdown(), rep.to_dict(), a.json)
    return 1 if rep.long_tail["share_le1_users"] > a.max_untouched else 0


def _cmd_dormant(a: argparse.Namespace) -> int:
    try:
        items, _, today = _load(a)
        rows = core.retirement_queue(items, core.DormancyRule(a.days, a.min_stars), today)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    md = (f"# Retirement queue: {len(rows)} of {len(items)} items idle >= {a.days} days with < {a.min_stars} star(s)\n\n"
          + md_table(["id", "name", "owner", "team", "last used", "idle days"],
                     [[r["id"], r["name"], r["owner"], r["team"], r["last_used"] or "never", r["days_idle"] if r["days_idle"] is not None else "-"]
                      for r in rows[:10]])
          + (f"\n... {len(rows) - 10} more" if len(rows) > 10 else ""))
    if a.out:
        n = core.write_retire_queue(rows, a.out)
        md += f"\n\nqueue written: {n} rows -> {a.out}"
    _emit(md, rows, a.json)
    return 0


def _cmd_view(a: argparse.Namespace) -> int:
    try:
        items, featured, today = _load(a)
        view = core.scoped_view(items, core.UserContext(a.team, a.role or ""), featured, a.top_global, a.min_team_users, today)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    payload = {"team": a.team, "role": a.role, "before": view.before, "after": view.after,
               "reduction": view.reduction, "visible": view.visible}
    _emit(view.to_markdown(a.limit), payload, a.json)
    return 0


def _cmd_duplicates(a: argparse.Namespace) -> int:
    try:
        items = core.load_catalog(a.catalog)
        clusters = core.duplicate_clusters(items, a.min_size)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    md = (f"# Duplicate-name clusters: {len(clusters)} clusters, {sum(c['size'] for c in clusters)} items\n\n"
          + md_table(["normalised", "items", "users", "teams", "names"],
                     [[c["key"], c["size"], c["total_users"], len(c["teams"]), "; ".join(c["names"][:6])] for c in clusters[:15]]))
    _emit(md, clusters, a.json)
    return 0


def _common(p: argparse.ArgumentParser, featured: bool = True) -> None:
    p.add_argument("--catalog", default=str(HERE / "data" / "catalog.jsonl"))
    if featured:
        p.add_argument("--featured", default=str(HERE / "data" / "featured.yaml"))
    p.add_argument("--today", default=None, help="ISO date; default 2026-10-24 (the fixture's as-of)")
    p.add_argument("--json", nargs="?", const="-", default=None)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="evalkit catalog", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="long tail, duplicates, dormancy, top, rising, scoping")
    _common(r)
    r.add_argument("--days", type=int, default=60)
    r.add_argument("--min-stars", type=int, default=1)
    r.add_argument("--max-untouched", type=float, default=1.0, help="exit 1 if share of <=1-user items exceeds this")
    r.set_defaults(fn=_cmd_report)
    d = sub.add_parser("dormant", help="retirement queue with owner notifications")
    _common(d, featured=False)
    d.add_argument("--days", type=int, default=60)
    d.add_argument("--min-stars", type=int, default=1)
    d.add_argument("--out", default=None, help="write retire_queue.jsonl here")
    d.set_defaults(fn=_cmd_dormant)
    v = sub.add_parser("view", help="what one user sees after scoping")
    _common(v)
    v.add_argument("--team", required=True)
    v.add_argument("--role", default=None)
    v.add_argument("--top-global", type=int, default=10)
    v.add_argument("--min-team-users", type=int, default=2)
    v.add_argument("--limit", type=int, default=20)
    v.set_defaults(fn=_cmd_view)
    u = sub.add_parser("duplicates", help="duplicate-name clusters")
    _common(u, featured=False)
    u.add_argument("--min-size", type=int, default=2)
    u.set_defaults(fn=_cmd_duplicates)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        a = build_parser().parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
