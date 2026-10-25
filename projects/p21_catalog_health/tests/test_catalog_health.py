"""Tests prove the README claims: long-tail shares, Gini/Pareto, duplicate clusters, the
dormancy rule, >90% hidden by scoping while featured and team items survive, rising detection,
and one retire-queue row per dormant item."""

import json
from datetime import timedelta
from pathlib import Path

import pytest

from evalkit.io import read_jsonl
from projects.p21_catalog_health import core
from projects.p21_catalog_health.cli import main

DATA = Path(__file__).resolve().parents[1] / "data"
TODAY = core.AS_OF


def _item(i: int, name: str = "x", team: str = "sales", users: int = 0, inv: int | None = None, stars: int = 0,
          idle: int | None = 0, inv7: int = 0, prev7: int = 0, tags=None, users_by_team=None, kind="app") -> core.CatalogItem:
    return core.CatalogItem(
        id=f"i{i}", name=name, owner=f"o{i}@x", team=team, kind=kind, created=TODAY - timedelta(days=400),
        last_used=None if idle is None else TODAY - timedelta(days=idle), invocations_30d=users * 3 if inv is None else inv,
        distinct_users_30d=users, stars=stars, tags=tags or [], invocations_7d=inv7, invocations_prev_7d=prev7,
        users_by_team=users_by_team if users_by_team is not None else ({team: users} if users else {}),
    )


@pytest.fixture(scope="module")
def catalog():
    return core.load_catalog(DATA / "catalog.jsonl")


@pytest.fixture(scope="module")
def featured():
    return core.load_featured(DATA / "featured.yaml")


# ---------------------------------------------------------------- long tail


def test_long_tail_shares_on_hand_built_catalog():
    items = [_item(i, users=u) for i, u in enumerate([0, 0, 0, 1, 1, 2, 5, 6, 100, 400])]
    lt = core.long_tail_stats(items)
    assert lt["share_0_users"] == pytest.approx(0.3)
    assert lt["share_le1_users"] == pytest.approx(0.5)
    assert lt["share_le5_users"] == pytest.approx(0.7)
    assert lt["n_items"] == 10


def test_gini_and_pareto_numbers():
    assert core.gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert core.gini([0, 0, 0, 10]) == pytest.approx(0.75)  # (n-1)/n for one item holding everything
    assert core.gini([]) == 0.0 and core.gini([0, 0]) == 0.0
    usage = [0] * 90 + [1] * 9 + [91]  # 100 items; the top 1% carries 91/100
    assert core.pareto_share(usage, 0.01) == pytest.approx(0.91)
    assert core.pareto_share(usage, 0.1) == pytest.approx(1.0)
    assert core.items_for_share(usage, 0.8) == pytest.approx(0.01)
    assert core.items_for_share([1, 1, 1, 1, 1], 0.8) == pytest.approx(0.8)


def test_fixture_is_a_long_tail(catalog):
    lt = core.long_tail_stats(catalog)
    assert len(catalog) == 2000
    assert lt["share_le1_users"] >= 0.88
    assert lt["gini_users"] > 0.9
    assert lt["top10pct_items_usage_share"] > 0.9
    assert lt["items_share_for_80pct_usage"] < 0.05


# ---------------------------------------------------------------- duplicates


def test_duplicate_clusters_normalise_names():
    for a, b in [("todo app v2", "to-do"), ("To Do List", "TODO (copy)"), ("hello world 7", "Hello-World 42"),
                 ("My First Agent", "first agent (copy)"), ("Chat Bot", "chat-bot")]:
        assert core.normalise_name(a) == core.normalise_name(b), (a, b)
    assert core.normalise_name("Sales Deal Desk") != core.normalise_name("Sales Dashboard")
    items = [_item(0, "todo app v2"), _item(1, "to-do"), _item(2, "Todo App"), _item(3, "Expense Bot"),
             _item(4, "hello world 3"), _item(5, "hello world 99")]
    clusters = core.duplicate_clusters(items)
    assert [c["key"] for c in clusters] == ["todo", "helloworld"]
    assert clusters[0]["size"] == 3 and set(clusters[0]["ids"]) == {"i0", "i1", "i2"}


def test_fixture_duplicates(catalog):
    clusters = {c["key"]: c for c in core.duplicate_clusters(catalog)}
    assert clusters["todo"]["size"] > 100 and clusters["helloworld"]["size"] > 100
    assert "Expense Bot" not in {n for c in clusters.values() for n in c["names"]}


# ---------------------------------------------------------------- dormancy


def test_dormancy_rule_respects_stars_and_window():
    rule = core.DormancyRule(days=60, min_stars=1)
    assert rule.is_dormant(_item(0, idle=61), TODAY)
    assert rule.is_dormant(_item(1, idle=None), TODAY)  # never used
    assert not rule.is_dormant(_item(2, idle=59), TODAY)
    assert not rule.is_dormant(_item(3, idle=400, stars=1), TODAY)  # starred: never auto-retired
    assert rule.is_dormant(_item(4, idle=60), TODAY)  # boundary: >= days
    assert not core.DormancyRule(days=500).is_dormant(_item(5, idle=400), TODAY)
    # the window moves with `today`
    later = TODAY + timedelta(days=30)
    assert rule.is_dormant(_item(6, idle=40), later)


def test_retire_queue_one_row_per_dormant_item_with_owner(tmp_path, catalog):
    rule = core.DormancyRule(days=60)
    rows = core.retirement_queue(catalog, rule, TODAY)
    expected = [i for i in catalog if rule.is_dormant(i, TODAY)]
    assert len(rows) == len(expected) > 0
    assert {r["id"] for r in rows} == {i.id for i in expected}
    assert all(r["owner"].endswith("@example.com") and r["owner"] in r["notification"] for r in rows)
    assert all(r["stars"] == 0 for r in rows)
    out = tmp_path / "retire_queue.jsonl"
    assert main(["dormant", "--catalog", str(DATA / "catalog.jsonl"), "--days", "60", "--out", str(out)]) == 0
    written = read_jsonl(out)
    assert len(written) == len(rows) and written[0]["id"] == rows[0]["id"]


# ---------------------------------------------------------------- scoping


def test_scoped_view_shrinks_by_over_90pct_and_keeps_featured_and_team_items(catalog, featured):
    for team in ("sales", "support", "legal"):
        view = core.scoped_view(catalog, core.UserContext(team=team, role="manager"), featured)
        assert view.before == 2000
        assert view.reduction > 0.9, (team, view.after)
        ids = [v["id"] for v in view.visible]
        for fid in featured.org + featured.teams[team]:
            assert fid in ids
        team_used = [i.id for i in catalog if i.users_by_team.get(team, 0) >= 2]
        assert set(team_used) <= set(ids)
        # ranking: org-featured items first, and every visible item has a reason
        assert view.visible[0]["id"] in set(featured.org) | set(featured.teams[team])
        assert all(v["reasons"] for v in view.visible)
        # a dead to-do app owned by the team is hidden
        dead = [i.id for i in catalog if i.team == team and i.distinct_users_30d == 0]
        assert not set(dead) & set(ids)


def test_scoped_view_hand_built():
    items = [
        _item(0, "Hub", team="hr", users=500, users_by_team={"hr": 300, "sales": 40}),
        _item(1, "Sales tool", team="sales", users=5, users_by_team={"sales": 5}),
        _item(2, "Dead sales todo", team="sales", users=0),
        _item(3, "Single-user sales thing", team="sales", users=1, users_by_team={"sales": 1}),
        _item(4, "Manager digest", team="ops", users=3, tags=["manager"], users_by_team={"ops": 3}),
        _item(5, "Other team thing", team="ops", users=3, users_by_team={"ops": 3}),
    ]
    feat = core.Featured(org=["i0"], teams={"sales": ["i5"]})
    view = core.scoped_view(items, core.UserContext("sales", "manager"), feat, top_global=1)
    ids = [v["id"] for v in view.visible]
    assert ids[0] == "i0" and set(ids) == {"i0", "i1", "i3", "i4", "i5"}
    assert "i2" not in ids
    by = {v["id"]: v["reasons"] for v in view.visible}
    assert "team-featured" in by["i5"] and "role" in by["i4"] and "team-owned" in by["i3"] and "team-used" in by["i1"]


# ---------------------------------------------------------------- rising / featured


def test_rising_detection():
    items = [_item(0, "steady", inv7=100, prev7=95), _item(1, "rising", inv7=40, prev7=5),
             _item(2, "tiny", inv7=4, prev7=0), _item(3, "new", inv7=20, prev7=0), _item(4, "falling", inv7=5, prev7=50)]
    rising = core.rising_items(items, min_inv=10, min_growth=1.0)
    assert [r["id"] for r in rising] == ["i3", "i1"]
    assert rising[1]["wow_growth"] == pytest.approx(7.0)


def test_featured_and_usage_ranking(catalog, featured):
    assert len(featured.org) == 6 and featured.for_team("sales")
    top = core.rank_usage(catalog, 3)
    assert [t.id for t in top] == ["item-0000", "item-0001", "item-0002"]
    assert core.load_featured(None).org == []


# ---------------------------------------------------------------- CLI


def test_cli_report_view_duplicates_json(tmp_path):
    assert main(["report", "--catalog", str(DATA / "catalog.jsonl"), "--json", str(tmp_path / "r.json")]) == 0
    payload = json.loads((tmp_path / "r.json").read_text())
    assert payload["n_items"] == 2000 and payload["long_tail"]["share_le1_users"] > 0.85
    assert main(["report", "--max-untouched", "0.5"]) == 1  # gate: too much of the catalog untouched
    assert main(["view", "--team", "sales", "--role", "manager", "--json", str(tmp_path / "v.json")]) == 0
    v = json.loads((tmp_path / "v.json").read_text())
    assert v["after"] < 200 < v["before"]
    assert main(["duplicates", "--json", str(tmp_path / "d.json")]) == 0
    assert main(["view", "--team", "sales", "--catalog", str(tmp_path / "missing.jsonl")]) == 2
    assert main([]) == 2
