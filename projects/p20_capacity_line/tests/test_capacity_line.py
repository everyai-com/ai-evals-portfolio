"""Tests prove the README claims: days-to-cap, composition, the migration cliff, placement
misplacements, the sampling undercount, and a self-contained HTML report."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from projects.p20_capacity_line import core
from projects.p20_capacity_line.cli import main

DATA = Path(__file__).resolve().parents[1] / "data"
GB = core.GB


def _snap(day: int, tables: dict[str, tuple[str, int]], cap: int = 10 * GB) -> core.StorageSnapshot:
    return core.StorageSnapshot(
        day, f"d{day}", cap, [core.TableStat(n, c, b // 100, b) for n, (c, b) in tables.items()]
    )


def _linear_series(days: int, start: int, per_day: int, cap: int = 10 * GB):
    return [_snap(d, {"t": ("platform", start + per_day * d)}, cap) for d in range(days)]


@pytest.fixture(scope="module")
def snapshots():
    return core.load_snapshots(DATA / "snapshots.jsonl")


# ---------------------------------------------------------------- growth / days-to-cap


def test_days_to_cap_on_hand_built_linear_series():
    # 2 GB + 100 MB/day, cap 10 GB: at day 10 the store holds 2048 + 1000 MB, 7192 MB remain
    snaps = _linear_series(11, 2 * GB, 100 * core.MB)
    rep = core.CapacityLine(window=5).analyse(snaps)
    assert rep.slope_bytes_per_day == pytest.approx(100 * core.MB)
    assert rep.p90_bytes_per_day == pytest.approx(100 * core.MB)
    assert rep.days_to_cap == pytest.approx(7192 / 100)
    assert rep.days_to_cap_p90 == pytest.approx(7192 / 100)
    assert rep.utilisation == pytest.approx(3048 / 10240)


def test_theil_sen_ignores_a_spike_and_flat_series_never_reaches_cap():
    xs = list(range(10))
    ys = [10.0 + 2 * x for x in xs]
    ys[5] = 500.0  # one-off spike
    assert core.theil_sen(xs, ys) == pytest.approx(2.0)
    flat = [_snap(d, {"t": ("platform", 1 * GB)}) for d in range(6)]
    rep = core.CapacityLine(window=6).analyse(flat)
    assert math.isinf(rep.days_to_cap)
    assert rep.to_dict()["days_to_cap"] is None  # JSON-safe


def test_days_to_cap_at_or_over_cap_is_zero():
    assert core.days_to_cap(11 * GB, 10 * GB, 1.0) == 0.0


# ---------------------------------------------------------------- composition


def test_composition_and_biggest_contributor_and_what_if():
    tables = {
        "users": ("platform", 1 * GB),
        "events": ("analytics", 5 * GB),
        "versions": ("versions", 2 * GB),
    }
    snaps = [_snap(d, {k: (c, b + (d * 10 * core.MB if c == "analytics" else 0)) for k, (c, b) in tables.items()}) for d in range(5)]
    rep = core.CapacityLine(window=5).analyse(snaps)
    shares = {c.category: c.share for c in rep.composition}
    total = 8 * GB + 40 * core.MB
    assert shares["analytics"] == pytest.approx((5 * GB + 40 * core.MB) / total)
    assert rep.composition[0].category == "analytics"  # sorted by bytes desc
    assert rep.biggest_table.name == "events" and rep.biggest_category == "analytics"
    assert rep.fastest_category == "analytics"
    assert rep.what_if_moved["utilisation"] == pytest.approx(0.3)
    assert math.isinf(rep.what_if_moved["days_to_cap"])  # nothing else grows


def test_alerts_report_first_crossing_day_and_projection():
    # 4096 MB + 200 MB/day: 50% (5120 MB) crossed on day 6 (5.12), 75% (7680 MB) on day 18 (17.92)
    snaps = _linear_series(20, 4 * GB, 200 * core.MB)
    rep = core.CapacityLine(window=5, thresholds=(0.5, 0.75, 0.9)).analyse(snaps)
    a50, a75, a90 = rep.alerts
    assert a50.crossed and a50.first_day == 6
    assert a75.crossed and a75.first_day == 18
    assert not a90.crossed and a90.projected_days == pytest.approx((9 * GB - (4 * GB + 19 * 200 * core.MB)) / (200 * core.MB))


# ---------------------------------------------------------------- the fixture story


def test_migration_at_day_60_drops_the_line(snapshots):
    by_day = {s.day: s for s in snapshots}
    assert len(snapshots) == 90
    before, after = by_day[59], by_day[60]
    assert before.utilisation > 0.8
    assert after.utilisation < before.utilisation - 0.3
    # analytics was about half the closet before the move
    share_before = before.by_category()["analytics"] / before.total_bytes
    assert 0.45 <= share_before <= 0.6
    assert after.by_category()["analytics"] / after.total_bytes < 0.05
    rep_before = core.CapacityLine().analyse(snapshots[:60])
    rep_after = core.CapacityLine().analyse(snapshots)
    assert rep_before.fastest_category == "analytics" and rep_before.biggest_category == "analytics"
    assert rep_before.days_to_cap < 40
    assert rep_after.days_to_cap > rep_before.days_to_cap * 3
    assert rep_before.what_if_moved["utilisation"] < 0.5
    assert [a.crossed for a in rep_before.alerts] == [True, True, False]
    assert rep_before.sparkline[-1] in "▆▇█" and rep_after.sparkline[-1] in "▃▄▅"


# ---------------------------------------------------------------- placement


def test_recommend_placement_decision_table():
    rec = lambda **k: core.recommend_placement({"id": "q", **k}).store_class
    assert rec(exactness="exact", retention="hot", write_rate="high", query="distinct") == "exact_counter"
    assert rec(exactness="approximate", retention="warm", write_rate="high") == "analytics_sampled"
    assert rec(exactness="exact", retention="cold", write_rate="low") == "cold_archive"
    assert rec(exactness="exact", retention="hot", write_rate="low") == "transactional"
    with pytest.raises(ValueError):
        core.recommend_placement({"id": "bad", "exactness": "sorta"})


def test_placement_flags_planted_misplacements():
    qs = core.load_questions(DATA / "questions.yaml")
    layout = core.load_layout(DATA / "layout.yaml")
    mis = {m.question_id: m for m in core.audit_layout(qs, layout)}
    assert mis["distinct_users_month"].severity == "critical"
    assert mis["distinct_users_month"].actual == "analytics_sampled"
    assert mis["distinct_users_month"].recommended == "exact_counter"
    assert mis["every_version_shipped"].severity == "warning"
    assert (mis["every_version_shipped"].actual, mis["every_version_shipped"].recommended) == ("transactional", "cold_archive")
    assert "user_profile" not in mis and "deploys_per_app_today" not in mis
    assert core.unserved_questions(qs, layout) == []
    assert main(["placement", "--questions", str(DATA / "questions.yaml"), "--layout", str(DATA / "layout.yaml")]) == 1


# ---------------------------------------------------------------- sampling audit


def test_sampling_undercount_grows_as_rate_falls_and_events_stay_unbiased():
    pop = core.generate_population(5000, seed=1, skew=1.5)
    results = [core.sampling_undercount(pop, r, seed=3) for r in (0.5, 0.2, 0.1, 0.05)]
    factors = [r["undercount_factor"] for r in results]
    assert factors == sorted(factors) and factors[0] > 1.2 and factors[-1] > 4
    for r in results:
        assert r["sampled_users"] < r["exact_users"]
        assert abs(r["event_rel_error"]) < 0.05  # scaled event count is unbiased
        assert r["users_missed_share"] == pytest.approx(core.expected_undercount(pop, r["sample_rate"]), abs=0.03)
    sweep = core.sampling_sweep([0.1], users=2000, seed=0, n_rep=5)
    lo, hi = sweep[0]["undercount_ci"]
    assert lo <= sweep[0]["undercount_factor"] <= hi
    assert abs(sweep[0]["event_rel_error"]) < 0.05 and sweep[0]["event_rel_error_ci"][0] <= sweep[0]["event_rel_error"] <= sweep[0]["event_rel_error_ci"][1]
    with pytest.raises(ValueError):
        core.sampling_undercount(pop, 0.0)


def test_event_count_mean_is_unbiased_over_many_seeds():
    pop = core.generate_population(1000, seed=0)
    errs = [core.sampling_undercount(pop, 0.1, seed=s)["event_rel_error"] for s in range(40)]
    assert abs(float(np.mean(errs))) < 0.01


# ---------------------------------------------------------------- outputs


def test_html_report_is_self_contained(tmp_path, snapshots):
    out = tmp_path / "r.html"
    rc = main(["analyse", "--snapshots", str(DATA / "snapshots.jsonl"), "--cap-gb", "10", "--html", str(out),
               "--json", str(tmp_path / "r.json")])
    assert rc == 0
    html = out.read_text()
    assert "<svg" in html and "</svg>" in html and "<polyline" in html
    for bad in ("http://", "https://", "<script", "<link", "src=", "href="):
        assert bad not in html
    payload = json.loads((tmp_path / "r.json").read_text())
    assert payload["utilisation"] == pytest.approx(snapshots[-1].utilisation)
    assert payload["days_to_cap"] > 0


def test_cli_gate_and_usage_errors(tmp_path):
    # day 59 is above 90%? no (85%), but above --max-util 0.8 -> blocked
    assert main(["analyse", "--until-day", "59", "--max-util", "0.8"]) == 1
    assert main(["analyse", "--until-day", "59"]) == 0
    assert main(["analyse", "--snapshots", str(tmp_path / "missing.jsonl")]) == 2
    assert main(["sampling", "--rate", "0.1", "--users", "500", "--seed", "0", "--reps", "2", "--rates", "0.5"]) == 0
    assert main(["nope"]) == 2
