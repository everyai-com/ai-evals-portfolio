from collections import Counter

from projects.p09_drift_monitor.core import TrafficSampler
from projects.p09_drift_monitor.synth import make_day, make_traffic


def test_sampler_rate_and_determinism():
    rows = make_day(0, per_day=1000, seed=0)
    s = TrafficSampler(rate=0.05, seed=0, stratify_by="hour")
    a = s.sample(rows, "2026-10-01")
    assert len(a) == 50
    b = TrafficSampler(rate=0.05, seed=0, stratify_by="hour").sample(rows, "2026-10-01")
    assert [r["task_id"] for r in a] == [r["task_id"] for r in b]
    c = TrafficSampler(rate=0.05, seed=1).sample(rows, "2026-10-01")
    assert [r["task_id"] for r in a] != [r["task_id"] for r in c]
    # same seed, different date -> different draw (a nightly job must not resample the same ids)
    d = s.sample(rows, "2026-10-02")
    assert [r["task_id"] for r in a] != [r["task_id"] for r in d]


def test_sampler_is_stratified_by_hour_and_tenant():
    rows = make_day(0, per_day=2000, seed=1)
    out = TrafficSampler(rate=0.10, seed=0, stratify_by="hour").sample(rows, "d")
    pop = Counter(r["meta"]["hour"] for r in rows)
    got = Counter(r["meta"]["hour"] for r in out)
    for h, n in pop.items():
        assert abs(got[h] - 0.10 * n) <= 1  # quota per stratum within rounding
    out_t = TrafficSampler(rate=0.10, seed=0, stratify_by="tenant").sample(rows, "d")
    pop_t = Counter(r["meta"]["tenant"] for r in rows)
    got_t = Counter(r["meta"]["tenant"] for r in out_t)
    for t, n in pop_t.items():
        assert abs(got_t[t] - 0.10 * n) <= 1


def test_feedback_weighting_oversamples_thumbs_down():
    rows = make_day(12, drift_start_day=10, per_day=2000, seed=0)  # drifted day -> plenty of thumbs-down
    base_rate = sum(r["meta"].get("feedback") == "down" for r in rows) / len(rows)
    uniform = TrafficSampler(rate=0.10, seed=0).sample(rows, "d")
    weighted = TrafficSampler(rate=0.10, seed=0, feedback_weight=8.0).sample(rows, "d")
    u_rate = sum(r["meta"].get("feedback") == "down" for r in uniform) / len(uniform)
    w_rate = sum(r["meta"].get("feedback") == "down" for r in weighted) / len(weighted)
    assert abs(u_rate - base_rate) < 0.08
    assert w_rate > 2 * base_rate and w_rate > u_rate + 0.15
    assert len(weighted) == len(uniform) == 200


def test_make_traffic_shape_and_determinism():
    t1 = make_traffic(days=3, drift_start_day=2, seed=0, per_day=20)
    t2 = make_traffic(days=3, drift_start_day=2, seed=0, per_day=20)
    assert list(t1) == ["2026-10-01", "2026-10-02", "2026-10-03"]
    assert t1 == t2 and all(len(v) == 20 for v in t1.values())
    assert make_traffic(days=1, seed=1, per_day=20) != make_traffic(days=1, seed=0, per_day=20)
