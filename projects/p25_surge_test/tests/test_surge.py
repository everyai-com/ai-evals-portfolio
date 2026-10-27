"""Virtual-clock tests: knee, spike with/without rate limiter, Little's law, headroom maths, step
monotonicity, determinism, HTML, thread-pool path. Whole file runs in a few seconds."""
from __future__ import annotations

import json
from pathlib import Path

from projects.p25_surge_test.cli import main
from projects.p25_surge_test.core import (
    Scenario,
    analyse,
    find_knee,
    headroom,
    littles_law,
    render_html,
    render_markdown,
    run_scenario,
)
from projects.p25_surge_test.targets import (
    FunctionTarget,
    RateLimiter,
    Request,
    Response,
    SimulatedTarget,
)

DATA = Path(__file__).resolve().parent.parent / "data"
# small twin of the demo target: knee at 30 / 0.05 s = 600 users at 1 req/s
SMALL = {"service_ms": 50, "service_cv": 0.3, "concurrency_limit": 30, "queue_limit": 30, "error_slope": 0.5,
             "cold_start_ms": 400, "cold_start_window_s": 1}


def _target(seed=0, **kw):
    return SimulatedTarget(**{**SMALL, **kw}, seed=seed)


def test_simulated_target_knee_near_configured_capacity():
    t = _target()
    nominal = t.nominal_capacity_users(1.0)
    assert abs(nominal - 30 / 0.055) < 1e-9        # 30 / (0.05 s x 1.1) ~= 545 users
    res = run_scenario(Scenario("ramp", users=1200, duration=40), t)
    an = analyse(res)
    assert an.knee_users is not None and 0.7 * nominal <= an.knee_users <= 1.3 * nominal, (an.knee_users, an.knee_reason)
    assert an.knee_second is not None and an.knee_second > 1          # cold-start seconds are not the knee
    assert 0 < an.max_sustainable_users < an.knee_users
    assert an.baseline_p95_ms < 100 and an.p95_max_ms >= 2 * an.baseline_p95_ms
    assert res.n_5xx > 0 and an.error_rate_max > 0.01
    # throughput plateaus: the last 10 seconds never exceed ~nominal rps
    assert max(p.throughput for p in res.series[-10:]) <= nominal * 1.1


def test_spike_errors_without_rate_limiter_and_429s_with_it():
    sc = Scenario("spike", users=2000, window=3, hold=3, join_burst=3)
    bare = run_scenario(sc, _target())
    an_bare = analyse(bare)
    assert bare.n_5xx / bare.n_requests > 0.05 and bare.n_429 == 0
    assert an_bare.error_rate_max > 0.01
    # cold start is a separate weak point (covered by the ramp test); here the claim is about shedding
    limited = RateLimiter(_target(cold_start_ms=0), per_user_rps=1.0, burst=1, global_rps=450)
    res = run_scenario(sc, limited)
    an = analyse(res)
    assert res.n_429 > 0 and limited.rejected == res.n_429 and limited.rejected_global > 0
    assert res.n_5xx / res.n_requests <= 0.01 and an.error_rate_max <= 0.01
    # p95 stays bounded: the target never queues, so no doubling vs the bare run's baseline
    assert an.p95_max_ms < 2 * an_bare.baseline_p95_ms or an.p95_max_ms < 150
    assert an.rate_limited_max > 0.5                              # most of the spike was shed as 429
    assert res.n_ok >= 0.9 * 450 * sc.hold                         # the target still served ~global_rps in the hold
    assert an.knee_users is None or an.knee_reason.startswith("no knee") or an.knee_users > 1000


def test_rate_limiter_bucket_maths():
    seen = []
    rl = RateLimiter(FunctionTarget(lambda r: seen.append(r.seq) or 200), per_user_rps=1.0, burst=2)
    out = [rl.call(Request(7, t, "/", i)).status for i, t in enumerate([0.0, 0.1, 0.2, 0.9, 1.25, 1.3])]
    assert out == [200, 200, 429, 429, 200, 429] and seen == [0, 1, 4]
    assert rl.call(Request(8, 1.3, "/", 9)).status == 200          # another user has its own bucket
    g = RateLimiter(FunctionTarget(lambda r: 200), per_user_rps=100, burst=100, global_rps=2, global_burst=2)
    assert [g.call(Request(i, 0.0, "/", i)).status for i in range(4)] == [200, 200, 429, 429]


def test_littles_law_check_within_tolerance():
    res = run_scenario(Scenario("soak", users=300, duration=12), _target())   # rho = 0.5, well below the knee
    an = analyse(res)
    assert an.littles_median_rel_error is not None and an.littles_median_rel_error < 0.1 and an.littles_law_ok
    # L = lambda W by hand on one steady second: 300 req/s x ~0.05 s ~= 15 in flight
    p = res.series[6]
    assert abs(p.offered * p.mean_latency_ms / 1000 - p.in_flight) / p.in_flight < 0.1
    assert 10 < p.in_flight < 25
    assert littles_law([]) is None


def test_headroom_statement_math():
    h = headroom(10_000, 3_200, 10_000)
    assert h.ratio == 3.125 and h.short and h.statement.endswith("harness shows knee at 3,200 -> 3.1x short")
    assert h.statement.startswith("projected 10,000 users at launch demo")
    h2 = headroom(2_000, 3_000, 5_000)
    assert not h2.short and "1.5x headroom" in h2.statement
    h3 = headroom(1_000, None, 5_000)
    assert not h3.short and "no knee up to 5,000" in h3.statement and "at least 5.0x headroom" in h3.statement
    h4 = headroom(10_000, None, 5_000)
    assert h4.short and "untested above 5,000" in h4.statement


def test_step_scenario_monotonic_load_and_levels():
    sc = Scenario("step", users=8000, start_users=500, step_seconds=2)
    assert sc.levels() == [500, 1000, 2000, 4000, 8000] and sc.total_seconds == 10
    users = [sc.users_at(s) for s in range(sc.total_seconds)]
    assert users == sorted(users) and users[0] == 500 and users[-1] == 8000
    res = run_scenario(Scenario("step", users=800, start_users=100, step_seconds=2), _target())
    offered = [p.offered for p in res.series]
    assert offered == sorted(offered) and offered[0] == 100 and offered[-1] == 800
    assert [p.users for p in res.series] == [100, 100, 200, 200, 400, 400, 800, 800]
    ramp = Scenario("ramp", users=100, duration=10)
    assert [ramp.users_at(s) for s in (0, 4, 9)] == [10, 50, 100]
    spike = Scenario("spike", users=100, window=4, hold=2, join_burst=2)
    assert [spike.users_at(s) for s in range(6)] == [25, 50, 75, 100, 100, 100]
    assert len(spike.schedule()) == sum(spike.users_at(s) for s in range(6)) + 100 * 2


def test_seeded_determinism():
    sc = Scenario("ramp", users=400, duration=8, seed=3)
    a = run_scenario(sc, _target(seed=3)).to_dict()
    b = run_scenario(sc, _target(seed=3)).to_dict()
    a.pop("wall_seconds"), b.pop("wall_seconds")
    assert a == b
    c = run_scenario(Scenario("ramp", users=400, duration=8, seed=4), _target(seed=4)).to_dict()
    assert c["series"] != a["series"]


def test_html_has_svg_and_no_external_urls_and_markdown_charts():
    res = run_scenario(Scenario("ramp", users=800, duration=10), _target())
    an = analyse(res)
    page = render_html(res, an, headroom(1000, an.knee_users, 800))
    assert page.count("<svg") == 4 and "</svg>" in page
    assert "http://" not in page and "https://" not in page and "<script" not in page and "<link" not in page
    assert "knee" in page
    md = render_markdown(res, an, headroom(1000, an.knee_users, 800))
    assert "### p95 latency per second" in md and "|####" in md and "**headroom:**" in md


def test_threadpool_path_on_tiny_function_target():
    calls = []

    def handler(req: Request):
        calls.append(req.user_id)
        return 500 if req.seq == 3 else (200, 10)

    res = run_scenario(Scenario("soak", users=5, duration=2), FunctionTarget(handler), virtual=False,
                       max_workers=4, time_scale=0.01)
    assert res.mode == "threads" and res.n_requests == 10 == len(calls)
    assert res.n_5xx == 1 and res.n_ok == 9 and res.bytes_total == 90
    assert all(p.offered == 5 for p in res.series) and all(0 <= p.p95 < 1000 for p in res.series)
    assert res.wall_seconds < 2.0
    ft = FunctionTarget(lambda r: 1 / 0)
    assert ft.call(Request(0, 0.0)).status == 500
    assert FunctionTarget(lambda r: Response(201, 7.0, 3)).call(Request(0, 0.0)) == Response(201, 7.0, 3)
    assert FunctionTarget(lambda r: b"abc").call(Request(0, 0.0)).bytes == 3


def test_simulated_target_on_wall_clock_through_threads():
    t = SimulatedTarget(service_ms=2, concurrency_limit=4, queue_limit=4, virtual_clock=False, seed=1)
    res = run_scenario(Scenario("soak", users=8, duration=1), t, virtual=False, max_workers=8, time_scale=0.05)
    assert res.n_requests == 8 and res.mode == "threads" and t.calls == 8


def test_find_knee_edge_cases():
    assert find_knee([]) == (0.0, None, "no traffic")


def test_cli_run_and_headroom(tmp_path, capsys):
    cfg = tmp_path / "s.yaml"
    cfg.write_text(
        "target:\n  simulated: {service_ms: 50, service_cv: 0.3, concurrency_limit: 30, queue_limit: 30, error_slope: 0.5}\n"
        "launch_target_users: 2000\nrate_limiter: {per_user_rps: 1.0, burst: 1, global_rps: 450}\n"
        "scenarios:\n  spike: {kind: spike, users: 1500, window: 3, hold: 3, join_burst: 2}\n  ramp: {kind: ramp, users: 1200, duration: 20}\n")
    out_json, out_html = tmp_path / "o.json", tmp_path / "r.html"
    rc = main(["run", "--scenario", "spike", "--target", "simulated", "--config", str(cfg), "--json", str(out_json), "--html", str(out_html)])
    assert rc == 1                                                  # the spike found a knee below 1,500 users
    text = capsys.readouterr().out
    assert "## surge test: spike" in text and "**headroom:**" in text
    d = json.loads(out_json.read_text())
    assert d["headroom"]["short"] is True and d["analysis"]["knee_users"] < 1500 and "<svg" in out_html.read_text()
    assert main(["run", "--scenario", "spike", "--target", "simulated", "--config", str(cfg), "--rate-limit"]) == 0
    assert "429" in capsys.readouterr().out
    assert main(["headroom", "--scenario", "ramp", "--users", "1200", "--target", "simulated", "--config", str(cfg)]) == 1
    assert "x short" in capsys.readouterr().out
    assert main(["headroom", "--scenario", "ramp", "--users", "300", "--launch-users", "200", "--config", str(cfg)]) == 0
    assert "headroom" in capsys.readouterr().out
    assert main(["run", "--target", "http", "--config", str(cfg)]) == 2      # --url missing
    assert main(["bogus"]) == 2
