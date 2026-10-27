import math

import pytest

from evalkit.llm import MockLLM
from evalkit.trajectory import Trajectory
from projects.p09_drift_monitor.core import (
    DayMetrics,
    DriftDetector,
    JudgeScorer,
    LengthScorer,
    OfflineEvaluator,
    RuleScorer,
    binom_sf,
    chi2_sf,
    cusum,
    default_judge_handler,
    load_thresholds,
    psi,
    psi_expected_noise,
)


def test_psi_known_case():
    # baseline: 50/30/20 ; current: 20/30/50 over three bins, raw shares (alpha=0)
    base = [0] * 50 + [1] * 30 + [2] * 20
    cur = [0] * 20 + [1] * 30 + [2] * 50
    bins = [0, 1, 2, 3]
    expected = sum((q - p) * math.log(q / p) for p, q in [(0.5, 0.2), (0.3, 0.3), (0.2, 0.5)])
    assert psi(base, cur, bins, alpha=0) == pytest.approx(expected)
    assert expected == pytest.approx(0.5498, abs=1e-3)
    assert psi(base, base, bins, alpha=0) == 0.0
    assert psi(base, base, bins) == 0.0            # smoothing cancels on identical samples
    assert psi([], cur, bins) == 0.0
    assert psi_expected_noise(350, 50, 5) == pytest.approx(4 * (1 / 350 + 1 / 50))


def test_psi_smoothing_handles_empty_bins():
    base = [0] * 95 + [1] * 5
    cur = [0] * 50
    val = psi(base, cur, [0, 1, 2])
    assert 0 < val < 0.2  # an empty bin must not blow up to ~0.3+


def test_chi2_and_binomial_tails():
    assert chi2_sf(9.488, 4) == pytest.approx(0.05, abs=1e-3)
    assert chi2_sf(3.841, 1) == pytest.approx(0.05, abs=1e-3)
    assert chi2_sf(0.0, 3) == 1.0
    assert binom_sf(0, 10, 0.1) == 1.0
    assert binom_sf(1, 10, 0.1) == pytest.approx(1 - 0.9 ** 10)
    assert binom_sf(3, 3, 0.5) == pytest.approx(0.125)


def test_cusum_fires_on_shift_not_on_noise():
    stable = [0.90, 0.91, 0.89, 0.90, 0.92, 0.88, 0.90]
    path, first = cusum(stable + [0.90], 0.90, 0.02, k=0.5, h=3.0)
    assert first is None and max(path) < 3.0
    path, first = cusum(stable + [0.70], 0.90, 0.02, k=0.5, h=3.0)
    assert first == 7 and path[-1] == pytest.approx(10 - 0.5)


def dm(date, rate, n=100, halluc=0.02, tools=None, lengths=None):
    tools = tools or [1] * 40 + [2] * 40 + [3] * 20
    lengths = lengths or [50] * 50 + [70] * 50
    k = int(round(rate * n))
    return DayMetrics(date, n, rate, halluc, 4.0, 1000.0, 0.0004, 1.0, 0.0, 0.0, 60.0,
                      tool_call_counts=tools, answer_lengths=lengths,
                      failed_ids=[f"{date}-f{i}" for i in range(n - k)],
                      hallucinated_ids=[f"{date}-h{i}" for i in range(int(round(halluc * n)))])


def test_no_alert_on_stable_days_and_alert_on_drop():
    det = DriftDetector()
    hist = [dm(f"2026-10-0{i}", r) for i, r in enumerate([0.90, 0.91, 0.89, 0.90, 0.92, 0.88, 0.90], 1)]
    assert det.detect(hist, dm("2026-10-08", 0.89)) == []
    alerts = det.detect(hist, dm("2026-10-08", 0.70))
    dets = {a.detector for a in alerts}
    assert {"cusum", "zscore"} <= dets
    a = next(a for a in alerts if a.detector == "zscore")
    assert a.metric == "success_rate" and a.severity == "critical" and a.baseline == pytest.approx(0.90, abs=0.01)
    assert a.current == 0.70 and len(a.sample_trajectory_ids) == 5 and a.date == "2026-10-08"


def test_insufficient_baseline_gives_no_alert():
    det = DriftDetector()
    hist = [dm("2026-10-01", 0.9), dm("2026-10-02", 0.9)]
    assert det.detect(hist, dm("2026-10-03", 0.3)) == []


def test_psi_and_hallucination_rules():
    det = DriftDetector()
    hist = [dm(f"2026-10-0{i}", 0.90) for i in range(1, 8)]
    shifted = dm("2026-10-08", 0.90, tools=[4] * 60 + [5] * 40, lengths=[150] * 100)
    alerts = det.detect(hist, shifted)
    assert {a.metric for a in alerts if a.detector == "psi"} == {"tool_call_count_dist", "answer_length_dist"}
    assert all(a.severity == "critical" for a in alerts if a.detector == "psi")
    spike = dm("2026-10-08", 0.90, halluc=0.15)
    (h,) = [a for a in det.detect(hist, spike) if a.detector == "hallucination_spike"]
    assert h.severity == "critical" and h.baseline == pytest.approx(0.02) and h.current == 0.15
    assert h.sample_trajectory_ids == [f"2026-10-08-h{i}" for i in range(5)]
    # 3 flags out of 100 at a 2% base is not a spike (binomial tail ~0.32)
    assert not [a for a in det.detect(hist, dm("2026-10-08", 0.90, halluc=0.03)) if a.detector == "hallucination_spike"]


def test_thresholds_yaml(tmp_path):
    cfg = tmp_path / "t.yaml"
    cfg.write_text("baseline_window_days: 3\nzscore:\n  warn: 1.0\n")
    th = load_thresholds(cfg)
    assert th["baseline_window_days"] == 3 and th["zscore"]["warn"] == 1.0 and th["zscore"]["critical"] == 4.5


def test_scorers_and_evaluator():
    judge = MockLLM("judge", handlers=[(r".", default_judge_handler)])
    ev = OfflineEvaluator([RuleScorer(), JudgeScorer(judge), LengthScorer()])
    good = Trajectory("g", meta={"tenant": "a"}).add_tool("lookup_order", {"order_id": "1"}, result={"ok": 1},
                                                          latency_ms=100, cost_usd=0.001).finish("Order 1 is on its way.")
    bad_args = Trajectory("b1", meta={"tenant": "a"}).add_tool("lookup_order", {}, result={}).finish("hi")
    errored = Trajectory("b2", meta={"tenant": "b"}).add_tool("search_menu", {"query": "x"}, error="timeout").finish("hi")
    empty = Trajectory("b3", meta={"tenant": "b"}).add_tool("search_menu", {"query": "x"}).finish("")
    halluc = Trajectory("h", meta={"tenant": "b"}).add_tool("search_menu", {"query": "x"}).finish(
        "We have 3 options. I personally guarantee it.")
    unknown_tool = Trajectory("u", meta={"tenant": "a"}).add_tool("rm_rf", {}).finish("done")
    m = ev.evaluate([good, bad_args, errored, empty, halluc, unknown_tool], "2026-10-01")
    assert m.n == 6 and m.success_rate == pytest.approx(2 / 6)
    assert m.hallucination_rate == pytest.approx(1 / 6) and m.hallucinated_ids == ["h"]
    assert set(m.failed_ids) == {"b1", "b2", "b3", "u"}
    assert m.tools_valid_rate == pytest.approx(4 / 6) and m.error_step_rate == pytest.approx(1 / 6)
    assert m.empty_answer_rate == pytest.approx(1 / 6)
    assert m.per_tenant["a"]["n"] == 3 and m.per_tenant["b"]["success_rate"] == pytest.approx(1 / 3)
    assert m.tool_call_counts == [1] * 6 and m.avg_steps == 2.0
    assert DayMetrics.from_dict(m.to_dict()) == m
    r = RuleScorer().score(bad_args)
    assert r["invalid_tool_calls"] == ["lookup_order missing order_id"]
