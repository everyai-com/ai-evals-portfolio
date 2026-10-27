import datetime as dt
import io
import json

from evalkit.llm import MockLLM
from projects.p09_drift_monitor.cli import main
from projects.p09_drift_monitor.core import (
    DriftDetector,
    JudgeScorer,
    LengthScorer,
    OfflineEvaluator,
    RuleScorer,
    TrafficSampler,
    default_judge_handler,
)
from projects.p09_drift_monitor.daemon import Daemon, JsonlSink, StdoutSink, WebhookSink
from projects.p09_drift_monitor.synth import date_for, make_day


def build(tmp_path, sinks=(), per_day=400, rate=0.25, drift_day=8):
    source = lambda date: make_day(int((dt.date.fromisoformat(date) - dt.date(2026, 10, 1)).days),
                                   drift_start_day=drift_day, seed=0, per_day=per_day)
    judge = MockLLM("judge", handlers=[(r".", default_judge_handler)])
    ev = OfflineEvaluator([RuleScorer(), JudgeScorer(judge), LengthScorer()])
    return Daemon(source, TrafficSampler(rate=rate, seed=0), ev, DriftDetector(), tmp_path / "state", sinks=sinks)


def test_alert_fires_within_two_days_of_drift_and_not_before(tmp_path):
    out = io.StringIO()
    jsonl = tmp_path / "sink.jsonl"
    d = build(tmp_path, sinks=[StdoutSink(out), JsonlSink(jsonl)], drift_day=8)
    first_alert_day = None
    for day in range(11):
        m, alerts = d.run_once(date_for(day))
        assert m.n == 100
        if day < 8:
            assert alerts == [], f"false alarm on stable day {day}: {alerts}"
        elif alerts and first_alert_day is None:
            first_alert_day = day
    assert first_alert_day is not None and first_alert_day - 8 <= 1
    # persisted state
    metrics = d.load_metrics()
    assert [m.date for m in metrics] == [date_for(i) for i in range(11)]
    assert all(a.date >= date_for(8) for a in d.load_alerts()) and len(d.load_alerts()) > 0
    assert (tmp_path / "state" / "alerts.jsonl").exists() and jsonl.read_text().count("\n") == len(d.load_alerts())
    assert "success_rate" in out.getvalue()
    # re-running a day replaces, not duplicates
    d.run_once(date_for(10))
    assert len(d.load_metrics()) == 11


def test_webhook_sink_uses_injected_opener_and_swallows_errors():
    calls = []

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"ok"

    def opener(req, timeout):
        calls.append((req.full_url, json.loads(req.data), req.get_header("Content-type")))
        return Resp()

    from projects.p09_drift_monitor.core import Alert
    a = Alert("warning", "success_rate", 0.9, 0.7, "z=3", ["t1"], "2026-10-09", "zscore")
    WebhookSink("http://hooks.local/x", opener=opener).emit([a])
    assert calls[0][0] == "http://hooks.local/x" and calls[0][1]["alerts"][0]["metric"] == "success_rate"
    assert calls[0][2] == "application/json"
    WebhookSink("http://hooks.local/x", opener=opener).emit([])  # nothing to send
    assert len(calls) == 1

    def boom(req, timeout):
        raise OSError("network down")

    WebhookSink("http://hooks.local/x", opener=boom).emit([a])  # must not raise


def test_run_loop_uses_injected_sleep_and_clock(tmp_path):
    d = build(tmp_path, per_day=100, rate=0.2)
    slept, days = [], iter([dt.date(2026, 10, 2), dt.date(2026, 10, 3), dt.date(2026, 10, 4)])
    n = d.run_loop(interval_s=5, max_iterations=3, now=lambda: next(days), sleep=slept.append)
    assert n == 3 and slept == [5, 5]
    assert [m.date for m in d.load_metrics()] == ["2026-10-01", "2026-10-02", "2026-10-03"]


def test_cli_simulate_run_once_report(tmp_path, capsys):
    state = tmp_path / "state"
    assert main(["simulate", "--days", "9", "--drift-day", "7", "--per-day", "300", "--rate", "0.2",
                 "--out", str(state)]) == 0
    out = capsys.readouterr().out
    assert "# Drift monitor report" in out and "## Alerts" in out and "2026-10-09" in out
    fixture = "projects/p09_drift_monitor/data/sample_day_2026-10-15.jsonl"
    rc = main(["run-once", "--traffic", fixture, "--date", "2026-10-15", "--rate", "0.5", "--state", str(state)])
    assert rc == 1  # drifted sample -> critical alert -> gate blocked
    assert main(["report", "--state", str(state)]) == 0
    assert "2026-10-15" in capsys.readouterr().out
    assert main(["nope"]) == 2
