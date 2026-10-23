"""Nightly daemon: sample -> evaluate -> detect -> persist -> notify."""
from __future__ import annotations

import datetime as dt
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from evalkit.io import read_jsonl, write_jsonl
from projects.p09_drift_monitor.core import Alert, DayMetrics, DriftDetector, OfflineEvaluator, TrafficSampler

log = logging.getLogger("p09_drift_monitor")


class AlertSink(Protocol):
    def emit(self, alerts: Sequence[Alert]) -> None: ...


class StdoutSink:
    def __init__(self, stream: Any = None) -> None:
        self.stream = stream or sys.stdout

    def emit(self, alerts: Sequence[Alert]) -> None:
        for a in alerts:
            print(f"[{a.severity.upper()}] {a.date} {a.metric}: {a.baseline:.3f} -> {a.current:.3f} "
                  f"({a.detector}: {a.evidence}) ids={a.sample_trajectory_ids}", file=self.stream)


class JsonlSink:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, alerts: Sequence[Alert]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            for a in alerts:
                f.write(json.dumps(a.to_dict(), sort_keys=True) + "\n")


class WebhookSink:
    """POSTs one JSON payload per run. `opener` is injectable so tests never touch the network."""

    def __init__(self, url: str, opener: Callable[..., Any] | None = None, timeout: float = 5.0) -> None:
        self.url, self.timeout = url, timeout
        self.opener = opener or urllib.request.urlopen

    def emit(self, alerts: Sequence[Alert]) -> None:
        if not alerts:
            return
        body = json.dumps({"alerts": [a.to_dict() for a in alerts]}).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self.opener(req, timeout=self.timeout) as resp:  # type: ignore[call-arg]
                getattr(resp, "read", lambda: b"")()
        except Exception as e:  # noqa: BLE001 - notification failure must not kill the daemon
            log.warning("webhook %s failed: %s", self.url, e)


TrafficSource = Callable[[str], list[dict[str, Any]]]


class Daemon:
    def __init__(self, traffic_source: TrafficSource, sampler: TrafficSampler, evaluator: OfflineEvaluator,
                 detector: DriftDetector, state_dir: str | Path, sinks: Sequence[AlertSink] = ()) -> None:
        self.traffic_source = traffic_source
        self.sampler, self.evaluator, self.detector = sampler, evaluator, detector
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.state_dir / "metrics.jsonl"
        self.alerts_path = self.state_dir / "alerts.jsonl"
        self.sinks = list(sinks)

    # ---- state
    def load_metrics(self) -> list[DayMetrics]:
        if not self.metrics_path.exists():
            return []
        return sorted((DayMetrics.from_dict(d) for d in read_jsonl(self.metrics_path)), key=lambda m: m.date)

    def load_alerts(self) -> list[Alert]:
        return [Alert.from_dict(d) for d in read_jsonl(self.alerts_path)] if self.alerts_path.exists() else []

    def _save_metrics(self, metrics: DayMetrics) -> None:
        rows = [m for m in self.load_metrics() if m.date != metrics.date] + [metrics]
        write_jsonl(self.metrics_path, (m.to_dict() for m in sorted(rows, key=lambda m: m.date)))

    def _save_alerts(self, alerts: Sequence[Alert], date: str) -> None:
        rows = [a for a in self.load_alerts() if a.date != date] + list(alerts)
        write_jsonl(self.alerts_path, (a.to_dict() for a in rows))

    # ---- work
    def run_once(self, date: str) -> tuple[DayMetrics, list[Alert]]:
        rows = self.traffic_source(date)
        sampled = self.sampler.sample(rows, date)
        metrics = self.evaluator.evaluate(sampled, date)
        history = self.load_metrics()
        alerts = self.detector.detect(history, metrics)
        self._save_metrics(metrics)
        self._save_alerts(alerts, date)
        for sink in self.sinks:
            try:
                sink.emit(alerts)
            except Exception as e:  # noqa: BLE001
                log.warning("sink %s failed: %s", type(sink).__name__, e)
        log.info("%s: sampled %d/%d, success=%.3f, alerts=%d", date, len(sampled), len(rows),
                 metrics.success_rate, len(alerts))
        return metrics, alerts

    def run_loop(self, interval_s: float = 24 * 3600, max_iterations: int | None = None,
                 now: Callable[[], dt.date] | None = None, sleep: Callable[[float], None] = time.sleep) -> int:
        """Evaluate yesterday's traffic, sleep `interval_s`, repeat. Returns iterations run."""
        now = now or dt.date.today
        done = 0
        while max_iterations is None or done < max_iterations:
            self.run_once((now() - dt.timedelta(days=1)).isoformat())
            done += 1
            if max_iterations is not None and done >= max_iterations:
                break
            sleep(interval_s)
        return done
