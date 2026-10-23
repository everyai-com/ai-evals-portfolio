"""Stage 1 practice: a pytest plugin that loads a JSONL eval dataset and emits a metric report.

Usage:
    pytest --eval-dataset data/cases.jsonl
    # in tests:
    def test_case(eval_case, eval_recorder):
        pred = my_model(eval_case["input"])
        eval_recorder.record(eval_case, pred == eval_case["expected"], pred=pred)

At session end the plugin prints accuracy with a Wilson 95% CI, per-tag breakdown, and writes
`eval-report.json` (path configurable with --eval-report).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from evalkit.io import read_jsonl
from evalkit.metrics import wilson_ci


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("evalkit")
    g.addoption("--eval-dataset", action="store", default=None, help="JSONL dataset to parametrize eval_case")
    g.addoption("--eval-report", action="store", default="eval-report.json", help="where to write the metric report")
    g.addoption("--eval-min-accuracy", action="store", type=float, default=None,
                help="fail the session if accuracy is below this value")


class EvalRecorder:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(self, case: dict[str, Any], correct: bool, **extra: Any) -> None:
        self.records.append({"id": case.get("id"), "tags": case.get("tags", []), "correct": bool(correct), **extra})

    def summary(self) -> dict[str, Any]:
        n = len(self.records)
        k = sum(r["correct"] for r in self.records)
        lo, hi = wilson_ci(k, n)
        by_tag: dict[str, list[bool]] = defaultdict(list)
        for r in self.records:
            for t in r["tags"] or ["untagged"]:
                by_tag[t].append(r["correct"])
        tags = {t: {"n": len(v), "accuracy": sum(v) / len(v)} for t, v in sorted(by_tag.items())}
        return {"n": n, "correct": k, "accuracy": k / n if n else 0.0, "ci95": [lo, hi], "by_tag": tags,
                "failures": [r for r in self.records if not r["correct"]]}


_RECORDER = EvalRecorder()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "eval_case" in metafunc.fixturenames:
        path = metafunc.config.getoption("--eval-dataset")
        if not path:
            metafunc.parametrize("eval_case", [], ids=[])
            return
        cases = read_jsonl(path)
        metafunc.parametrize("eval_case", cases, ids=[str(c.get("id", i)) for i, c in enumerate(cases)])


@pytest.fixture
def eval_recorder() -> EvalRecorder:
    return _RECORDER


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if not _RECORDER.records:
        return
    cfg = session.config
    summary = _RECORDER.summary()
    Path(cfg.getoption("--eval-report")).write_text(json.dumps(summary, indent=2))
    tr = cfg.pluginmanager.get_plugin("terminalreporter")
    if tr:
        tr.write_sep("=", "evalkit metric report")
        tr.write_line(f"accuracy: {summary['accuracy']:.3f}  (n={summary['n']}, Wilson 95% CI "
                      f"[{summary['ci95'][0]:.3f}, {summary['ci95'][1]:.3f}])")
        for t, v in summary["by_tag"].items():
            tr.write_line(f"  {t:<24} n={v['n']:<4} acc={v['accuracy']:.3f}")
        tr.write_line(f"report written to {cfg.getoption('--eval-report')}")
    floor = cfg.getoption("--eval-min-accuracy")
    if floor is not None and summary["accuracy"] < floor:
        session.exitstatus = 1
        if tr:
            tr.write_line(f"FAIL: accuracy {summary['accuracy']:.3f} < floor {floor}")
