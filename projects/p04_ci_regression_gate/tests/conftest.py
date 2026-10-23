from __future__ import annotations

import json
from pathlib import Path

import pytest

from projects.p04_ci_regression_gate.core import EvalCache, EvalConfig, EvalRunner
from projects.p04_ci_regression_gate.demo_sut import make_sut

HERE = Path(__file__).resolve().parent.parent
CONFIG = HERE / "data" / "eval_config.yaml"
BASELINE = HERE / "baseline" / "main.json"


def run_variant(variant: str, cache_dir=None, config: EvalConfig | None = None) -> dict:
    cfg = config or EvalConfig.load(CONFIG)
    ex = cfg.load_examples()
    return EvalRunner(cfg, make_sut(variant, ex, cfg.model), cache=EvalCache(cache_dir), variant=variant).run(ex)


@pytest.fixture(scope="session")
def config() -> EvalConfig:
    return EvalConfig.load(CONFIG)


@pytest.fixture(scope="session")
def baseline() -> dict:
    return json.loads(BASELINE.read_text())


@pytest.fixture(scope="session")
def results() -> dict[str, dict]:
    return {v: run_variant(v) for v in ["pr_good", "pr_minor", "pr_regressed", "pr_slow", "pr_flaky"]}
