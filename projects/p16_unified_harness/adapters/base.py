"""Framework adapter protocol and the unified result every adapter returns."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

from evalkit.io import read_jsonl


@dataclass
class UnifiedResult:
    framework: str
    metrics: dict[str, float]                 # framework-native metric names
    per_example: list[dict[str, Any]]         # {id, score, passed, details}
    cost_usd: float = 0.0
    duration_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    backend: str = "fallback"                 # "real" when the external framework ran, else "fallback"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class Framework(Protocol):
    name: str

    def available(self) -> bool: ...

    def run(self, dataset_path: str, config: dict[str, Any]) -> UnifiedResult: ...


def load_dataset(path: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    for r in rows:
        r.setdefault("tags", [])
        r.setdefault("context", [])
    return rows


def run_sut(examples: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[str], float]:
    """Run the configured SUT (a callable in config['sut_fn']) over examples. Returns outputs + cost."""
    sut = config["sut_fn"]
    outputs, cost = [], 0.0
    for ex in examples:
        text, resp = sut(ex)
        outputs.append(text)
        cost += getattr(resp, "cost_usd", 0.0)
    return outputs, cost


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed = time.perf_counter() - self.t0
