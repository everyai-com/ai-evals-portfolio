"""Demo system-under-test: a MockLLM-backed "agent" whose behaviour is selected by a variant.

The variant simulates what a PR might do to the agent so the gate can be demonstrated both ways:

| variant        | task success vs main | latency | notes                                   |
|----------------|----------------------|---------|-----------------------------------------|
| main           | baseline (90%)       | 1x      | the committed baseline                  |
| pr_good        | +1 point             | 1x      | fixes 3 items                           |
| pr_minor       | -1 point             | 1x      | small, non-significant drop -> warn     |
| pr_regressed   | -5 points            | 1x      | breaks 15 more items -> block           |
| pr_slow        | baseline             | 2.5x    | latency p95 breach -> block after k runs|
| pr_flaky       | baseline             | 1x      | 15 items fail only on odd seeds         |

Failures are deterministic functions of (item id, variant, seed): items are ranked by the stable
hash of their id and the variant decides how many of the lowest-ranked ids the "model" gets wrong.
Latency is *simulated* (reported by the SUT, not measured) so the demo is fast and reproducible;
``EvalRunner`` falls back to wall-clock latency when a SUT does not report one.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from evalkit.io import stable_hash
from evalkit.llm import MockLLM

VARIANTS = ["main", "pr_good", "pr_minor", "pr_regressed", "pr_slow", "pr_flaky"]

# (always-failing items, seed-dependent flaky items, latency multiplier) per variant
_FAIL_FRAC = {
    "main": (0.10, 0.0, 1.0),
    "pr_good": (0.09, 0.0, 1.0),
    "pr_minor": (0.11, 0.0, 1.0),
    "pr_regressed": (0.15, 0.0, 1.0),
    "pr_slow": (0.10, 0.0, 2.5),
    "pr_flaky": (0.10, 0.05, 1.0),
}


@dataclass
class SUTOutput:
    output: str
    latency_ms: float | None = None   # None => runner measures wall clock
    cost_usd: float = 0.0
    meta: dict[str, Any] | None = None


SUT = Callable[[dict[str, Any], int], SUTOutput]  # (example, seed) -> output


def _solve(prompt: str) -> str:
    """The 'competent model': solves the demo task families exactly."""
    m = re.search(r"What is (\d+) ([+\-*]) (\d+)", prompt)
    if m:
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        return str({"+": a + b, "-": a - b, "*": a * b}[op])
    m = re.search(r"Reverse the word '([^']+)'", prompt)
    if m:
        return m.group(1)[::-1]
    m = re.search(r"Uppercase the word '([^']+)'", prompt)
    if m:
        return m.group(1).upper()
    m = re.search(r"How many letters are in '([^']+)'", prompt)
    if m:
        return str(len(m.group(1)))
    return "I don't know"


def make_sut(variant: str, examples: list[dict[str, Any]], model: str = "mock-system-one") -> SUT:
    """Build the demo agent for a variant. `examples` fixes the hash ranking of item ids."""
    if variant not in _FAIL_FRAC:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    fail_frac, flaky_frac, lat_mult = _FAIL_FRAC[variant]
    ranked = sorted((e["id"] for e in examples), key=lambda i: stable_hash(i))
    n = len(ranked)
    always_fail = set(ranked[: int(round(fail_frac * n))])
    flaky = set(ranked[int(round(0.10 * n)): int(round((0.10 + flaky_frac) * n))])

    def handler(system: str, prompt: str) -> str:
        m = re.search(r"\[case (\S+) seed (\d+)\]", system)
        cid, seed = m.group(1), int(m.group(2))
        answer = _solve(prompt)
        if cid in always_fail or (cid in flaky and seed % 2 == 1):
            return answer + "0" if answer.isdigit() else answer[::-1] + "x"   # a confident wrong answer
        return answer

    llm = MockLLM(model=model, handlers=[(r".", handler)])

    def sut(example: dict[str, Any], seed: int) -> SUTOutput:
        system = f"You are a precise assistant. [case {example['id']} seed {seed}]"
        resp = llm.complete(example["input"], system=system, seed=seed)
        rng = np.random.default_rng(int(stable_hash([example["id"], seed])[:8], 16))
        base = 120.0 if "hard" in example.get("tags", []) else 80.0
        latency = float(base * lat_mult + rng.gamma(2.0, 15.0))
        return SUTOutput(output=resp.text.strip(), latency_ms=latency, cost_usd=resp.cost_usd,
                         meta={"variant": variant})

    return sut
