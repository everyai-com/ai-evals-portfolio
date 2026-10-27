"""Offline demo: two scripted MockLLMs that disagree on a slice of prompts, plus a traffic generator.

primary  = mock-system-one: cheap; has a bug on sums above 100 (drops the hundreds digit).
shadow   = mock-system-two: pricey; 2% simulated backend failures; answers capitals verbosely,
           labels sentiment in lower case, and gets unit conversions wrong for odd inputs.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from evalkit.llm import MockLLM

CAPITALS = {"France": "Paris", "Japan": "Tokyo", "Kenya": "Nairobi", "Peru": "Lima", "Norway": "Oslo",
            "Egypt": "Cairo", "Canada": "Ottawa", "Chile": "Santiago", "Italy": "Rome", "Ghana": "Accra"}
REVIEWS = [("The pizza arrived hot and the crust was perfect", "positive"),
           ("Forty minutes late and cold, never again", "negative"),
           ("Decent toppings but the sauce was bland", "negative"),
           ("Best delivery experience I've had all year", "positive"),
           ("App crashed twice before I could order", "negative"),
           ("Loved the new stuffed crust, will reorder", "positive")]
TOPICS = ["loyalty points", "delivery radius", "allergen labels", "order tracking", "refund policy"]


def _sum_args(prompt: str) -> tuple[int, int]:
    a, b = map(int, re.findall(r"-?\d+", prompt)[:2])
    return a, b


def _capital(prompt: str) -> str:
    country = re.search(r"capital of (\w+)", prompt, re.IGNORECASE).group(1)
    return CAPITALS.get(country, "Unknown")


def _review(prompt: str) -> str:
    text = prompt.split(":", 1)[1].strip().strip('"')
    for t, label in REVIEWS:
        if t.lower() in text.lower():
            return label
    return "neutral"


def _metres(prompt: str) -> str:
    km = int(re.search(r"(\d+) kilometres", prompt).group(1))
    return f"{km * 1000} metres"


def primary_client() -> MockLLM:
    return MockLLM("mock-system-one", handlers=[
        (r"capital of", lambda s, p: _capital(p)),
        (r"sum of", lambda s, p: str(sum(_sum_args(p)) % 100 if sum(_sum_args(p)) > 100 else sum(_sum_args(p)))),
        (r"sentiment", lambda s, p: _review(p).upper()),
        (r"kilometres", lambda s, p: _metres(p)),
        (r"summar", lambda s, p: "Summary: " + p.split("about", 1)[1].strip(".") + " explained briefly."),
    ])


def shadow_client(latency_ms: float = 2.0, fail_rate: float = 0.02, seed: int = 0) -> MockLLM:
    def km(s: str, p: str) -> str:
        n = int(re.search(r"(\d+) kilometres", p).group(1))
        return f"{n * 1000 if n % 2 == 0 else n * 100} metres"  # wrong on odd inputs
    return MockLLM("mock-system-two", latency_ms=latency_ms, fail_rate=fail_rate, seed=seed, handlers=[
        (r"capital of", lambda s, p: f"The capital is {_capital(p)}."),
        (r"sum of", lambda s, p: str(sum(_sum_args(p)))),
        (r"sentiment", lambda s, p: _review(p)),
        (r"kilometres", km),
        (r"summar", lambda s, p: "In short, " + p.split("about", 1)[1].strip(".") + " matters for guests."),
    ])


def make_traffic(n: int = 200, seed: int = 0, start_ts: float = 1_792_540_800.0,
                 span_s: float = 86_400.0) -> list[dict[str, Any]]:
    """Prod-like requests spread evenly over `span_s` seconds; each carries an `expected` answer
    (used only by the report to label success, never shown to the models)."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    countries = list(CAPITALS)
    for i in range(n):
        kind = rng.choice(["capital", "sum", "sentiment", "convert", "summary"],
                          p=[0.3, 0.25, 0.2, 0.15, 0.10])
        ts = start_ts + i * span_s / n
        if kind == "capital":
            c = countries[int(rng.integers(len(countries)))]
            prompt, expected = f"What is the capital of {c}?", CAPITALS[c]
        elif kind == "sum":
            a, b = int(rng.integers(1, 120)), int(rng.integers(1, 120))
            prompt, expected = f"What is the sum of {a} and {b}? Reply with the number only.", str(a + b)
        elif kind == "sentiment":
            t, label = REVIEWS[int(rng.integers(len(REVIEWS)))]
            prompt, expected = f'Classify the sentiment of this review as positive or negative: "{t}"', label
        elif kind == "convert":
            n_km = int(rng.integers(1, 40))
            prompt, expected = f"Convert {n_km} kilometres to metres.", f"{n_km * 1000} metres"
        else:
            prompt, expected = f"Summarize our policy about {TOPICS[int(rng.integers(len(TOPICS)))]}.", None
        row = {"request_id": f"req-{i:05d}", "prompt": prompt, "ts": ts, "tenant": f"t{int(rng.integers(3))}",
               "kind": str(kind)}
        if expected is not None:
            row["expected"] = expected
        rows.append(row)
    return rows
