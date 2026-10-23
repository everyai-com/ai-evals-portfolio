"""Generate the committed demo dataset `tasks.jsonl` (300 small, auto-gradable tasks).

Run: python -m projects.p04_ci_regression_gate.data.make_tasks
Deterministic (numpy seed 0) so re-running reproduces the file byte-for-byte.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from evalkit.io import write_jsonl

WORDS = ["pizza", "oven", "crust", "garlic", "basil", "pepper", "mozzarella", "delivery",
         "coupon", "receipt", "topping", "dough", "sauce", "olive", "anchovy", "mushroom"]


def make_tasks(n: int = 300, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    families = ["add", "sub", "mul", "reverse", "upper", "count"]
    for i in range(n):
        fam = families[i % len(families)]
        hard = bool(rng.random() < 0.3)
        if fam in {"add", "sub", "mul"}:
            hi = 999 if hard else 50
            a, b = int(rng.integers(1, hi)), int(rng.integers(1, hi))
            if fam == "mul" and hard:
                a, b = int(rng.integers(10, 99)), int(rng.integers(10, 99))
            op = {"add": "+", "sub": "-", "mul": "*"}[fam]
            q = f"What is {a} {op} {b}? Answer with just the number."
            exp = str(eval(f"{a}{op}{b}"))
        else:
            w = WORDS[int(rng.integers(0, len(WORDS)))]
            if hard:
                w = w + WORDS[int(rng.integers(0, len(WORDS)))]
            if fam == "reverse":
                q, exp = f"Reverse the word '{w}'. Answer with just the word.", w[::-1]
            elif fam == "upper":
                q, exp = f"Uppercase the word '{w}'. Answer with just the word.", w.upper()
            else:
                q, exp = f"How many letters are in '{w}'? Answer with just the number.", str(len(w))
        rows.append({"id": f"t{i:03d}", "input": q, "expected": exp,
                     "tags": [fam, "hard" if hard else "easy"]})
    return rows


if __name__ == "__main__":
    out = Path(__file__).with_name("tasks.jsonl")
    n = write_jsonl(out, make_tasks())
    print(f"wrote {n} tasks to {out}")
