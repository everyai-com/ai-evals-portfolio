"""Tiny markdown helpers so every project emits the same report shape."""
from __future__ import annotations

from typing import Any, Sequence


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3g}" if 0 < abs(v) < 1e-3 else f"{v:.3f}"
        return str(v)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(out)


def pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"
