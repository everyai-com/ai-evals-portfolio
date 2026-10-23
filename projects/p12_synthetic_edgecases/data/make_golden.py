"""Generate the committed seed golden set `golden_seed.jsonl` (40 in-distribution cases).

Run: python -m projects.p12_synthetic_edgecases.data.make_golden
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np

from evalkit.io import write_jsonl

TEMPLATES = [
    ("ship_units", "Ship {qty} units of {sku} on {date}"),
    ("send_x", "Please send {qty} x {sku} by {date}"),
    ("need_pcs", "Need {qty} pcs {sku} delivered {date}"),
    ("order_boxes", "Order: {qty} boxes of {sku}, ship date {date}"),
    ("sku_first", "{sku}: {qty} units, dispatch on {date}"),
]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]


def fmt_date(d: date, style: str) -> str:
    if style == "iso":
        return d.isoformat()
    if style == "us":
        return f"{d.month:02d}/{d.day:02d}/{d.year}"
    return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"


def make_golden(n: int = 40, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        name, tpl = TEMPLATES[i % len(TEMPLATES)]
        qty = int(rng.choice([1, 2, 3, 5, 10, 12, 24, 48, 100, 250, 500, 1200, 2500]))
        sku = f"SKU-{int(rng.integers(0, 10000)):04d}"
        d = date(2026, 1, 1) + timedelta(days=int(rng.integers(0, 365)))
        style = ["iso", "iso", "iso", "us", "long"][int(rng.integers(0, 5))]
        rows.append({"id": f"g{i:03d}", "input": tpl.format(qty=qty, sku=sku, date=fmt_date(d, style)),
                     "expected": {"qty": qty, "sku": sku, "date": d.isoformat()},
                     "tags": [name, f"{style}_date"]})
    return rows


if __name__ == "__main__":
    out = Path(__file__).with_name("golden_seed.jsonl")
    print(f"wrote {write_jsonl(out, make_golden())} cases to {out}")
