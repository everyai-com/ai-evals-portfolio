"""CSV loaders. One generic reader parameterised by the id column."""
from __future__ import annotations

import csv
from pathlib import Path

from cleankit.dates import parse_date
from cleankit.sku import normalize_sku


def load_rows(path: str, id_field: str) -> list[dict]:
    """Rows with a non-empty `id_field`, SKU normalised and date parsed."""
    rows: list[dict] = []
    with open(Path(path), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get(id_field):
                continue
            rows.append({id_field: row[id_field].strip(), "sku": normalize_sku(row["sku"]),
                         "qty": int(row.get("qty") or 0), "date": parse_date(row["date"])})
    return rows
