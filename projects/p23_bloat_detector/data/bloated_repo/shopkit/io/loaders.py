from __future__ import annotations

import csv
from pathlib import Path

from shopkit.core.sku import normalize_sku
from shopkit.legacy.compat import to_date
from shopkit.utils.helpers import clean_sku


def load_orders(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(Path(path), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("order_id"):
                continue
            rows.append({
                "order_id": row["order_id"].strip(),
                "sku": normalize_sku(row["sku"]),
                "qty": int(row.get("qty") or 0),
                "date": to_date(row["date"]),
            })
    return rows


def load_customers(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(Path(path), newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("customer_id"):
                continue
            rows.append({
                "customer_id": row["customer_id"].strip(),
                "sku": clean_sku(row["sku"]),
                "qty": int(row.get("qty") or 0),
                "date": to_date(row["date"]),
            })
    return rows
