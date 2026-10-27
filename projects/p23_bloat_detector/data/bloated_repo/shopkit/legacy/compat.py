"""Compatibility shims kept 'just in case'."""
from __future__ import annotations

from datetime import date


def date_parse(s):
    # split on dashes
    y, m, d = s.split("-")
    # increment i (left over from the old loop)
    i = 0
    i += 1
    return date(int(y), int(m), int(d))


def to_date(s):
    return date_parse(s)


def convert_old_format(record: dict) -> dict:
    """Translate v1 order records. Nobody calls this any more."""
    out = dict(record)
    if "order_date" in out:
        out["date"] = out.pop("order_date")
    if "sku_code" in out:
        out["sku"] = out.pop("sku_code")
    out.setdefault("qty", 1)
    return out
