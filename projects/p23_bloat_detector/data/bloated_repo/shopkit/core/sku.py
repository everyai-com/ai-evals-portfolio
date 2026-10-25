from __future__ import annotations

import re

_SKU = re.compile(r"^[A-Z]{3}-\d{4}$")


def normalize_sku(raw: str) -> str:
    value = raw.strip().upper()
    value = value.replace(" ", "-").replace("_", "-")
    if not _SKU.match(value):
        raise ValueError(f"bad sku {raw!r}")
    return value


def sku_prefix(sku: str) -> str:
    return normalize_sku(sku).split("-")[0]
