"""SKU normalisation."""
from __future__ import annotations

import re

_SKU = re.compile(r"^[A-Z]{3}-\d{4}$")


def normalize_sku(raw: str) -> str:
    """Upper-case, dash-separated `ABC-1234`; raise ValueError for anything else."""
    value = raw.strip().upper().replace(" ", "-").replace("_", "-")
    if not _SKU.match(value):
        raise ValueError(f"bad sku {raw!r}")
    return value
