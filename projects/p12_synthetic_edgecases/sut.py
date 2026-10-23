"""Tiny rule-based baseline system-under-test for `parse_shipping_request`.

Known blind spots (on purpose, so the difficulty filter has something to find):
  * only ASCII digits are recognised (`[0-9]`), so full-width / Arabic-Indic digits fail;
  * two-digit years (03/14/26) are not understood;
  * quantity words ("a dozen", "twelve") are not understood;
  * no range checks on qty, so 0 or 10001 are returned as-is;
  * no multi-intent detection: the first triple wins.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any

MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_QTY = re.compile(r"(?<![\w-])([0-9]+)\s*(?:units?|pcs|pieces|boxes|x)\b", re.I)
_SKU = re.compile(r"SKU-([0-9]{4})")
_DATES = [
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), lambda m: (int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), lambda m: (int(m[3]), int(m[1]), int(m[2]))),
    (re.compile(rf"\b({MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I),
     lambda m: (int(m[3]), datetime.strptime(m[1][:3].title(), "%b").month, int(m[2]))),
    (re.compile(rf"\b(\d{{1,2}})\s+({MONTHS})\s+(\d{{4}})\b", re.I),
     lambda m: (int(m[3]), datetime.strptime(m[2][:3].title(), "%b").month, int(m[1]))),
]


def baseline_sut(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        return {"error": "empty_input"}
    q, s = _QTY.search(text), _SKU.search(text)
    d = None
    for rx, conv in _DATES:
        m = rx.search(text)
        if m:
            try:
                y, mo, da = conv(m)
                d = datetime(y, mo, da).date().isoformat()
            except ValueError:
                return {"error": "invalid_date"}
            break
    if not (q and s and d):
        return {"error": "parse_failed"}
    return {"qty": int(q.group(1)), "sku": f"SKU-{s.group(1)}", "date": d}
