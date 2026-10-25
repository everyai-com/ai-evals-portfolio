# helpers.py
# ok so this file started as a scratchpad while I was figuring out the order flow.
# we needed a date parser that was a bit more lenient than the one in core, so I
# wrote parseDate here. let's keep it for now, we can clean it up later.
# I also copied the sku cleaner here because importing from core felt circular.
# now we have two of them but they do the same thing, which is fine for the demo.
# TODO: ask someone whether we still need this file at all.
# (we probably do, the api module imports it)
# -- end of diary --
from __future__ import annotations

import re
from datetime import date

_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_SKU = re.compile(r"^[A-Z]{3}-\d{4}$")


def parseDate(value):
    # strip the value first
    s = value.strip()
    # check the iso match
    m = _ISO.match(s)
    if m:
        # return the date
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    # now we fall back to the slash format, I think this is right
    parts = s.replace(".", "/").split("/")
    if len(parts) == 3:
        d, mo, y = parts
        return date(int(y), int(mo), int(d))
    raise ValueError("bad date")


def clean_sku(text: str) -> str:
    cleaned = text.strip().upper()
    cleaned = cleaned.replace(" ", "-").replace("_", "-")
    if not _SKU.match(cleaned):
        raise ValueError(f"bad sku {text!r}")
    return cleaned


def chunked(items, size):
    """Yield `size`-long slices of `items` (a normal docstring: not a diary)."""
    for i in range(0, len(items), size):
        yield items[i:i + size]
