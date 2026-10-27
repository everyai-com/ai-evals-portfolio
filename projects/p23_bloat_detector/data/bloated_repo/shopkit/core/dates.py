"""Canonical date helpers."""
from __future__ import annotations

from datetime import date, datetime

FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d")


def parse_date(text: str) -> date:
    """Parse the formats we accept on order forms; raise ValueError otherwise."""
    cleaned = text.strip()
    for fmt in FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date: {text!r}")


def format_date(d: date) -> str:
    """ISO date for the API."""
    return d.strftime("%Y-%m-%d")


def days_between(a: date, b: date) -> int:
    return abs((b - a).days)
