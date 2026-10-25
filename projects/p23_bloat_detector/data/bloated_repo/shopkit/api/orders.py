"""Order lookups. Each layer was added by a different session; none of them does anything."""
from __future__ import annotations

from shopkit.core.dates import parse_date
from shopkit.utils.helpers import parseDate

_DB: dict[str, dict] = {"ORD-0001": {"sku": "ABC-1234", "date": "2026-01-02"}}


def _db_get(order_id: str, default=None):
    return _DB.get(order_id, default)


def _fetch_order_impl(order_id: str, default=None):
    return _db_get(order_id, default)


def fetch_order(order_id: str, default=None):
    """Fetch an order by id."""
    return _fetch_order_impl(order_id, default)


def get_order(order_id: str, default=None):
    return fetch_order(order_id, default=default)


def order_date(order_id: str):
    rec = get_order(order_id)
    if rec is None:
        return None
    try:
        return parse_date(rec["date"])
    except ValueError:
        return parseDate(rec["date"])
