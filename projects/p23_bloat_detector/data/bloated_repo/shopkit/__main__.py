import sys

from shopkit.api.orders import order_date
from shopkit.core.dates import days_between, format_date
from shopkit.core.sku import sku_prefix
from shopkit.io.loaders import load_customers, load_orders
from shopkit.utils.helpers import chunked


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    orders = load_orders(argv[0]) if argv else []
    customers = load_customers(argv[1]) if len(argv) > 1 else []
    for batch in chunked(orders, 10):
        for o in batch:
            print(o["order_id"], sku_prefix(o["sku"]), format_date(o["date"]), order_date(o["order_id"]))
    d = order_date("ORD-0001")
    print(len(customers), days_between(d, d) if d else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
