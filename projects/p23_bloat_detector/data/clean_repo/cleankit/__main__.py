import sys

from cleankit.dates import format_date
from cleankit.loaders import load_rows


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    for row in load_rows(argv[0], "order_id") if argv else []:
        print(row["order_id"], row["sku"], format_date(row["date"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
