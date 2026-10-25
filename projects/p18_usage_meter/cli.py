"""CLI: `ingest` events into a state dir, `decide` a request, `report` chargeback, `anomalies`, `invoice`."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from evalkit.io import read_jsonl
from evalkit.report import md_table
from projects.p18_usage_meter.core import (
    Enforcer,
    Limits,
    Meter,
    Request,
    chargeback,
    chargeback_csv_rows,
    detect_anomalies,
    dump_json,
    invoice,
    load_prices,
    parse_ts,
    render_chargeback,
    render_invoice,
    write_csv,
)

HERE = Path(__file__).parent


def load(args: argparse.Namespace) -> tuple[Meter, Limits]:
    return Meter.load(args.state, load_prices(args.prices)), Limits.from_yaml(args.limits)


def _ingest(args: argparse.Namespace) -> int:
    meter, _ = load(args)
    before = meter.total()
    res = meter.ingest(read_jsonl(args.events))
    meter.save(args.state)
    print(f"ingested {res.added} new events (+${res.cost_added:.2f}), {res.duplicates} duplicates skipped, "
          f"{res.rejected} rejected; ledger now {len(meter.events)} events, ${meter.total():.2f} "
          f"(was ${before:.2f}); days {meter.days()[0]}..{meter.days()[-1]}" if meter.events else "nothing ingested")
    return 0


def _decide(args: argparse.Namespace) -> int:
    meter, limits = load(args)
    now = parse_ts(args.now) if args.now else datetime.now(UTC)
    enf = Enforcer(meter, limits, lambda: now)
    st_path = Path(args.state) / "enforcer.json"
    if st_path.exists():
        enf.load_state(json.loads(st_path.read_text(encoding="utf-8")))
    req = Request(tenant=args.tenant, user=args.user, agent_id=args.agent, kind=args.kind, model=args.model,
                  input_tokens=args.tokens, output_tokens=args.output_tokens, units=args.units)
    d = enf.decide(req)
    st_path.parent.mkdir(parents=True, exist_ok=True)
    st_path.write_text(dump_json(enf.to_state()), encoding="utf-8")
    verdict = "ALLOW" if d.allow else "BLOCK"
    print(f"{verdict}: {d.reason} (est ${d.estimated_cost_usd:.4f})" + (f"; retry after {d.retry_after_s:.0f}s" if d.retry_after_s else ""))
    rem = ", ".join(f"{k}={'n/a' if v is None else f'{v:.2f}'}" for k, v in d.remaining.items())
    print(f"remaining: {rem}")
    for w in d.warnings:
        print(f"WARNING {w}")
    if args.json:
        print(dump_json(d.to_dict()))
    return 0 if d.allow else 1


def _report(args: argparse.Namespace) -> int:
    meter, limits = load(args)
    anomalies = detect_anomalies(meter, limits.anomaly, args.day)
    rep = chargeback(meter, limits, anomalies, args.month)
    print(render_chargeback(rep))
    if args.csv:
        rows = chargeback_csv_rows(meter, args.month)
        write_csv(rows, args.csv)
        print(f"wrote {len(rows)} rows to {args.csv}")
    if args.json:
        Path(args.json).write_text(dump_json(rep), encoding="utf-8")
    return 1 if any(b["status"] == "over" for b in rep["budgets"].values()) else 0


def _anomalies(args: argparse.Namespace) -> int:
    meter, limits = load(args)
    anomalies = detect_anomalies(meter, limits.anomaly, args.day)
    if args.json:
        print(dump_json([a.to_dict() for a in anomalies]))
        return 1 if anomalies else 0
    if not anomalies:
        print("no anomalies")
        return 0
    print(md_table(["label", "entity", "day", "hour", "spend", "baseline median", "threshold", "recurrence"],
                   [[a.label, f"{a.entity_type}={a.entity}", a.day, f"{a.hour:02d}:00", f"${a.spend:.2f}",
                     f"${a.baseline_median:.2f}", f"${a.threshold:.2f}", f"{a.recurrence_k}/{a.recurrence_n} (lo {a.wilson_lo:.2f})"]
                    for a in anomalies]))
    return 1


def _invoice(args: argparse.Namespace) -> int:
    meter, limits = load(args)
    tenants = [args.tenant] if args.tenant else sorted({e.tenant for e in meter.events})
    for t in tenants:
        months = [args.month] if args.month else sorted({e.month for e in meter.events if e.tenant == t})
        for m in months:
            print(render_invoice(invoice(meter, limits, t, m)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p18_usage_meter", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--state", default="state/", help="state dir holding events.jsonl and enforcer.json")
        sp.add_argument("--prices", default=str(HERE / "data" / "prices.yaml"))
        sp.add_argument("--limits", default=str(HERE / "data" / "limits.yaml"))

    i = sub.add_parser("ingest", help="ingest an events JSONL file (dedup by event_id) into the state dir")
    i.add_argument("--events", required=True)
    common(i)
    i.set_defaults(fn=_ingest)

    d = sub.add_parser("decide", help="evaluate one request against caps, budget and rate limit")
    common(d)
    d.add_argument("--tenant", default="acme")
    d.add_argument("--user", required=True)
    d.add_argument("--agent", required=True)
    d.add_argument("--kind", default="model_call")
    d.add_argument("--model", default="claude-sonnet-5-5")
    d.add_argument("--tokens", type=int, default=0, help="input tokens (model_call)")
    d.add_argument("--output-tokens", type=int, default=0)
    d.add_argument("--units", type=float, default=0.0, help="units for non-model kinds")
    d.add_argument("--now", default=None, help="ISO timestamp (default: wall clock)")
    d.add_argument("--json", action="store_true")
    d.set_defaults(fn=_decide)

    r = sub.add_parser("report", help="chargeback report (markdown) with optional CSV / JSON")
    common(r)
    r.add_argument("--csv", default=None)
    r.add_argument("--json", default=None)
    r.add_argument("--month", default=None)
    r.add_argument("--day", default=None, help="day to scan for anomalies (default: last day)")
    r.set_defaults(fn=_report)

    a = sub.add_parser("anomalies", help="hourly spend anomalies per user / agent")
    common(a)
    a.add_argument("--day", default=None)
    a.add_argument("--json", action="store_true")
    a.set_defaults(fn=_anomalies)

    v = sub.add_parser("invoice", help="invoice-style statement per tenant")
    common(v)
    v.add_argument("--tenant", default=None)
    v.add_argument("--month", default=None)
    v.set_defaults(fn=_invoice)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
