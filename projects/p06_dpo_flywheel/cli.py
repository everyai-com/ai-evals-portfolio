"""CLI for the DPO flywheel.

    python -m projects.p06_dpo_flywheel.cli ingest --events data/feedback.jsonl --state state/
    python -m projects.p06_dpo_flywheel.cli build-pairs --state state/ --out pairs/
    python -m projects.p06_dpo_flywheel.cli train --pairs pairs/train.jsonl --out adapters/ [--dry-run]
    python -m projects.p06_dpo_flywheel.cli nightly --events data/feedback.jsonl --state state/

Exit codes: 0 ok / promoted, 1 nightly skipped or adapter not promoted, 2 usage / data error.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

from evalkit.io import read_jsonl
from evalkit.llm import get_client
from evalkit.report import md_table
from projects.p06_dpo_flywheel.core import FeedbackStore, PairBuilder, append_golden, write_pairs
from projects.p06_dpo_flywheel.nightly import NightlyJob, render_report
from projects.p06_dpo_flywheel.trainer import DataValidationError, train_lora

HERE = Path(__file__).parent
GOLDEN = HERE / "data" / "golden_additions.jsonl"
CHANGELOG = HERE / "data" / "CHANGELOG.md"


def demo_regen_handler(system: str, prompt: str) -> str:
    """MockLLM handler for strategy 3: a plausible support answer keyed on the prompt topic."""
    m = re.search(r"order (\d+)", prompt)
    n = m.group(1) if m else "your order"
    low = prompt.lower()
    if "charged" in low or "charge" in low:
        return f"Sorry about the duplicate charge on order {n}. One line is a pending authorisation that drops off within 3-5 business days; you are billed once. Reply if it persists and I will escalate."
    if "refund" in low or "cold" in low:
        return f"I'm sorry order {n} arrived cold. I have refunded it in full to your original payment method (5-7 business days) and added a credit for your next order."
    if "password" in low:
        return "Tap Account > Security > Reset password in the app and we will email a one-time link valid for 30 minutes; check spam if it does not arrive."
    if "sauce" in low:
        return "Yes, tap Customise on the pizza, choose Sauce and pick marinara, garlic cream or BBQ at no extra cost."
    if "order" in low:
        return f"Order {n} is with the courier; the tracking link in your confirmation email updates every 15 minutes and I have raised a trace so you get an update within 24 hours."
    return "Thanks for flagging this. Here is what I can do: " + prompt[:80]


def make_client():
    return get_client(handlers=[(r".", demo_regen_handler)])


def cmd_ingest(a) -> int:
    store = FeedbackStore(a.state)
    st = store.ingest(read_jsonl(a.events))
    print("## ingest\n")
    print(md_table(["received", "duplicates", "accepted", "rejected", "pii_scrubbed", "watermark"],
                   [[st.received, st.duplicates, st.accepted, json.dumps(st.rejected), json.dumps(st.pii_scrubbed), st.watermark]]))
    print(f"\n{len(store.events())} events in {store.events_path}")
    return 0


def cmd_build_pairs(a) -> int:
    store = FeedbackStore(a.state)
    events = store.events()
    if not events:
        print("no events in state; run ingest first", file=sys.stderr)
        return 2
    builder = PairBuilder(client=None if a.no_regen else make_client())
    pairs = builder.build(events)
    counts = write_pairs(pairs, a.out, a.val_frac, a.seed)
    added = append_golden(pairs, a.golden, a.changelog)
    print("## build-pairs\n")
    print(md_table(["counter", "n"], [[k, v] for k, v in sorted(builder.stats.items())]))
    print(f"\nsplit: {counts} -> {a.out}/{{all,train,val}}.jsonl (trl DPO format)")
    print(f"golden additions: +{added} -> {a.golden} (changelog {a.changelog})")
    return 0


def cmd_train(a) -> int:
    try:
        m = train_lora(a.pairs, a.base_model, a.out, json.loads(a.config) if a.config else None,
                       dry_run=True if a.dry_run else None)
    except DataValidationError as e:
        print(f"data validation failed: {e}", file=sys.stderr)
        return 2
    d = m["dataset"]
    print(f"## train ({'dry-run' if m['dry_run'] else 'real'})\n")
    print(md_table(["pairs", "tokens", "est GPU-h", "lora r/alpha/dropout", "beta", "lr", "epochs", "status"],
                   [[d["n_pairs"], d["total_tokens"], d["estimated_gpu_hours"],
                     f"{m['hyperparams']['lora_r']}/{m['hyperparams']['lora_alpha']}/{m['hyperparams']['lora_dropout']}",
                     f"{m['hyperparams']['beta']:g}", f"{m['hyperparams']['learning_rate']:g}", m["hyperparams"]["epochs"], m["status"]]]))
    print(f"\nmanifest: {Path(a.out) / 'training_manifest.json'}")
    return 0


def cmd_nightly(a) -> int:
    job = NightlyJob(a.state, a.events, client=None if a.no_regen else make_client(),
                     golden_path=a.golden, changelog_path=a.changelog)
    r = job.run(a.date or date.today().isoformat())
    print(render_report(r))
    if a.json:
        Path(a.json).write_text(json.dumps(r.to_dict(), indent=1))
    return 0 if r.status == "promoted" else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p06_dpo_flywheel", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest")
    i.add_argument("--events", default=str(HERE / "data" / "feedback.jsonl"))
    i.add_argument("--state", default="state")
    i.set_defaults(fn=cmd_ingest)
    b = sub.add_parser("build-pairs")
    b.add_argument("--state", default="state")
    b.add_argument("--out", default="pairs")
    b.add_argument("--golden", default=str(GOLDEN))
    b.add_argument("--changelog", default=str(CHANGELOG))
    b.add_argument("--val-frac", type=float, default=0.2)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--no-regen", action="store_true", help="skip strategy 3 (no LLM calls)")
    b.set_defaults(fn=cmd_build_pairs)
    t = sub.add_parser("train")
    t.add_argument("--pairs", required=True)
    t.add_argument("--out", default="adapters")
    t.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    t.add_argument("--config", default=None, help="JSON dict of TrainConfig overrides")
    t.add_argument("--dry-run", action="store_true")
    t.set_defaults(fn=cmd_train)
    n = sub.add_parser("nightly")
    n.add_argument("--events", default=str(HERE / "data" / "feedback.jsonl"))
    n.add_argument("--state", default="state")
    n.add_argument("--date", default=None)
    n.add_argument("--golden", default=str(GOLDEN))
    n.add_argument("--changelog", default=str(CHANGELOG))
    n.add_argument("--no-regen", action="store_true")
    n.add_argument("--json", default=None)
    n.set_defaults(fn=cmd_nightly)
    return p


def main(argv: list[str] | None = None) -> int:
    try:
        a = build_parser().parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
