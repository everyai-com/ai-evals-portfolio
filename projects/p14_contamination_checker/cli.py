"""CLI for the contamination checker.

    scan --eval data/eval.jsonl --train data/train.jsonl [--ngram 8] [--cosine 0.9] [--json [out.json]]
         [--clean-out clean_eval.jsonl] [--removed-out removed.jsonl] [--results data/results.jsonl]
         [--field text] [--backend hashed|st] [--no-embed] [--no-minhash] [--max-rate 0.0]
    fingerprint --dataset file.jsonl

Exit codes: 0 contamination rate <= --max-rate, 1 above it (gate blocked), 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evalkit.io import dataset_fingerprint, read_jsonl
from evalkit.report import md_table

from . import core

HERE = Path(__file__).parent


def _emit(markdown: str, payload: dict, json_opt: str | None) -> None:
    if json_opt == "-":
        print(json.dumps(payload, indent=2, default=float))
        return
    print(markdown)
    if json_opt:
        Path(json_opt).write_text(json.dumps(payload, indent=2, default=float))
        print(f"\njson written to {json_opt}")


def _cmd_scan(a: argparse.Namespace) -> int:
    try:
        eval_rows, train_rows = read_jsonl(a.eval), read_jsonl(a.train)
        if not eval_rows or not train_rows:
            raise ValueError("empty eval or train file")
        if a.field not in eval_rows[0] or a.field not in train_rows[0]:
            raise ValueError(
                f"field {a.field!r} missing; eval keys {sorted(eval_rows[0])}, train keys {sorted(train_rows[0])}"
            )
        backend = core.get_backend(a.backend)
        report = core.scan(
            eval_rows,
            train_rows,
            ngram=a.ngram,
            exact_ngram=a.exact_ngram,
            cosine=a.cosine,
            near_duplicate=a.near_dup,
            partial=a.partial,
            text_field=a.field,
            id_field=a.id_field,
            strata_field=a.strata_field,
            use_embeddings=not a.no_embed,
            use_minhash=not a.no_minhash,
            backend=backend,
        )
        if a.results:
            core.contamination_aware_metrics(report, read_jsonl(a.results), a.metric)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    md = report.to_markdown()
    extra = ""
    if a.clean_out:
        split = core.clean_split(report, eval_rows, a.clean_out, a.removed_out, id_field=a.id_field)
        extra = (
            f"\n\nclean split: kept {split['kept']} rows -> {split['clean_path']}; removed {split['removed']}"
            + (f" -> {split['removed_path']}" if split["removed_path"] else "")
        )
    _emit(md + extra, report.to_dict(), a.json)
    return 1 if report.rate > a.max_rate else 0


def _cmd_fingerprint(a: argparse.Namespace) -> int:
    try:
        fp = dataset_fingerprint(a.dataset)
    except (ValueError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    md = md_table(
        ["field", "value"],
        [
            ["path", fp["path"]],
            ["rows", fp["n_rows"]],
            ["sha256", fp["sha256"]],
            ["first row hash", fp["row_hashes"][0] if fp["row_hashes"] else "-"],
        ],
    )
    _emit(md, fp, a.json)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="evalkit contamination", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="scan an eval set against a training corpus")
    s.add_argument("--eval", default=str(HERE / "data" / "eval.jsonl"))
    s.add_argument("--train", default=str(HERE / "data" / "train.jsonl"))
    s.add_argument("--ngram", type=int, default=8)
    s.add_argument("--exact-ngram", type=int, default=13)
    s.add_argument("--cosine", type=float, default=0.9)
    s.add_argument("--near-dup", type=float, default=0.8, help="n-gram overlap for near_duplicate")
    s.add_argument("--partial", type=float, default=0.3, help="n-gram overlap for partial")
    s.add_argument("--field", default="text")
    s.add_argument("--id-field", default="id")
    s.add_argument("--strata-field", default="strata")
    s.add_argument(
        "--backend",
        default="hashed",
        help="hashed (default, numpy) or st (sentence-transformers if installed)",
    )
    s.add_argument("--no-embed", action="store_true")
    s.add_argument("--no-minhash", action="store_true")
    s.add_argument(
        "--results",
        default=None,
        help="per-eval-row results JSONL for contamination-aware reporting",
    )
    s.add_argument("--metric", default=None)
    s.add_argument("--clean-out", default=None, help="write the decontaminated eval set here")
    s.add_argument("--removed-out", default=None, help="write the removed rows here")
    s.add_argument(
        "--max-rate", type=float, default=0.0, help="exit 1 when contamination rate exceeds this"
    )
    s.add_argument("--json", nargs="?", const="-", default=None)
    s.set_defaults(fn=_cmd_scan)
    f = sub.add_parser("fingerprint", help="sha256 fingerprint of a JSONL dataset")
    f.add_argument("--dataset", required=True)
    f.add_argument("--json", nargs="?", const="-", default=None)
    f.set_defaults(fn=_cmd_fingerprint)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    try:
        a = build_parser().parse_args(argv)
    except SystemExit as e:
        return 0 if e.code == 0 else 2
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
