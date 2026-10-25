"""CLI for the deploy gate.

    scan  [--dir DIR] [--json [out.json]] [--findings findings.md] [--no-review]
          [--rules data/secret_rules.yaml] [--policy data/policy.yaml] [--schema data/manifest_schema.yaml]
          (no --dir: builds the planted demo artifact in a temp dir and scans that)
    demo  --out DIR    write the planted demo artifact to DIR (see fixtures.py)
    rules [--rules ...] [--policy ...]

Exit codes: 0 deploy allowed, 1 blocked, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from evalkit.report import md_table

from . import core
from .fixtures import materialize_sample

HERE = Path(__file__).parent


def _emit(markdown: str, payload: object, json_opt: str | None) -> None:
    if json_opt == "-":
        print(json.dumps(payload, indent=2, default=str))
        return
    print(markdown)
    if json_opt:
        Path(json_opt).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\njson written to {json_opt}")


def _cmd_scan(a: argparse.Namespace) -> int:
    if a.dir is None:  # demo: the planted artifact only exists once built (no secrets are committed)
        with tempfile.TemporaryDirectory(prefix="p22-demo-") as tmp:
            a.dir = str(materialize_sample(Path(tmp) / "sample_artifact"))
            print(f"(scanning the planted demo artifact built at {a.dir})\n")
            return _cmd_scan(a)
    try:
        policy = core.Policy.load(a.policy)
        checks = core.PolicyChecks(policy, core.ManifestValidator(a.schema))
        scanner = core.SecretScanner(a.rules, policy)
        gate = core.Gate(scanner, checks, use_review=not a.no_review)
        verdict = gate.run(a.dir)
    except (ValueError, KeyError, FileNotFoundError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    md = verdict.to_markdown()
    if a.findings:
        Path(a.findings).write_text(core.findings_md(verdict), encoding="utf-8")
        md += f"\n\nfix-loop packet written to {a.findings} ({len(verdict.findings)} tasks)"
    _emit(md, verdict.to_dict(), a.json)
    return 1 if verdict.blocked else 0


def _cmd_demo(a: argparse.Namespace) -> int:
    try:
        out = materialize_sample(a.out)
    except FileExistsError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"planted demo artifact written to {out}; scan it with: scan --dir {out}")
    return 0


def _cmd_rules(a: argparse.Namespace) -> int:
    try:
        policy = core.Policy.load(a.policy)
        scanner = core.SecretScanner(a.rules, policy)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    md = "# Secret rules\n\n" + md_table(
        ["id", "severity", "blocks", "pattern"],
        [[r.id, r.severity, "yes" if r.severity in core.BLOCKING_SECRET else "no", f"`{r.pattern.pattern[:60]}`"] for r in scanner.rules]
        + [["high_entropy_string", scanner.entropy_severity, "no",
            f"Shannon entropy >= {scanner.entropy_bits} bits/char over {scanner.entropy_min_len}+ chars"]],
    )
    md += "\n\n# Policy\n\n" + md_table(
        ["check", "detail"],
        [["required files", ", ".join(policy.required_files)],
         ["forbidden files", ", ".join(f["glob"] for f in policy.forbidden_files)],
         ["max blob", f"{policy.max_blob_mb:.0f} MB"],
         ["own servers", "allowed" if policy.allow_own_server else "forbidden (0.0.0.0 flagged)"],
         ["test paths skipped", ", ".join(policy.test_path_markers)],
         ["placeholders ignored", ", ".join(policy.placeholder_markers)]]
        + [[f"pattern {p.id}", f"{p.severity}: {p.message} ({', '.join(sorted(p.extensions))})"] for p in policy.patterns],
    )
    _emit(md, {"rules": [r.id for r in scanner.rules], "patterns": [p.id for p in policy.patterns]}, a.json)
    return 0


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rules", default=str(HERE / "data" / "secret_rules.yaml"))
    p.add_argument("--policy", default=str(HERE / "data" / "policy.yaml"))
    p.add_argument("--json", nargs="?", const="-", default=None)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="evalkit deploygate", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="secret scan + policy checks + model review of an artifact directory")
    s.add_argument("--dir", default=None, help="artifact to scan (default: build and scan the planted demo)")
    s.add_argument("--schema", default=str(HERE / "data" / "manifest_schema.yaml"))
    s.add_argument("--findings", default=None, help="write the fix-loop packet (markdown) here")
    s.add_argument("--no-review", action="store_true", help="skip the model review")
    _common(s)
    s.set_defaults(fn=_cmd_scan)
    d = sub.add_parser("demo", help="write the planted demo artifact to a directory")
    d.add_argument("--out", required=True)
    d.set_defaults(fn=_cmd_demo)
    r = sub.add_parser("rules", help="list secret rules and policy checks")
    _common(r)
    r.set_defaults(fn=_cmd_rules)
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
