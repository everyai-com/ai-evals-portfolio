"""CLI: `resolve` one email, `audit` a user directory, `magic-link issue|verify`."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from evalkit.io import read_jsonl
from projects.p19_signin_router.core import (
    IdentityRegistry,
    InvalidEmail,
    MagicLink,
    PasskeyStore,
    Router,
    StdoutSender,
    audit,
    render_audit,
)

HERE = Path(__file__).parent


def build_router(args: argparse.Namespace) -> Router:
    reg = IdentityRegistry.from_yaml(args.providers)
    pk = PasskeyStore.from_jsonl(args.passkeys) if Path(args.passkeys).exists() else PasskeyStore()
    return Router(reg, pk)


def _resolve(args: argparse.Namespace) -> int:
    router = build_router(args)
    try:
        r = router.resolve(args.email)
    except InvalidEmail as e:
        print(f"invalid: {e}")
        return 2
    if args.json:
        print(json.dumps(r.to_dict(), indent=2, sort_keys=True))
    else:
        print(f"{r.email} -> {r.method}" + (f" via {r.provider}" if r.method == "idp" else "") +
              (f" [{r.kind}]" if r.kind else "") + f"\n  reason: {r.reason}" +
              (f"\n  did you mean: {r.suggestion}" if r.suggestion else "") +
              (f"\n  fallbacks: {', '.join(r.fallbacks)}" if r.fallbacks else ""))
    return 1 if r.method == "rejected" else 0


def _audit(args: argparse.Namespace) -> int:
    router = build_router(args)
    rep = audit(read_jsonl(args.users), router)
    if args.json:
        slim = {k: v for k, v in rep.items() if k != "routes"}
        print(json.dumps(slim, indent=2, sort_keys=True))
    else:
        print(render_audit(rep))
    return 1 if rep["unknown_users"] else 0


def _magic(args: argparse.Namespace) -> tuple[MagicLink, Path]:
    secret = args.secret or os.environ.get("P19_MAGIC_SECRET", "dev-secret-change-me")
    clock = (lambda: float(args.now)) if args.now is not None else time.time
    ml = MagicLink(secret, clock, ttl_s=args.ttl, sender=StdoutSender())
    st = Path(args.state) / "magic_links.json"
    if st.exists():
        ml.load_state(json.loads(st.read_text(encoding="utf-8")))
    return ml, st


def _issue(args: argparse.Namespace) -> int:
    ml, _ = _magic(args)
    try:
        token = ml.issue(args.email)
    except InvalidEmail as e:
        print(f"invalid: {e}")
        return 2
    print(f"token: {token}")
    return 0


def _verify(args: argparse.Namespace) -> int:
    ml, st = _magic(args)
    r = ml.verify(args.token)
    st.parent.mkdir(parents=True, exist_ok=True)
    st.write_text(json.dumps(ml.to_state(), indent=2), encoding="utf-8")
    print(("OK" if r.ok else "REJECTED") + f": {r.reason}" + (f" ({r.email})" if r.email else ""))
    return 0 if r.ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="p19_signin_router", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--providers", default=str(HERE / "data" / "providers.yaml"))
        sp.add_argument("--passkeys", default=str(HERE / "data" / "passkeys.jsonl"))

    r = sub.add_parser("resolve", help="route one email address")
    r.add_argument("--email", required=True)
    r.add_argument("--json", action="store_true")
    common(r)
    r.set_defaults(fn=_resolve)

    a = sub.add_parser("audit", help="reachability audit over a user directory (JSONL: email, business_unit, country)")
    a.add_argument("--users", default=str(HERE / "data" / "users.jsonl"))
    a.add_argument("--json", action="store_true")
    common(a)
    a.set_defaults(fn=_audit)

    m = sub.add_parser("magic-link", help="issue / verify HMAC-signed single-use tokens")
    msub = m.add_subparsers(dest="mcmd", required=True)

    def mcommon(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--state", default="state/", help="dir holding magic_links.json (used nonces)")
        sp.add_argument("--secret", default=None, help="HMAC secret (default: $P19_MAGIC_SECRET or a dev value)")
        sp.add_argument("--ttl", type=float, default=900.0)
        sp.add_argument("--now", type=float, default=None, help="epoch seconds (default: wall clock)")

    i = msub.add_parser("issue")
    i.add_argument("--email", required=True)
    mcommon(i)
    i.set_defaults(fn=_issue)
    v = msub.add_parser("verify")
    v.add_argument("--token", required=True)
    mcommon(v)
    v.set_defaults(fn=_verify)
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
