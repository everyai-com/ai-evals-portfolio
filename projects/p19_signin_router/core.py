"""Email-first sign-in routing: one email box, the system picks passkey / IdP / magic link.

Rules (so the README claims are checkable)
------------------------------------------
Normalisation  trim whitespace, lowercase, drop a `+tag` from the local part (so
               ben+oncall@x.io and ben@x.io are the same account). Anything that is not
               `local@domain.tld` raises InvalidEmail.
Precedence     1. passkey registered for the normalised email          -> method=passkey
               2. domain (or alias) owned by a provider                 -> method=idp
               3. domain acquired by a company and federated=true       -> method=idp (parent's)
               4. anything else                                         -> method=magic_link
               Disposable domains are rejected before any of this      -> method=rejected
               Every route carries the fallbacks that come after it, so the UI can offer
               "use a magic link instead" without a second lookup.
Typo hint      An unknown domain within Levenshtein distance 1 of a known domain gets
               `suggestion` ("did you mean"). The route is still magic link (never silently
               rewrite what the user typed).
Magic link     token = base64url(json{email, iat, exp, nonce}) + "." + base64url(HMAC-SHA256(secret,
               payload)). verify() checks the signature (constant-time compare), expiry against the
               injected clock, and single use via the nonce set; a used or expired token is rejected
               with a reason.
Audit          For a directory of users, count routes per method and provider, and list the
               domains that fall to magic link or rejection ("domains we did not know existed")
               grouped by business unit. Reachability = share of users that can sign in at all.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import secrets
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import yaml

from evalkit.io import read_jsonl
from evalkit.report import md_table, pct

PROVIDER_KINDS = ("google_workspace", "entra", "okta")
_EMAIL_RX = re.compile(r"^[a-z0-9!#$%&'*/=?^_`{|}~.+-]+@([a-z0-9-]+\.)+[a-z]{2,}$")


class InvalidEmail(ValueError):
    pass


def normalise_email(raw: str) -> str:
    """trim, lowercase, strip +tag; raise InvalidEmail if the result is not local@domain.tld."""
    s = (raw or "").strip().lower()
    if s.count("@") != 1:
        raise InvalidEmail(f"not an email address: {raw!r}")
    local, domain = s.split("@")
    local = local.split("+", 1)[0]
    s = f"{local}@{domain}"
    if not local or not _EMAIL_RX.match(s):
        raise InvalidEmail(f"not an email address: {raw!r}")
    return s


def domain_of(email: str) -> str:
    return email.rsplit("@", 1)[1]


def levenshtein(a: str, b: str, cap: int = 2) -> int:
    """Edit distance with early exit once every cell in a row exceeds `cap`."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------- registry

@dataclass(frozen=True)
class Provider:
    id: str
    kind: str                       # google_workspace | entra | okta | saml:<name>
    business_unit: str
    domains: tuple[str, ...]
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in PROVIDER_KINDS and not self.kind.startswith("saml:"):
            raise ValueError(f"provider {self.id}: unknown kind {self.kind!r}")


@dataclass(frozen=True)
class Acquisition:
    domain: str
    acquired_by: str                # a domain owned by a provider
    federated: bool
    business_unit: str = ""


class IdentityRegistry:
    def __init__(self, providers: Sequence[Provider], acquired: Sequence[Acquisition] = (),
                 disposable: Iterable[str] = (), default_method: str = "magic_link"):
        self.providers = {p.id: p for p in providers}
        self.acquired = {a.domain.lower(): a for a in acquired}
        self.disposable = {d.lower() for d in disposable}
        self.default_method = default_method
        self._by_domain: dict[str, Provider] = {}
        self._primary = {d.lower() for p in providers for d in p.domains}
        for p in providers:
            for d in p.domains + p.aliases:
                if d.lower() in self._by_domain:
                    raise ValueError(f"domain {d} claimed by two providers")
                self._by_domain[d.lower()] = p
        for a in self.acquired.values():
            if a.acquired_by.lower() not in self._by_domain:
                raise ValueError(f"{a.domain}: acquired_by {a.acquired_by} is not a provider domain")

    @classmethod
    def from_yaml(cls, path: str | Path) -> IdentityRegistry:
        cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        providers = [Provider(p["id"], p["kind"], p.get("business_unit", ""), tuple(p.get("domains", [])),
                              tuple(p.get("aliases", []))) for p in cfg.get("providers", [])]
        acquired = [Acquisition(a["domain"], a["acquired_by"], bool(a.get("federated", False)), a.get("business_unit", ""))
                    for a in cfg.get("acquired", [])]
        return cls(providers, acquired, cfg.get("disposable_domains", []), cfg.get("default_method", "magic_link"))

    def provider_for(self, domain: str) -> Provider | None:
        return self._by_domain.get(domain.lower())

    def known_domains(self) -> list[str]:
        return sorted(set(self._by_domain) | set(self.acquired))

    def is_known(self, domain: str) -> bool:
        return domain.lower() in self._by_domain or domain.lower() in self.acquired

    def is_disposable(self, domain: str) -> bool:
        return domain.lower() in self.disposable

    def suggest(self, domain: str, max_distance: int = 1) -> str | None:
        """Closest known domain within `max_distance` edits, or None. Ties prefer a provider's
        primary domain over aliases / acquired brands, then alphabetical order."""
        best = None
        for known in self.known_domains():
            d = levenshtein(domain, known, cap=max_distance)
            key = (d, known not in self._primary, known)
            if d <= max_distance and (best is None or key < best):
                best = key
        return best[2] if best else None


class PasskeyStore:
    def __init__(self, emails: Iterable[str] = ()):
        self.emails = {normalise_email(e) for e in emails}

    @classmethod
    def from_jsonl(cls, path: str | Path) -> PasskeyStore:
        return cls(r["email"] for r in read_jsonl(path))

    def has(self, email: str) -> bool:
        return normalise_email(email) in self.emails

    def register(self, email: str) -> None:
        self.emails.add(normalise_email(email))


# --------------------------------------------------------------------------- router

@dataclass
class Route:
    email: str                       # normalised
    method: str                      # passkey | idp | magic_link | rejected
    provider: str | None             # provider id for idp, "passkey", "magic_link" or None
    kind: str | None                 # provider kind for idp routes
    reason: str
    fallbacks: list[str] = field(default_factory=list)
    suggestion: str | None = None
    business_unit: str | None = None
    domain_known: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Router:
    def __init__(self, registry: IdentityRegistry, passkeys: PasskeyStore | None = None):
        self.registry = registry
        self.passkeys = passkeys or PasskeyStore()

    def _idp_route(self, email: str, domain: str) -> tuple[Provider, str] | None:
        """(provider, reason) for a direct or federated IdP route, else None."""
        p = self.registry.provider_for(domain)
        if p:
            how = "alias of" if domain in p.aliases else "owned by"
            return p, f"domain {domain} {how} {p.id} ({p.kind})"
        a = self.registry.acquired.get(domain)
        if a and a.federated:
            parent = self.registry.provider_for(a.acquired_by)
            if parent:
                return parent, f"{domain} acquired by {a.acquired_by}, federated into {parent.id} ({parent.kind})"
        return None

    def resolve(self, raw_email: str) -> Route:
        email = normalise_email(raw_email)
        domain = domain_of(email)
        reg = self.registry
        if reg.is_disposable(domain):
            return Route(email, "rejected", None, None, f"{domain} is a disposable mail domain", [],
                         None, None, False)
        idp = self._idp_route(email, domain)
        acq = reg.acquired.get(domain)
        bu = idp[0].business_unit if idp else (acq.business_unit if acq else None)
        known = reg.is_known(domain)
        suggestion = None if known else reg.suggest(domain)
        fallbacks: list[str] = []
        if idp:
            fallbacks.append(f"idp:{idp[0].id}")
        fallbacks.append("magic_link")
        if self.passkeys.has(email):
            return Route(email, "passkey", "passkey", None, "passkey registered for this email", fallbacks,
                         suggestion, bu, known)
        if idp:
            return Route(email, "idp", idp[0].id, idp[0].kind, idp[1], fallbacks[1:], suggestion, bu, known)
        if acq and not acq.federated:
            reason = f"{domain} acquired by {acq.acquired_by} but not federated; magic link until SSO lands"
        elif suggestion:
            reason = f"unknown domain {domain}; did you mean {suggestion}?"
        else:
            reason = f"unknown domain {domain}; magic link fallback"
        return Route(email, "magic_link", "magic_link", None, reason, [], suggestion, bu, known)


# --------------------------------------------------------------------------- magic link

class EmailSender(Protocol):
    def send(self, to: str, subject: str, body: str) -> None: ...


class RecordingSender:
    def __init__(self) -> None:
        self.sent: list[dict[str, str]] = []

    def send(self, to: str, subject: str, body: str) -> None:
        self.sent.append({"to": to, "subject": subject, "body": body})


class StdoutSender:
    def send(self, to: str, subject: str, body: str) -> None:
        print(f"[email] to={to} subject={subject!r}\n{body}")


@dataclass
class VerifyResult:
    ok: bool
    email: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class MagicLink:
    """HMAC-signed, single-use, expiring tokens. `clock` returns epoch seconds; `used` is the
    nonce set (persist it with `to_state` / `load_state` across processes)."""

    def __init__(self, secret: str | bytes, clock: Callable[[], float], ttl_s: float = 900.0,
                 sender: EmailSender | None = None, link_base: str = "https://signin.example.internal/magic",
                 nonce: Callable[[], str] | None = None):
        self.secret = secret.encode() if isinstance(secret, str) else secret
        self.clock = clock
        self.ttl_s = float(ttl_s)
        self.sender = sender
        self.link_base = link_base
        self.nonce = nonce or (lambda: secrets.token_urlsafe(12))
        self.used: set[str] = set()

    def _sign(self, payload: bytes) -> str:
        return _b64(hmac.new(self.secret, payload, hashlib.sha256).digest())

    def issue(self, email: str) -> str:
        email = normalise_email(email)
        iat = self.clock()
        payload = json.dumps({"email": email, "iat": iat, "exp": iat + self.ttl_s, "nonce": self.nonce()},
                             sort_keys=True).encode()
        token = f"{_b64(payload)}.{self._sign(payload)}"
        if self.sender:
            self.sender.send(email, "Your sign-in link",
                             f"Click to sign in (valid {int(self.ttl_s // 60)} min, single use):\n"
                             f"{self.link_base}?token={token}")
        return token

    def verify(self, token: str) -> VerifyResult:
        try:
            p64, sig = token.strip().split(".", 1)
            payload = _unb64(p64)
        except (ValueError, TypeError):
            return VerifyResult(False, None, "malformed token")
        if not hmac.compare_digest(self._sign(payload), sig):
            return VerifyResult(False, None, "bad signature")
        data = json.loads(payload)
        now = self.clock()
        if now > data["exp"]:
            return VerifyResult(False, data["email"], f"expired {now - data['exp']:.0f}s ago")
        if data["nonce"] in self.used:
            return VerifyResult(False, data["email"], "token already used")
        self.used.add(data["nonce"])
        return VerifyResult(True, data["email"], "ok")

    def to_state(self) -> dict[str, Any]:
        return {"used": sorted(self.used)}

    def load_state(self, st: dict[str, Any]) -> None:
        self.used = set(st.get("used", []))


# --------------------------------------------------------------------------- audit

def audit(users: Sequence[dict[str, Any]], router: Router) -> dict[str, Any]:
    """Reachability report over a user directory. Each user: {email, business_unit, country}."""
    by_method: Counter = Counter()
    by_provider: Counter = Counter()
    by_bu_method: dict[str, Counter] = defaultdict(Counter)
    unknown: dict[str, dict[str, Any]] = {}
    invalid: list[str] = []
    routes: list[dict[str, Any]] = []
    for u in users:
        bu = u.get("business_unit", "unknown")
        try:
            r = router.resolve(u["email"])
        except InvalidEmail:
            invalid.append(u["email"])
            by_method["invalid"] += 1
            by_bu_method[bu]["invalid"] += 1
            continue
        routes.append({"email": r.email, "method": r.method, "provider": r.provider, "business_unit": bu})
        by_method[r.method] += 1
        by_bu_method[bu][r.method] += 1
        if r.method == "idp":
            by_provider[f"{r.provider} ({r.kind})"] += 1
        elif r.method == "passkey":
            by_provider["passkey"] += 1
        if not r.domain_known:
            d = domain_of(r.email)
            e = unknown.setdefault(d, {"domain": d, "users": 0, "rejected": r.method == "rejected",
                                       "suggestion": r.suggestion, "business_units": Counter(), "sample": []})
            e["users"] += 1
            e["business_units"][bu] += 1
            if len(e["sample"]) < 3:
                e["sample"].append(r.email)
    n = len(users)
    reachable = n - by_method["rejected"] - by_method["invalid"]
    unknown_users = sum(e["users"] for e in unknown.values())
    by_bu_unknown: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in unknown.values():
        for bu, c in e["business_units"].items():
            by_bu_unknown[bu].append({"domain": e["domain"], "users": c, "rejected": e["rejected"],
                                      "suggestion": e["suggestion"]})
    for rows in by_bu_unknown.values():
        rows.sort(key=lambda x: (-x["users"], x["domain"]))
    return {
        "n_users": n, "reachable": reachable, "reachability": reachable / n if n else 0.0,
        "by_method": dict(sorted(by_method.items())), "by_provider": dict(sorted(by_provider.items())),
        "by_business_unit": {bu: dict(sorted(c.items())) for bu, c in sorted(by_bu_method.items())},
        "unknown_domains": sorted(({**e, "business_units": dict(e["business_units"])} for e in unknown.values()),
                                  key=lambda e: (-e["users"], e["domain"])),
        "unknown_users": unknown_users, "unknown_share": unknown_users / n if n else 0.0,
        "unknown_by_business_unit": dict(sorted(by_bu_unknown.items())),
        "invalid": invalid, "routes": routes,
    }


def render_audit(rep: dict[str, Any]) -> str:
    out = ["# Sign-in reachability audit", "",
           (f"{rep['n_users']} users; {rep['reachable']} reachable ({pct(rep['reachability'])}); "
           f"{rep['unknown_users']} on domains the registry does not know ({pct(rep['unknown_share'])})."), "",
           "## By method", md_table(["method", "users", "share"],
                                    [[m, c, pct(c / rep["n_users"])] for m, c in rep["by_method"].items()]), "",
           "## By provider", md_table(["provider", "users"], [[p, c] for p, c in rep["by_provider"].items()]), "",
           "## By business unit",
           md_table(["business unit"] + list(rep["by_method"]),
                    [[bu] + [c.get(m, 0) for m in rep["by_method"]] for bu, c in rep["by_business_unit"].items()]), "",
           "## Domains we did not know existed (would fall to magic link or be rejected)"]
    for bu, items in rep["unknown_by_business_unit"].items():
        out.append(f"### {bu}")
        out.append(md_table(["domain", "users", "outcome", "did you mean"],
                            [[i["domain"], i["users"], "rejected (disposable)" if i["rejected"] else "magic link",
                              i["suggestion"] or "-"] for i in items]))
        out.append("")
    if rep["invalid"]:
        out += ["## Invalid addresses", ", ".join(rep["invalid"]), ""]
    return "\n".join(out) + "\n"


def iso_now() -> float:
    return datetime.now(UTC).timestamp()
