"""Magic link single-use + expiry on an injected clock; audit counts on a hand-built directory and the demo."""
import json
from pathlib import Path

from evalkit.io import read_jsonl
from projects.p19_signin_router.cli import main
from projects.p19_signin_router.core import (
    IdentityRegistry,
    MagicLink,
    PasskeyStore,
    RecordingSender,
    Router,
    audit,
    render_audit,
)

HERE = Path(__file__).resolve().parent.parent


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_magic_link_single_use_and_expiry():
    clock, sender = Clock(), RecordingSender()
    ml = MagicLink("s3cret", clock, ttl_s=600, sender=sender)
    tok = ml.issue("Ana+x@Northwire.com")
    assert len(sender.sent) == 1 and sender.sent[0]["to"] == "ana@northwire.com" and tok in sender.sent[0]["body"]
    first = ml.verify(tok)
    assert first.ok and first.email == "ana@northwire.com"
    second = ml.verify(tok)
    assert not second.ok and second.reason == "token already used"
    tok2 = ml.issue("ana@northwire.com")
    clock.t += 601
    exp = ml.verify(tok2)
    assert not exp.ok and exp.reason.startswith("expired")
    assert not MagicLink("other", clock).verify(ml.issue("ana@northwire.com")).ok      # wrong secret
    assert ml.verify("garbage").reason == "malformed token"
    tampered = ml.issue("ana@northwire.com")
    p, sig = tampered.split(".")
    assert ml.verify(p + "." + sig[:-2] + "AA").reason == "bad signature"
    # used-nonce state survives a restart
    tok3 = ml.issue("ana@northwire.com")
    assert ml.verify(tok3).ok
    ml2 = MagicLink("s3cret", clock, ttl_s=600)
    ml2.load_state(json.loads(json.dumps(ml.to_state())))
    assert ml2.verify(tok3).reason == "token already used"


def test_audit_counts_match_hand_built_directory():
    reg = IdentityRegistry.from_yaml(HERE / "data" / "providers.yaml")
    router = Router(reg, PasskeyStore(["pk@northwire.com", "pk@unknown.org"]))
    users = [
        {"email": "pk@northwire.com", "business_unit": "telecom"},          # passkey
        {"email": "pk@unknown.org", "business_unit": "telecom"},            # passkey, unknown domain
        {"email": "a@northwire.com", "business_unit": "telecom"},           # idp google
        {"email": "b@northwire.ca", "business_unit": "telecom"},            # idp google (alias)
        {"email": "c@fieldco.ca", "business_unit": "telecom"},              # idp google (federated)
        {"email": "d@plainsmobile.ca", "business_unit": "telecom"},         # magic link, known (not federated)
        {"email": "e@careline.io", "business_unit": "health"},              # idp okta
        {"email": "f@clinicpath.com", "business_unit": "health"},           # idp entra (federated)
        {"email": "g@gmail.com", "business_unit": "health"},                # magic link, unknown
        {"email": "h@gmail.com", "business_unit": "finance"},               # magic link, unknown
        {"email": "i@mailinator.com", "business_unit": "finance"},          # rejected
        {"email": "not-an-email", "business_unit": "finance"},              # invalid
    ]
    rep = audit(users, router)
    assert rep["n_users"] == 12
    assert rep["by_method"] == {"idp": 5, "invalid": 1, "magic_link": 3, "passkey": 2, "rejected": 1}
    assert rep["by_provider"] == {"careline-okta (okta)": 1, "northwire-google (google_workspace)": 3,
                                  "northwire-health-entra (entra)": 1, "passkey": 2}
    assert rep["reachable"] == 10
    unknown = {e["domain"]: e for e in rep["unknown_domains"]}
    assert set(unknown) == {"unknown.org", "gmail.com", "mailinator.com"}
    assert unknown["gmail.com"]["users"] == 2 and unknown["gmail.com"]["business_units"] == {"health": 1, "finance": 1}
    assert unknown["mailinator.com"]["rejected"]
    assert rep["unknown_users"] == 4
    assert [d["domain"] for d in rep["unknown_by_business_unit"]["finance"]] == ["gmail.com", "mailinator.com"]
    assert rep["by_business_unit"]["telecom"] == {"idp": 3, "magic_link": 1, "passkey": 2}
    md = render_audit(rep)
    assert "## Domains we did not know existed" in md and "### finance" in md and "not-an-email" in md


def test_demo_audit_finds_all_unknown_domain_users(capsys):
    reg = IdentityRegistry.from_yaml(HERE / "data" / "providers.yaml")
    users = read_jsonl(HERE / "data" / "users.jsonl")
    router = Router(reg, PasskeyStore.from_jsonl(HERE / "data" / "passkeys.jsonl"))
    rep = audit(users, router)
    expected = [u for u in users if not reg.is_known(u["email"].strip().lower().rsplit("@", 1)[1])]
    assert len(expected) == 60 == round(0.12 * len(users))
    assert rep["unknown_users"] == len(expected) and rep["unknown_share"] == 0.12
    assert {u["email"].strip().lower() for u in expected} == {
        r["email"] for r in rep["routes"] if r["provider"] in ("magic_link", None) and
        r["email"].rsplit("@", 1)[1] in {e["domain"] for e in rep["unknown_domains"]}}
    assert rep["invalid"] == []
    passkeys = PasskeyStore.from_jsonl(HERE / "data" / "passkeys.jsonl").emails
    assert rep["by_method"]["passkey"] == len({r["email"] for r in rep["routes"]} & passkeys) >= 4   # ana, ben, chen, dana
    assert any(e["domain"] == "mailinator.com" and e["rejected"] for e in rep["unknown_domains"])
    rc = main(["audit", "--users", str(HERE / "data" / "users.jsonl"), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["unknown_users"] == 60 and "routes" not in out


def test_cli_resolve_and_magic_link(tmp_path, capsys):
    assert main(["resolve", "--email", "Ana.Ruiz@northwire.com"]) == 0
    assert "-> passkey" in capsys.readouterr().out
    assert main(["resolve", "--email", "x@fieldco.ca", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["provider"] == "northwire-google"
    assert main(["resolve", "--email", "x@mailinator.com"]) == 1
    assert main(["resolve", "--email", "nope"]) == 2
    capsys.readouterr()
    st = str(tmp_path / "state")
    assert main(["magic-link", "issue", "--email", "x@plainsmobile.ca", "--state", st, "--now", "100", "--secret", "k"]) == 0
    token = [line for line in capsys.readouterr().out.splitlines() if line.startswith("token: ")][0].split(" ", 1)[1]
    assert main(["magic-link", "verify", "--token", token, "--state", st, "--now", "200", "--secret", "k"]) == 0
    assert main(["magic-link", "verify", "--token", token, "--state", st, "--now", "200", "--secret", "k"]) == 1
    assert "already used" in capsys.readouterr().out
    assert main(["magic-link", "issue", "--email", "x@plainsmobile.ca", "--state", st, "--now", "100", "--secret", "k"]) == 0
    token2 = [line for line in capsys.readouterr().out.splitlines() if line.startswith("token: ")][0].split(" ", 1)[1]
    assert main(["magic-link", "verify", "--token", token2, "--state", st, "--now", "100000", "--secret", "k"]) == 1
    assert "expired" in capsys.readouterr().out
