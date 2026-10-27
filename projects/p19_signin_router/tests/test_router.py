"""Precedence, federation, unknown / disposable / typo domains, normalisation."""
import pytest

from projects.p19_signin_router.core import (
    Acquisition,
    IdentityRegistry,
    InvalidEmail,
    PasskeyStore,
    Provider,
    Router,
    levenshtein,
    normalise_email,
)


def registry():
    return IdentityRegistry(
        providers=[Provider("nw-google", "google_workspace", "telecom", ("northwire.com",), ("northwire.ca",)),
                   Provider("health-entra", "entra", "health", ("northwirehealth.com",)),
                   Provider("careline-okta", "okta", "health", ("careline.io",)),
                   Provider("agri-saml", "saml:agri-adfs", "agriculture", ("agrinorth.com",))],
        acquired=[Acquisition("fieldco.ca", "northwire.com", federated=True, business_unit="telecom"),
                  Acquisition("plainsmobile.ca", "northwire.com", federated=False, business_unit="telecom")],
        disposable=["mailinator.com"])


def router(passkeys=()):
    return Router(registry(), PasskeyStore(passkeys))


def test_precedence_passkey_over_idp_over_magic_link():
    r = router(["ana@northwire.com", "zed@nowhere.org"])
    a = r.resolve("ana@northwire.com")
    assert a.method == "passkey" and a.fallbacks == ["idp:nw-google", "magic_link"]
    b = r.resolve("bob@northwire.com")
    assert (b.method, b.provider, b.kind) == ("idp", "nw-google", "google_workspace") and b.fallbacks == ["magic_link"]
    z = r.resolve("zed@nowhere.org")
    assert z.method == "passkey" and z.fallbacks == ["magic_link"] and not z.domain_known
    assert r.resolve("x@agrinorth.com").kind == "saml:agri-adfs"
    assert r.resolve("x@northwire.ca").provider == "nw-google" and "alias" in r.resolve("x@northwire.ca").reason


def test_acquired_brand_federation():
    r = router()
    fed = r.resolve("dana@fieldco.ca")
    assert fed.method == "idp" and fed.provider == "nw-google" and "federated" in fed.reason
    assert fed.business_unit == "telecom" and fed.domain_known
    not_fed = r.resolve("chen@plainsmobile.ca")
    assert not_fed.method == "magic_link" and "not federated" in not_fed.reason and not_fed.domain_known
    with pytest.raises(ValueError):
        IdentityRegistry([], [Acquisition("x.com", "unknown.com", True)])


def test_unknown_domain_falls_to_magic_link_with_typo_suggestion():
    r = router()
    u = r.resolve("someone@example.com")
    assert u.method == "magic_link" and u.suggestion is None and not u.domain_known and u.fallbacks == []
    t = r.resolve("someone@nortwire.com")          # one deletion from northwire.com
    assert t.method == "magic_link" and t.suggestion == "northwire.com" and "did you mean" in t.reason
    assert r.resolve("x@northwire.co").suggestion == "northwire.com"   # tie with alias northwire.ca: primary wins
    assert r.resolve("x@carelime.io").suggestion == "careline.io"
    assert r.resolve("x@northwire.org").suggestion is None     # two edits away: no hint
    assert levenshtein("kitten", "sitting", cap=5) == 3 and levenshtein("a", "a") == 0


def test_disposable_domain_is_rejected_even_with_passkey():
    r = router(["x@mailinator.com"])
    d = r.resolve("x@mailinator.com")
    assert d.method == "rejected" and d.fallbacks == [] and "disposable" in d.reason


def test_normalisation():
    assert normalise_email("  Ana.Ruiz+Tag@NorthWire.COM ") == "ana.ruiz@northwire.com"
    r = router(["ben@careline.io"])
    assert r.resolve("Ben+oncall@Careline.IO").method == "passkey"     # plus-addressing maps to the same account
    for bad in ("", "nope", "a@b", "@x.com", "a@@b.com", "a b@x.com"):
        with pytest.raises(InvalidEmail):
            normalise_email(bad)
    store = PasskeyStore()
    store.register("X+y@Z.com")
    assert store.has("x@z.com")
