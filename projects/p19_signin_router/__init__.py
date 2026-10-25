"""p19 Email-first sign-in router: passkey -> company IdP -> magic link, plus a reachability audit."""
from projects.p19_signin_router.core import (
    Acquisition,
    IdentityRegistry,
    InvalidEmail,
    MagicLink,
    PasskeyStore,
    Provider,
    RecordingSender,
    Route,
    Router,
    VerifyResult,
    audit,
    levenshtein,
    normalise_email,
    render_audit,
)

__all__ = ["Acquisition", "IdentityRegistry", "InvalidEmail", "MagicLink", "PasskeyStore", "Provider",
           "RecordingSender", "Route", "Router", "VerifyResult", "audit", "levenshtein", "normalise_email",
           "render_audit"]
