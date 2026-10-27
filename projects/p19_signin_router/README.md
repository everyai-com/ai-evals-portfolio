# p19 Email-first sign-in router

**Mistake it answers:** "Sign in with Google" only — the company is "a bunch of companies in a
trench coat" with 30 ways to sign in and more than a dozen mail domains the builders did not know
existed, so a popular platform locked people out on launch day. **Lesson encoded:** one email box;
the system picks a passkey, the company's own IdP, or a magic link sent by email — and a
reachability audit over the user directory is run *before* launch. **Tests that prove it:**
`test_demo_audit_finds_all_unknown_domain_users` (every one of the 12 % unknown-domain users in
the demo directory is listed, grouped by business unit) and
`test_precedence_passkey_over_idp_over_magic_link`.

## What
* `IdentityRegistry` from `data/providers.yaml`: domains → provider (`google_workspace`, `entra`,
  `okta`, `saml:<name>`), `aliases` (historical mail domains on the same workspace), `acquired`
  relationships (`acquired_by` + `federated`) and a disposable-domain blocklist. `PasskeyStore`
  (which emails have a registered passkey) from `data/passkeys.jsonl`.
* `Router.resolve(email) -> Route(method, provider, kind, reason, fallbacks, suggestion)`:

| step | condition | method |
|---|---|---|
| 0 | domain on the disposable blocklist | `rejected` (no fallbacks) |
| 1 | passkey registered for the normalised email | `passkey` (fallbacks: the IdP route if any, then magic link) |
| 2 | domain or alias owned by a provider | `idp` via that provider |
| 3 | domain acquired by a provider's company with `federated: true` | `idp` via the parent's provider |
| 4 | anything else (unknown domain, acquired but not federated) | `magic_link`; unknown domains within edit distance 1 of a known domain carry a *did you mean* `suggestion` |

  Normalisation: trim, lowercase, drop `+tag`; malformed input raises `InvalidEmail`.
* `MagicLink(secret, clock, ttl_s)`: `issue(email)` returns `base64url(payload).base64url(HMAC-SHA256)`
  and sends it through an `EmailSender` (`RecordingSender` in tests, `StdoutSender` in the CLI);
  `verify(token)` checks signature (constant-time), expiry on the injected clock and single use
  (nonce set, persisted in `state/magic_links.json`).
* `audit(users, router)`: routes every user of `data/users.jsonl` (email, business unit, country),
  counts by method / provider / business unit, and lists **the domains that would fall to magic
  link or be rejected, grouped by business unit** — the report that would have caught the lockout.

## Why
A sign-in page that assumes one IdP is a bet that the org chart is simple. It never is: brands keep
their own workspaces, acquisitions keep their Okta for years, contractors have no company mail at
all. Asking for the email first turns "which button do I press" into a lookup the platform owns,
and the audit turns "we did not know that domain existed" into a table you read before launch.

## How to run
```bash
python -m projects.p19_signin_router.cli resolve --email someone@example.com [--json]
python -m projects.p19_signin_router.cli audit --users projects/p19_signin_router/data/users.jsonl [--json]
python -m projects.p19_signin_router.cli magic-link issue --email x@plainsmobile.ca --state state/
python -m projects.p19_signin_router.cli magic-link verify --token <token> --state state/
python projects/p19_signin_router/data/make_users.py          # regenerate the 500-user directory
.venv/bin/pytest projects/p19_signin_router -q
```
Exit codes: `resolve` 1 = rejected, 2 = not an email; `audit` 1 = unknown domains found;
`magic-link verify` 1 = rejected (used / expired / bad signature). `--secret` or `$P19_MAGIC_SECRET`
sets the HMAC key (a dev default is used otherwise); `--now` fixes the clock for reproducible runs.

## Sample output
Real run on the committed `data/providers.yaml` (8 known domains + 3 acquired brands across 4
business units), `data/passkeys.jsonl` and `data/users.jsonl` (500 users, 60 on unknown domains).

```
$ python -m projects.p19_signin_router.cli resolve --email 'Ana.Ruiz@northwire.com'
ana.ruiz@northwire.com -> passkey
  reason: passkey registered for this email
  fallbacks: idp:northwire-google, magic_link
(exit 0)
$ python -m projects.p19_signin_router.cli resolve --email 'ben.okafor+oncall@careline.io'
ben.okafor@careline.io -> passkey
  reason: passkey registered for this email
  fallbacks: idp:careline-okta, magic_link
(exit 0)
$ python -m projects.p19_signin_router.cli resolve --email 'x@fieldco.ca'
x@fieldco.ca -> idp via northwire-google [google_workspace]
  reason: fieldco.ca acquired by northwire.com, federated into northwire-google (google_workspace)
  fallbacks: magic_link
(exit 0)
$ python -m projects.p19_signin_router.cli resolve --email 'x@plainsmobile.ca'
x@plainsmobile.ca -> magic_link
  reason: plainsmobile.ca acquired by northwire.com but not federated; magic link until SSO lands
(exit 0)
$ python -m projects.p19_signin_router.cli resolve --email 'x@clinicpath.com'
x@clinicpath.com -> idp via northwire-health-entra [entra]
  reason: clinicpath.com acquired by northwirehealth.com, federated into northwire-health-entra (entra)
  fallbacks: magic_link
(exit 0)
$ python -m projects.p19_signin_router.cli resolve --email 'x@nortwire.com'
x@nortwire.com -> magic_link
  reason: unknown domain nortwire.com; did you mean northwire.com?
  did you mean: northwire.com
(exit 0)
$ python -m projects.p19_signin_router.cli resolve --email 'x@quickpay-labs.com'
x@quickpay-labs.com -> magic_link
  reason: unknown domain quickpay-labs.com; magic link fallback
(exit 0)
$ python -m projects.p19_signin_router.cli resolve --email 'x@mailinator.com'
x@mailinator.com -> rejected
  reason: mailinator.com is a disposable mail domain
(exit 1)
$ python -m projects.p19_signin_router.cli audit --users projects/p19_signin_router/data/users.jsonl
# Sign-in reachability audit

500 users; 498 reachable (99.6%); 60 on domains the registry does not know (12.0%).

## By method
| method | users | share |
|---|---|---|
| idp | 426 | 85.2% |
| magic_link | 66 | 13.2% |
| passkey | 6 | 1.2% |
| rejected | 2 | 0.4% |

## By provider
| provider | users |
|---|---|
| agrinorth-adfs (saml:agrinorth-adfs) | 56 |
| careline-okta (okta) | 49 |
| ledgerly-entra (entra) | 54 |
| northwire-google (google_workspace) | 145 |
| northwire-health-entra (entra) | 98 |
| passkey | 6 |
| pebble-google (google_workspace) | 24 |

## By business unit
| business unit | idp | magic_link | passkey | rejected |
|---|---|---|---|---|
| agriculture | 56 | 17 | 0 | 0 |
| finance | 54 | 13 | 0 | 0 |
| health | 147 | 7 | 2 | 0 |
| telecom | 169 | 29 | 4 | 2 |

## Domains we did not know existed (would fall to magic link or be rejected)
### agriculture
| domain | users | outcome | did you mean |
|---|---|---|---|
| agrinorth-seeds.ca | 17 | magic link | - |

### finance
| domain | users | outcome | did you mean |
|---|---|---|---|
| quickpay-labs.com | 12 | magic link | - |
| outlook.com | 1 | magic link | - |

### health
| domain | users | outcome | did you mean |
|---|---|---|---|
| gmail.com | 7 | magic link | - |

### telecom
| domain | users | outcome | did you mean |
|---|---|---|---|
| northwire-business.com | 12 | magic link | - |
| contractor-pool.net | 9 | magic link | - |
| mailinator.com | 2 | rejected (disposable) | - |


(exit 1: unknown domains found)
$ python -m projects.p19_signin_router.cli magic-link issue --email x@plainsmobile.ca --state state/ --now 1700000000
[email] to=x@plainsmobile.ca subject='Your sign-in link'
Click to sign in (valid 15 min, single use):
https://signin.example.internal/magic?token=eyJlbWFpbCI6ICJ4QHBsYWluc21vYmlsZS5jYSIsICJleHAiOiAxNzAwMDAwOTAwLjAsICJpYXQiOiAxNzAwMDAwMDAwLjAsICJub25jZSI6ICJHQUR1REtkMFlqZVNlalRlIn0.-2ZCRr_3po0NX9WvaMJP58MgLBJr4QGX61Ldk0_6msw
token: eyJlbWFpbCI6ICJ4QHBsYWluc21vYmlsZS5jYSIsICJleHAiOiAxNzAwMDAwOTAwLjAsICJpYXQiOiAxNzAwMDAwMDAwLjAsICJub25jZSI6ICJHQUR1REtkMFlqZVNlalRlIn0.-2ZCRr_3po0NX9WvaMJP58MgLBJr4QGX61Ldk0_6msw
$ python -m projects.p19_signin_router.cli magic-link verify --token $TOKEN --state state/ --now 1700000300
OK: ok (x@plainsmobile.ca)
(exit 0)
$ python -m projects.p19_signin_router.cli magic-link verify --token $TOKEN --state state/ --now 1700000301
REJECTED: token already used (x@plainsmobile.ca)
(exit 1)
$ python -m projects.p19_signin_router.cli magic-link verify --token $TOKEN --state state/ --now 1700001000   # a fresh token, 1000 s later
REJECTED: expired 100s ago (x@plainsmobile.ca)
(exit 1)
```
Reading the run: `+oncall` and mixed case resolve to the same passkey; `fieldco.ca` and
`clinicpath.com` are acquired brands federated into their parents' Google / Entra tenants, while
`plainsmobile.ca` is acquired but not federated and gets a magic link with the reason spelled out.
The audit finds all 60 unknown-domain users (12.0 %): `quickpay-labs.com` (a finance acquisition
nobody told IT about), `northwire-business.com` (a telecom mail domain IT did not know existed),
`agrinorth-seeds.ca`, contractors and personal mail — and two disposable addresses that cannot
sign in at all. The magic link verifies once, is refused on replay, and expires after its TTL.

## Design notes
* **Reject before route.** Disposable domains are refused before the passkey check, so a passkey
  registered on a throwaway address cannot bypass the policy.
* **Fallbacks travel with the route.** A passkey route carries the IdP and magic-link alternatives
  so the UI can offer "use another way" without a second lookup; a magic-link route has none,
  which is the signal the audit counts.
* **Never rewrite the email.** A near-miss domain yields a `suggestion`, not a silent correction:
  `nortwire.com` could be a real domain we do not know yet, which is exactly what the audit is for.
  Ties between candidates prefer a provider's primary domain over aliases.
* **Magic link is boring on purpose.** Stdlib `hmac` + `hashlib`, `compare_digest`, an injected
  clock and a nonce set. The secret is the only state that must be shared between the issuing and
  verifying process; the used-nonce set is small because tokens expire.
* **Audit output is grouped by business unit** because that is who can answer "whose domain is
  this?" — the engineering team cannot.
* **evalkit reuse:** `read_jsonl` (passkeys, users), `md_table` / `pct` (audit report). The edit
  distance helper lives in `core.py`; evalkit has no string-similarity utility (gap noted).
