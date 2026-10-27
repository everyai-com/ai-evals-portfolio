"""Generate data/users.jsonl: ~500 users across 4 business units, 12% on domains the registry
does not know (the acquisitions nobody told IT about, contractors, personal mail).

    python projects/p19_signin_router/data/make_users.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
N = 500
UNKNOWN_SHARE = 0.12
# (domain, business_unit, country, weight)
KNOWN = [
    ("northwire.com", "telecom", "CA", 22), ("northwire.ca", "telecom", "CA", 4), ("northwiremobility.com", "telecom", "CA", 2),
    ("pebblemobile.ca", "telecom", "CA", 6), ("fieldco.ca", "telecom", "CA", 4), ("plainsmobile.ca", "telecom", "CA", 3),
    ("northwirehealth.com", "health", "CA", 14), ("northwirehealth.ca", "health", "CA", 4), ("careline.io", "health", "US", 8),
    ("clinicpath.com", "health", "US", 3), ("agrinorth.com", "agriculture", "CA", 10), ("ledgerly.com", "finance", "CA", 8),
]
UNKNOWN = [
    ("quickpay-labs.com", "finance", "US", 5),       # acquisition nobody told IT about
    ("northwire-business.com", "telecom", "CA", 4),  # a mail domain IT did not know existed
    ("agrinorth-seeds.ca", "agriculture", "CA", 3),
    ("contractor-pool.net", "telecom", "IN", 3),
    ("gmail.com", "health", "CA", 2),
    ("outlook.com", "finance", "CA", 1),
    ("mailinator.com", "telecom", "CA", 1),          # disposable: rejected, not reachable at all
]


def make_users(seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    n_unknown = int(round(N * UNKNOWN_SHARE))
    rows = []

    def draw(pool, count, start):
        w = np.array([p[3] for p in pool], float)
        w /= w.sum()
        for i in range(count):
            dom, bu, country, _ = pool[int(rng.choice(len(pool), p=w))]
            rows.append({"email": f"user-{start + i}@{dom}", "business_unit": bu, "country": country})

    draw(KNOWN, N - n_unknown, 1)
    draw(UNKNOWN, n_unknown, N - n_unknown + 1)
    # a few hand-picked people with passkeys / odd formatting
    rows[0] = {"email": "Ana.Ruiz@northwire.com", "business_unit": "telecom", "country": "CA"}
    rows[1] = {"email": "ben.okafor+oncall@careline.io", "business_unit": "health", "country": "US"}
    rows[2] = {"email": " chen.wei@plainsmobile.ca ", "business_unit": "telecom", "country": "CA"}
    rows[3] = {"email": "dana.li@fieldco.ca", "business_unit": "telecom", "country": "CA"}
    rng.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = make_users()
    path = HERE / "users.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    print(f"wrote {len(rows)} users to {path}")
