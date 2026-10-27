"""Metamorphic relations: input pairs whose expected relation is known, so follow-up cases get
labels without a human. relation(expected_source) -> expected_follow_up."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any

import numpy as np

from evalkit.io import stable_hash
from projects.p12_synthetic_edgecases.spec import parse_iso, replace_date, replace_qty


@dataclass
class MetamorphicPair:
    source_id: str
    source_input: str
    follow_up_input: str
    relation: str              # name of the relation on expected labels
    expected: dict[str, Any] | None
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def same(e: dict[str, Any]) -> dict[str, Any]:
    return dict(e)


def qty_doubled(e: dict[str, Any]) -> dict[str, Any]:
    return {**e, "qty": e["qty"] * 2}


def date_plus_one(e: dict[str, Any]) -> dict[str, Any]:
    return {**e, "date": (parse_iso(e["date"]) + timedelta(days=1)).isoformat()}


RELATIONS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "paraphrase": same, "add_irrelevant": same, "reorder_clauses": same,
    "scale_qty": qty_doubled, "shift_date": date_plus_one,
}

PARAPHRASES = ["We need {qty} units of {sku} shipped on {date}.", "Kindly dispatch {qty} of {sku}; delivery date {date}.",
               "Could you ship {sku} (qty {qty}) on {date}?", "Requesting {qty} units, part {sku}, to go out {date}."]
IRRELEVANT = ["Thanks, and have a nice day.", "P.S. the weather is lovely here.", "Our PO number will follow by email."]


def _rng(seed: int, rel: str, pid: str) -> np.random.Generator:
    return np.random.default_rng(int(stable_hash([seed, rel, pid])[:8], 16))


def generate_pairs(case: dict[str, Any], seed: int = 0) -> list[MetamorphicPair]:
    e = case.get("expected")
    if not isinstance(e, dict) or "error" in e:
        return []
    s, pid = case["input"], case["id"]
    out = []
    r = _rng(seed, "paraphrase", pid)
    tpl = PARAPHRASES[int(r.integers(0, len(PARAPHRASES)))]
    out.append(MetamorphicPair(pid, s, tpl.format(qty=e["qty"], sku=e["sku"], date=e["date"]), "paraphrase", same(e),
                               "paraphrased request -> same answer"))
    r = _rng(seed, "add_irrelevant", pid)
    out.append(MetamorphicPair(pid, s, s + " " + IRRELEVANT[int(r.integers(0, len(IRRELEVANT)))], "add_irrelevant", same(e),
                               "irrelevant sentence appended -> same answer"))
    m = re.search(r"\bon\b|\bby\b|\bdelivered\b|,\s*ship date\b|\bdispatch on\b", s)
    if m:
        out.append(MetamorphicPair(pid, s, f"On {e['date']}: " + s[: m.start()].strip().rstrip(","), "reorder_clauses", same(e),
                                   "date clause moved to the front -> same answer"))
    if e["qty"] * 2 <= 10000:
        t = replace_qty(s, e["qty"], str(e["qty"] * 2))
        if t:
            out.append(MetamorphicPair(pid, s, t, "scale_qty", qty_doubled(e), "qty doubled in text -> qty doubled in answer"))
    t = replace_date(s, date_plus_one(e)["date"])
    if t:
        out.append(MetamorphicPair(pid, s, t, "shift_date", date_plus_one(e), "date +1 day in text -> date +1 day in answer"))
    return out


def relation_holds(pair: MetamorphicPair, source_expected: dict[str, Any]) -> bool:
    return RELATIONS[pair.relation](source_expected) == pair.expected
