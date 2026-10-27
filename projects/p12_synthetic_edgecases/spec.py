"""Task spec loading, case schema validation and shared text helpers."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TaskSpec:
    task: str
    description: str
    input: dict[str, Any]
    output_schema: dict[str, dict[str, Any]]
    error_reasons: list[str]
    invariants: list[str]
    failure_taxonomy: list[str]
    operator_category: dict[str, str | None] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> TaskSpec:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(task=raw["task"], description=raw.get("description", ""), input=raw.get("input", {}),
                   output_schema=raw["output_schema"], error_reasons=raw.get("error_reasons", []),
                   invariants=raw.get("invariants", []), failure_taxonomy=raw["failure_taxonomy"],
                   operator_category=raw.get("operator_category", {}), path=Path(path))

    def category_for(self, operator: str) -> str | None:
        return self.operator_category.get(operator)


def validate_expected(expected: Any, spec: TaskSpec) -> list[str]:
    """Errors for an expected label: None is allowed (needs review); {"error": reason} is allowed;
    otherwise exactly the schema fields with the right types/ranges/patterns."""
    if expected is None:
        return []
    if not isinstance(expected, dict):
        return ["expected must be a dict or null"]
    if "error" in expected:
        if set(expected) != {"error"} or not isinstance(expected["error"], str):
            return ["error label must be {'error': str}"]
        if spec.error_reasons and expected["error"] not in spec.error_reasons:
            return [f"unknown error reason {expected['error']!r}"]
        return []
    errs = []
    if set(expected) != set(spec.output_schema):
        errs.append(f"expected keys {sorted(expected)} != schema keys {sorted(spec.output_schema)}")
    for k, rule in spec.output_schema.items():
        if k not in expected:
            continue
        v = expected[k]
        if rule["type"] == "int":
            if not isinstance(v, int) or isinstance(v, bool):
                errs.append(f"{k}: not an int")
            elif not rule.get("min", -1e18) <= v <= rule.get("max", 1e18):
                errs.append(f"{k}: {v} outside [{rule.get('min')}, {rule.get('max')}]")
        elif rule["type"] == "string":
            if not isinstance(v, str):
                errs.append(f"{k}: not a string")
            elif rule.get("pattern") and not re.fullmatch(rule["pattern"], v):
                errs.append(f"{k}: {v!r} does not match {rule['pattern']}")
            elif rule.get("format") == "YYYY-MM-DD":
                try:
                    datetime.strptime(v, "%Y-%m-%d")
                except ValueError:
                    errs.append(f"{k}: {v!r} is not a valid YYYY-MM-DD date")
    return errs


def validate_case(case: dict[str, Any], spec: TaskSpec) -> list[str]:
    errs = []
    inp = case.get("input")
    if not isinstance(inp, str):
        errs.append("input must be a string")
    else:
        lo, hi = spec.input.get("min_chars", 0), spec.input.get("max_chars", 10 ** 9)
        if len(inp) > hi:
            errs.append(f"input longer than {hi} chars")
        if len(inp) < lo and case.get("expected") not in (None, {"error": "empty_input"}):
            errs.append("input shorter than min_chars without an empty_input label")
    errs += validate_expected(case.get("expected"), spec)
    if not isinstance(case.get("tags", []), list):
        errs.append("tags must be a list")
    return errs


# ----------------------------------------------------------------------------- text helpers

def normalise_text(text: str) -> str:
    """Lower-case + whitespace collapse. Deliberately *not* NFKC: unicode variants must stay
    distinct from their parents for dedup."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def char_ngrams(text: str, n: int = 5) -> set[str]:
    t = normalise_text(text)
    return {t[i:i + n] for i in range(max(1, len(t) - n + 1))}


def ngram_jaccard(a: str, b: str, n: int = 5) -> float:
    ga, gb = char_ngrams(a, n), char_ngrams(b, n)
    return len(ga & gb) / len(ga | gb) if ga | gb else 1.0


ISO_RX = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
US_RX = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
LONG_RX = re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December) (\d{1,2}), (\d{4})\b")


def find_date_span(text: str) -> tuple[int, int] | None:
    """Span of the first recognised date surface form (ISO, US or 'Month D, YYYY')."""
    for rx in (ISO_RX, US_RX, LONG_RX):
        m = rx.search(text)
        if m:
            return m.span()
    return None


def replace_date(text: str, new: str) -> str | None:
    span = find_date_span(text)
    return text[: span[0]] + new + text[span[1]:] if span else None


def replace_qty(text: str, qty: int, new: str) -> str | None:
    rx = re.compile(rf"(?<![\w-]){qty}(?![\w-])")
    return rx.sub(new, text, count=1) if rx.search(text) else None


def parse_iso(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)
