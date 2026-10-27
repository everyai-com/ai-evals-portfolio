"""Model-driven generation: ask a frontier model for N new OOD cases as JSON, validate each row."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from evalkit.llm import LLMClient
from projects.p12_synthetic_edgecases.spec import TaskSpec, validate_expected

SYSTEM = ("You are an adversarial test designer. Given a task spec, examples, mutation operators and a "
          "failure taxonomy, invent NEW out-of-distribution inputs that a parser is likely to get wrong. "
          "Reply with a JSON array only.")


def build_prompt(spec: TaskSpec, examples: list[dict[str, Any]], operators: list[str], n: int) -> str:
    ex = "\n".join(json.dumps({"input": e["input"], "expected": e["expected"]}, ensure_ascii=False) for e in examples[:8])
    return (f"TASK: {spec.task}\n{spec.description}\nOUTPUT SCHEMA: {json.dumps(spec.output_schema)}\n"
            f"ERROR REASONS: {spec.error_reasons}\nINVARIANTS:\n- " + "\n- ".join(spec.invariants) +
            f"\n\nEXAMPLES:\n{ex}\n\nOPERATORS ALREADY COVERED BY RULES: {operators}\n"
            f"FAILURE TAXONOMY: {spec.failure_taxonomy}\n\n"
            f"Produce {n} new cases as a JSON array of objects with keys: input (string), expected (object matching "
            f"the schema, or {{\"error\": reason}}, or null if a human must decide), why_hard (string), "
            f"category (one of the taxonomy). Prefer categories the rules cannot reach.")


@dataclass
class LLMGenerator:
    client: LLMClient
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def generate(self, spec: TaskSpec, examples: list[dict[str, Any]], operators: list[str], n: int = 10,
                 seed: int = 0) -> list[dict[str, Any]]:
        prompt = build_prompt(spec, examples, operators, n)
        resp = self.client.complete(prompt, system=SYSTEM, temperature=0.7, max_tokens=4096, seed=seed)
        try:
            rows = resp.json()
        except (ValueError, json.JSONDecodeError) as e:
            self.rejected.append({"row": resp.text[:200], "reason": f"unparseable: {e}"})
            return []
        if not isinstance(rows, list):
            self.rejected.append({"row": rows, "reason": "not a JSON array"})
            return []
        out = []
        for row in rows:
            reason = self._reject_reason(row, spec)
            if reason:
                self.rejected.append({"row": row, "reason": reason})
                continue
            out.append({"input": row["input"], "expected": row.get("expected"), "rationale": row["why_hard"],
                        "operator_name": "llm", "category": row["category"], "parent_id": None})
        return out

    @staticmethod
    def _reject_reason(row: Any, spec: TaskSpec) -> str | None:
        if not isinstance(row, dict):
            return "row is not an object"
        if not isinstance(row.get("input"), str) or not row["input"].strip():
            return "missing/empty input"
        if not isinstance(row.get("why_hard"), str) or not row["why_hard"]:
            return "missing why_hard"
        if row.get("category") not in spec.failure_taxonomy:
            return f"category {row.get('category')!r} not in taxonomy"
        errs = validate_expected(row.get("expected"), spec)
        return "; ".join(errs) if errs else None


DEMO_BATCH = [
    {"input": "Ship a dozen of SKU-1234 on 2026-03-14", "expected": {"qty": 12, "sku": "SKU-1234", "date": "2026-03-14"},
     "why_hard": "quantity expressed as an idiom", "category": "quantity_words"},
    {"input": "Ship 12 units of sku-1234 on 2026-03-14", "expected": {"qty": 12, "sku": "SKU-1234", "date": "2026-03-14"},
     "why_hard": "lower-case SKU prefix", "category": "date_format"},
    {"input": "Ship 12 units of SKU-1234 on 14/03/2026", "expected": None,
     "why_hard": "day-first date is ambiguous without locale", "category": "date_format"},
    {"input": "Ship twelve units of SKU-1234 on March 14th, 2026", "expected": {"qty": 12, "sku": "SKU-1234", "date": "2026-03-14"},
     "why_hard": "number word plus ordinal suffix", "category": "quantity_words"},
    {"input": "Ship 12 units of SKU-1234 on 2026-03-14T00:00:00Z", "expected": {"qty": 12, "sku": "SKU-1234", "date": "2026-03-14"},
     "why_hard": "ISO datetime instead of date", "category": "date_format"},
    {"input": "Ship 1,2 units of SKU-1234 on 2026-03-14", "expected": {"error": "non_integer_qty"},
     "why_hard": "decimal comma", "category": "numeric_precision"},
    {"input": "Ship 12 units of SKU-1234 on the 29th of February 2026", "expected": {"error": "invalid_date"},
     "why_hard": "non-leap-year Feb 29 spelled out", "category": "temporal_edge"},
    {"input": "Ship 12 units of SKU-1234 on 2026-03-14, no wait, 2026-03-15", "expected": None,
     "why_hard": "self-correction mid-sentence", "category": "contradiction"},
    {"input": "SKU-1234 x12 2026-03-14", "expected": {"qty": 12, "sku": "SKU-1234", "date": "2026-03-14"},
     "why_hard": "telegraphic style, no units word", "category": "paraphrase"},
    {"input": "Ship 12 units of SKU-１２３４ on 2026-03-14", "expected": {"qty": 12, "sku": "SKU-1234", "date": "2026-03-14"},
     "why_hard": "full-width digits inside the SKU", "category": "unicode_digits"},
    {"input": "Ship 12 units of SKU-1234 on 2026-03-14", "expected": {"qty": "12", "sku": "SKU-1234", "date": "2026-03-14"},
     "why_hard": "malformed: qty as string (should be rejected)", "category": "type_confusion"},
    {"input": "", "expected": {"error": "empty_input"}, "why_hard": "malformed: empty input (should be rejected)", "category": "length"},
    {"input": "Ship 12 units of SKU-1234 on 2026-03-14", "expected": None, "why_hard": "x", "category": "not_a_category"},
]


def demo_llm_handler(system: str, prompt: str) -> str:
    """MockLLM handler: a deterministic batch with 10 good rows and 3 malformed ones."""
    return "Here are the cases:\n" + json.dumps(DEMO_BATCH, ensure_ascii=False)
