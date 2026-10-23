"""A tiny deterministic 7-hop research agent used to demonstrate record / replay / bisect.

    plan (llm) -> search (tool) -> read (tool) -> extract (llm) -> compute (tool)
         -> verify (llm) -> answer (llm)

Every LLM hop is a scripted ``MockLLM`` handler that is a *function of its prompt*, so a
corrupted upstream output propagates downstream exactly like it would with a real model.
``make_tools(corrupt_search=True)`` returns a search tool that surfaces the wrong company's
report; the agent then faithfully sums the wrong numbers and gives a wrong final answer.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from evalkit.llm import MockLLM

from projects.p11_counterfactual_replay.core import ExecutionContext

QUESTION = "What is the combined headcount of Acme Corp's three largest offices?"
GOLDEN_ANSWER = "The three largest Acme Corp offices have a combined headcount of 265."

CORPUS: dict[str, str] = {
    "acme-offices-2025": ("Acme Corp office report 2025. Austin headcount=120. Berlin headcount=85. "
                          "Tokyo headcount=60. Lisbon headcount=15."),
    "acme-history": "Acme Corp was founded in 1999 in Austin and opened its Berlin office in 2008.",
    "acme-benefits": "Acme Corp benefits handbook: remote stipend, 25 days leave, office lunch.",
    "zenith-offices-2025": ("Zenith Ltd office report 2025. Denver headcount=300. Oslo headcount=40. "
                            "Pune headcount=210."),
}
COMPANIES = {"acme": "Acme Corp", "zenith": "Zenith Ltd"}


def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def make_tools(corrupt_search: bool = False) -> dict[str, Callable[..., Any]]:
    def search(query: str, k: int = 2) -> list[str]:
        """Lexical overlap ranking, restricted to the company named in the query."""
        q = _tokens(query)
        wanted = {c for c in COMPANIES if c in q} or set(COMPANIES)
        pool = [d for d in CORPUS if d.split("-")[0] in wanted]
        scored = sorted(pool, key=lambda d: (-len(q & _tokens(CORPUS[d])), d))
        return scored[:k]

    def corrupted_search(query: str, k: int = 2) -> list[str]:
        # Simulates a retrieval bug: the wrong company's report ranks first.
        return ["zenith-offices-2025", "acme-history"][:k]

    def read(doc_ids: list[str]) -> dict[str, str]:
        return {d: CORPUS.get(d, "") for d in doc_ids}

    def compute(op: str, values: list[float]) -> float:
        if op != "sum":
            raise ValueError(f"unsupported op {op}")
        return float(sum(values))

    return {"search": corrupted_search if corrupt_search else search, "read": read, "compute": compute}


def _extract(system: str, prompt: str) -> str:
    nums = [int(x) for x in re.findall(r"headcount=(\d+)", prompt)]
    return json.dumps(sorted(nums, reverse=True)[:3])


def _verify(system: str, prompt: str) -> str:
    m = re.search(r"values=(\[.*?\]) total=([\d.]+)", prompt)
    if not m:
        return "MISMATCH"
    values, total = json.loads(m.group(1)), float(m.group(2))
    return "VERIFIED" if abs(sum(values) - total) < 1e-9 else "MISMATCH"


def _answer(system: str, prompt: str) -> str:
    total = float(re.search(r"total=([\d.]+)", prompt).group(1))
    verdict = re.search(r"verdict=(\w+)", prompt).group(1)
    if verdict != "VERIFIED":
        return "I could not verify the total."
    return f"The three largest Acme Corp offices have a combined headcount of {total:g}."


def demo_handlers() -> list[tuple[str, Any]]:
    return [
        (r"^PLAN:", "search for: acme corp office headcount report 2025"),
        (r"^EXTRACT:", _extract),
        (r"^VERIFY:", _verify),
        (r"^ANSWER:", _answer),
    ]


def make_llm(model: str = "mock-system-one") -> MockLLM:
    return MockLLM(model=model, handlers=demo_handlers())


def demo_agent(ctx: ExecutionContext) -> str:
    plan = ctx.llm("plan", f"PLAN: {QUESTION}")
    query = plan.split("search for:", 1)[-1].strip()
    docs = ctx.tool("search", "search", query=query)
    texts = ctx.tool("read", "read", doc_ids=docs)
    extracted = ctx.llm("extract", "EXTRACT: largest three headcounts from " + json.dumps(texts, sort_keys=True))
    values = json.loads(extracted)
    total = ctx.tool("compute", "compute", op="sum", values=values)
    verdict = ctx.llm("verify", f"VERIFY: values={json.dumps(values)} total={total}")
    return ctx.llm("answer", f"ANSWER: question={QUESTION} total={total} verdict={verdict}")


NODE_ORDER = ["plan", "search", "read", "extract", "compute", "verify", "answer"]
