"""Context Window Eviction Tester.

Fills an agent's working memory with seeded noise, plants key facts at controlled positions
(with later contradicting updates), then asks probe questions against the rendered context
to measure whether a ``ContextManager`` keeps the right semantic state under a token budget.

Interface::

    class ContextManager:
        name: str
        def add(self, message: Message) -> None
        def render(self, budget_tokens: int, query: str = "") -> str

Reference implementations:

* ``TruncateOldest``   — keep the most recent messages that fit.
* ``SummarizeOldest``  — compress the oldest messages through an ``LLMClient``; the MockLLM
  handler (``summary_handler``) keeps every line containing ``KEY:`` in order.
* ``RetrievalContext`` — keep everything in a store; at render time retrieve the messages
  with the highest BM25 score for the probe query, then fill the remaining budget with the
  most recent messages; output in chronological order so "latest wins" still works.

Metrics per (manager, noise level) scenario, all over the planted facts:

* recall rate        = facts whose key is found in the rendered context / facts
* latest-correct     = facts whose answer equals the LATEST planted value / facts
* position breakdown = recall and latest-correct split by early / middle / late planting
* compression ratio  = rendered tokens / total conversation tokens (mean over probes)
* false retention    = noise tokens in the rendered context / rendered tokens (mean over probes)

Token counting uses ``evalkit.llm.approx_tokens`` (len // 4) throughout.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from evalkit.llm import LLMClient, approx_tokens

# ---------------------------------------------------------------------------
# messages
# ---------------------------------------------------------------------------


@dataclass
class Message:
    role: str
    content: str

    def render(self) -> str:
        return f"[{self.role}] {self.content}"

    @property
    def tokens(self) -> int:
        """Conservative per-message count (includes the joining newline, rounds up) so that a
        budget enforced per message is never exceeded by the rendered string."""
        return approx_tokens(self.render() + "\n") + 1


def render_messages(msgs: list[Message]) -> str:
    return "\n".join(m.render() for m in msgs)


def total_tokens(msgs: list[Message]) -> int:
    return sum(m.tokens for m in msgs)


def fit_newest(msgs: list[Message], budget: int) -> list[Message]:
    """Newest-first greedy fill, returned in chronological order."""
    kept: list[Message] = []
    used = 0
    for m in reversed(msgs):
        if used + m.tokens > budget:
            break
        kept.append(m)
        used += m.tokens
    kept.reverse()
    return kept


# ---------------------------------------------------------------------------
# context managers
# ---------------------------------------------------------------------------


class ContextManager(Protocol):
    name: str

    def add(self, message: Message) -> None: ...

    def render(self, budget_tokens: int, query: str = "") -> str: ...


class TruncateOldest:
    """Sliding window: drop the oldest messages until the rest fit the budget."""
    name = "truncate"

    def __init__(self) -> None:
        self.messages: list[Message] = []

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def render(self, budget_tokens: int, query: str = "") -> str:
        return render_messages(fit_newest(self.messages, budget_tokens))


SUMMARY_SYSTEM = ("You compress conversation history. Keep every line that states a durable fact "
                  "(lines containing 'KEY:'), in their original order. Drop logs and chatter.")


def summary_handler(system: str, prompt: str) -> str:
    """Deterministic MockLLM summariser: keep ``KEY:`` lines, in order, de-duplicated."""
    seen: set[str] = set()
    out: list[str] = []
    for line in prompt.splitlines():
        if "KEY:" not in line:
            continue
        fact = line[line.index("KEY:"):].strip()
        if fact not in seen:
            seen.add(fact)
            out.append(fact)
    return "\n".join(out) if out else "(no durable facts)"


class SummarizeOldest:
    """Compress the oldest ``chunk_fraction`` of the un-summarised messages through an LLM
    whenever the context exceeds the budget. The running summary is itself re-summarised
    together with each new chunk, so facts survive many rounds. State persists across
    ``render`` calls, so repeated probes do not redo the compression."""
    name = "summarize"

    def __init__(self, llm: LLMClient, chunk_fraction: float = 0.5):
        self.llm = llm
        self.chunk_fraction = chunk_fraction
        self.messages: list[Message] = []
        self.summary: Message | None = None
        self.llm_calls = 0

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def _current(self) -> list[Message]:
        return ([self.summary] if self.summary else []) + self.messages

    def render(self, budget_tokens: int, query: str = "") -> str:
        while total_tokens(self._current()) > budget_tokens and len(self.messages) > 1:
            n = max(1, math.ceil(len(self.messages) * self.chunk_fraction))
            chunk, self.messages = self.messages[:n], self.messages[n:]
            prior = self.summary.content if self.summary else ""
            prompt = "SUMMARIZE the following history.\n" + prior + "\n" + render_messages(chunk)
            resp = self.llm.complete(prompt, system=SUMMARY_SYSTEM)
            self.llm_calls += 1
            self.summary = Message("summary", resp.text.strip())
        current = self._current()
        if total_tokens(current) <= budget_tokens:
            return render_messages(current)
        # Summary alone does not fit: keep the summary's newest lines (latest facts) plus whatever
        # recent messages still fit.
        if self.summary:
            lines = self.summary.content.splitlines()
            kept: list[str] = []
            used = approx_tokens("[summary] ")
            for line in reversed(lines):
                if used + approx_tokens(line + "\n") > budget_tokens:
                    break
                kept.append(line)
                used += approx_tokens(line + "\n")
            summary = Message("summary", "\n".join(reversed(kept)))
            rest = fit_newest(self.messages, budget_tokens - summary.tokens)
            return render_messages([summary] + rest)
        return render_messages(fit_newest(self.messages, budget_tokens))


_TOKEN_RX = re.compile(r"[a-z0-9_]+")
STOPWORDS = frozenset(["a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "what", "which", "who", "with", "current", "value"])


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RX.findall(text.lower()) if t not in STOPWORDS]


class RetrievalContext:
    """Store everything; at render time rank messages by BM25 against the query.

    BM25(q, d) = Σ_t idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl)),
    idf(t) = ln((N − n_t + 0.5)/(n_t + 0.5) + 1)   (Robertson/Sparck Jones, k1=1.5, b=0.75).

    Messages with a positive score are taken best-first until the budget is used; any budget
    left is filled with the most recent messages (a hybrid retrieval + recency window). The
    selection is emitted in chronological order.
    """
    name = "retrieval"

    def __init__(self, k1: float = 1.5, b: float = 0.75, recency_fill: bool = True):
        self.k1, self.b, self.recency_fill = k1, b, recency_fill
        self.messages: list[Message] = []
        self._docs: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._lens: list[int] = []

    def add(self, message: Message) -> None:
        toks = tokenize(message.content)
        c = Counter(toks)
        self.messages.append(message)
        self._docs.append(c)
        self._lens.append(len(toks))
        self._df.update(c.keys())

    def scores(self, query: str) -> np.ndarray:
        n = len(self.messages)
        out = np.zeros(n)
        if n == 0:
            return out
        q = set(tokenize(query))
        avgdl = max(1.0, float(np.mean(self._lens)))
        for t in q:
            n_t = self._df.get(t, 0)
            if n_t == 0:
                continue
            idf = math.log((n - n_t + 0.5) / (n_t + 0.5) + 1.0)
            for i, c in enumerate(self._docs):
                tf = c.get(t, 0)
                if tf:
                    out[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self._lens[i] / avgdl))
        return out

    def select(self, budget_tokens: int, query: str = "") -> list[int]:
        s = self.scores(query) if query else np.zeros(len(self.messages))
        # best score first, newest first among ties
        order = sorted((i for i in range(len(self.messages)) if s[i] > 0), key=lambda i: (-s[i], -i))
        chosen: list[int] = []
        used = 0
        for i in order:
            t = self.messages[i].tokens
            if used + t <= budget_tokens:
                chosen.append(i)
                used += t
        if self.recency_fill:
            taken = set(chosen)
            for i in range(len(self.messages) - 1, -1, -1):
                if i in taken:
                    continue
                t = self.messages[i].tokens
                if used + t > budget_tokens:
                    break
                chosen.append(i)
                used += t
        return sorted(chosen)

    def render(self, budget_tokens: int, query: str = "") -> str:
        return render_messages([self.messages[i] for i in self.select(budget_tokens, query)])


ManagerFactory = Callable[[], ContextManager]


def default_managers(llm: LLMClient) -> dict[str, ManagerFactory]:
    return {"truncate": TruncateOldest, "summarize": lambda: SummarizeOldest(llm),
            "retrieval": RetrievalContext}


# ---------------------------------------------------------------------------
# noise and facts
# ---------------------------------------------------------------------------

_SERVICES = ["ingest", "billing-worker", "cache-node", "edge-proxy", "scheduler", "indexer", "mailer"]
_LEVELS = ["INFO", "DEBUG", "WARN", "INFO", "INFO"]
_FILES = ["main.py", "utils.py", "handlers.py", "schema.sql", "deploy.sh", "notes.txt", "Makefile"]


class NoiseGenerator:
    """Seeded distractor turns: near-duplicate chatter, irrelevant tool outputs, long logs.

    Noise vocabulary deliberately avoids the fact keys (``FactPlanter.KEYS``) so that a lexical
    retriever is tested on finding the facts, not on the noise accidentally mentioning them.
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self._n = 0

    def _near_duplicate(self) -> list[Message]:
        svc = self._SERVICE()
        n = self._n
        minute = int(self.rng.integers(0, 60))
        return [Message("user", f"can you re-check the rollout status for {svc} (ticket {n})?"),
                Message("assistant", f"Rollout status for {svc}: healthy, 0 restarts, checked at 10:{minute:02d} (ticket {n}).")]

    def _tool_output(self) -> list[Message]:
        files = list(self.rng.choice(_FILES, size=int(self.rng.integers(2, 5)), replace=False))
        sizes = {f: int(self.rng.integers(100, 90000)) for f in files}
        return [Message("tool", f'{{"tool": "list_files", "call": {self._n}, "result": {sizes}}}')]

    def _log(self) -> list[Message]:
        svc = self._SERVICE()
        lines = [f"[{self.rng.choice(_LEVELS)}] 2026-10-{int(self.rng.integers(1, 28)):02d}T10:{int(self.rng.integers(0, 60)):02d}:{int(self.rng.integers(0, 60)):02d} "
                 f"{svc} batch {int(self.rng.integers(1, 500))} processed in {int(self.rng.integers(5, 900))}ms"
                 for _ in range(int(self.rng.integers(8, 20)))]
        return [Message("tool", f"log tail (call {self._n}):\n" + "\n".join(lines))]

    def _SERVICE(self) -> str:
        return str(self.rng.choice(_SERVICES))

    def generate(self, target_tokens: int) -> list[Message]:
        out: list[Message] = []
        used = 0
        kinds = [self._near_duplicate, self._tool_output, self._log]
        while used < target_tokens:
            self._n += 1
            msgs = kinds[int(self.rng.integers(0, len(kinds)))]()
            out.extend(msgs)
            used += total_tokens(msgs)
        return out


@dataclass
class Fact:
    key: str
    value: str
    position: str               # early | middle | late
    frac: float                 # planting position as a fraction of the stream
    update_value: str | None = None
    update_frac: float | None = None

    @property
    def latest(self) -> str:
        return self.update_value if self.update_value is not None else self.value

    @property
    def statement(self) -> Message:
        return Message("user", f"Note for later: KEY: {self.key}={self.value}")

    @property
    def update_statement(self) -> Message | None:
        if self.update_value is None:
            return None
        return Message("user", f"Correction: KEY: {self.key} changed to {self.update_value}")

    @property
    def question(self) -> str:
        return f"What is the current value of {self.key}?"


POSITIONS = ("early", "middle", "late")
_POSITION_RANGE = {"early": (0.0, 0.2), "middle": (0.4, 0.6), "late": (0.8, 1.0)}


class FactPlanter:
    """Plant N ``KEY: k=v`` facts at early/middle/late positions; ``update_fraction`` of them
    get a later contradicting ``KEY: k changed to v2`` so the tester checks LATEST-value recall."""

    KEYS = ("user_id", "budget", "deadline", "region", "project", "tier", "currency", "owner",
            "sprint", "locale", "timezone", "priority", "cluster_id", "quota", "release", "vendor")

    def __init__(self, seed: int = 0, update_fraction: float = 0.5):
        self.rng = np.random.default_rng(seed + 1000)
        self.update_fraction = update_fraction

    def _value(self, key: str) -> str:
        return f"{key[:3]}_{int(self.rng.integers(1000, 9999))}"

    def plant(self, n_facts: int) -> list[Fact]:
        facts: list[Fact] = []
        for i in range(n_facts):
            key = self.KEYS[i % len(self.KEYS)] + ("" if i < len(self.KEYS) else f"_{i // len(self.KEYS)}")
            pos = POSITIONS[i % 3]
            lo, hi = _POSITION_RANGE[pos]
            frac = float(self.rng.uniform(lo, hi))
            f = Fact(key=key, value=self._value(key), position=pos, frac=frac)
            if self.rng.uniform() < self.update_fraction:
                f.update_value = self._value(key)
                f.update_frac = min(0.999, frac + float(self.rng.uniform(0.1, 0.3)))
            facts.append(f)
        return facts


def build_conversation(noise: list[Message], facts: list[Fact]) -> tuple[list[Message], set[str]]:
    """Interleave facts into the noise stream by position fraction. Returns (messages, noise_contents)."""
    n = max(1, len(noise))
    items: list[tuple[float, int, Message]] = [(i / n, i, m) for i, m in enumerate(noise)]
    seq = len(noise)
    for f in facts:
        items.append((f.frac, seq, f.statement))
        seq += 1
        if f.update_statement is not None:
            items.append((f.update_frac or 1.0, seq, f.update_statement))
            seq += 1
    items.sort(key=lambda t: (t[0], t[1]))
    return [m for _, _, m in items], {m.content for m in noise}


# ---------------------------------------------------------------------------
# probe answerers
# ---------------------------------------------------------------------------


class ProbeAnswerer(Protocol):
    def answer(self, context: str, question: str, key: str) -> str | None: ...


class LookupAnswerer:
    """Deterministic string lookup: the LAST statement about ``key`` in the context wins."""

    def answer(self, context: str, question: str, key: str) -> str | None:
        rx = re.compile(rf"KEY: {re.escape(key)}(?:=| changed to )(\S+)")
        found = rx.findall(context)
        return found[-1] if found else None


class LLMAnswerer:
    """Ask an ``LLMClient`` the probe question over the rendered context."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def answer(self, context: str, question: str, key: str) -> str | None:
        prompt = (f"Context:\n{context}\n\nQuestion: {question}\n"
                  "Reply with only the current value, or UNKNOWN if it is not in the context.")
        text = self.llm.complete(prompt, system="You answer strictly from the given context.").text.strip()
        return None if not text or text.upper().startswith("UNKNOWN") else text.split()[0].strip(".,")


def probe_handler(system: str, prompt: str) -> str:
    """MockLLM handler that answers probe prompts the same way ``LookupAnswerer`` would."""
    m = re.search(r"Question: What is the current value of (\w+)\?", prompt)
    if not m:
        return "UNKNOWN"
    ctx = prompt.split("\n\nQuestion:", 1)[0]
    ans = LookupAnswerer().answer(ctx, "", m.group(1))
    return ans or "UNKNOWN"


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    key: str
    position: str
    expected: str
    answer: str | None
    found: bool
    latest_correct: bool
    has_update: bool
    rendered_tokens: int
    noise_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ScenarioResult:
    manager: str
    noise_tokens: int
    budget: int
    n_facts: int
    total_tokens: int
    probes: list[ProbeResult] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return _rate([p.found for p in self.probes])

    @property
    def latest_correct(self) -> float:
        return _rate([p.latest_correct for p in self.probes])

    @property
    def latest_given_found(self) -> float:
        return _rate([p.latest_correct for p in self.probes if p.found])

    @property
    def update_correct(self) -> float:
        return _rate([p.latest_correct for p in self.probes if p.has_update])

    def by_position(self, metric: str = "found") -> dict[str, float]:
        return {pos: _rate([getattr(p, metric) for p in self.probes if p.position == pos]) for pos in POSITIONS}

    @property
    def compression_ratio(self) -> float:
        return float(np.mean([p.rendered_tokens for p in self.probes])) / max(1, self.total_tokens)

    @property
    def false_retention(self) -> float:
        return float(np.mean([p.noise_tokens / max(1, p.rendered_tokens) for p in self.probes]))

    @property
    def mean_rendered_tokens(self) -> float:
        return float(np.mean([p.rendered_tokens for p in self.probes]))

    def to_dict(self) -> dict[str, Any]:
        return {"manager": self.manager, "noise_tokens": self.noise_tokens, "budget": self.budget,
                "n_facts": self.n_facts, "total_tokens": self.total_tokens, "recall": self.recall,
                "latest_correct": self.latest_correct, "latest_given_found": self.latest_given_found,
                "update_correct": self.update_correct,
                "recall_by_position": self.by_position("found"),
                "latest_by_position": self.by_position("latest_correct"),
                "compression_ratio": self.compression_ratio, "false_retention": self.false_retention,
                "mean_rendered_tokens": self.mean_rendered_tokens,
                "probes": [p.to_dict() for p in self.probes]}


def _rate(xs: list[bool]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def run_scenario(factory: ManagerFactory, name: str, messages: list[Message], noise_contents: set[str],
                 facts: list[Fact], budget: int, noise_tokens: int,
                 answerer: ProbeAnswerer | None = None) -> ScenarioResult:
    answerer = answerer or LookupAnswerer()
    mgr = factory()
    for m in messages:
        mgr.add(m)
    total = approx_tokens(render_messages(messages))
    res = ScenarioResult(manager=name, noise_tokens=noise_tokens, budget=budget, n_facts=len(facts), total_tokens=total)
    for f in facts:
        ctx = mgr.render(budget, query=f.question)
        ans = answerer.answer(ctx, f.question, f.key)
        rendered = approx_tokens(ctx)
        noise_in = sum(approx_tokens(f"[x] {c}") for c in noise_contents if c in ctx)
        res.probes.append(ProbeResult(key=f.key, position=f.position, expected=f.latest, answer=ans,
                                      found=ans is not None, latest_correct=ans == f.latest,
                                      has_update=f.update_value is not None, rendered_tokens=rendered,
                                      noise_tokens=min(noise_in, rendered)))
    return res


def run_matrix(managers: dict[str, ManagerFactory], noise_levels: list[int], budget: int, n_facts: int,
               seed: int = 0, answerer: ProbeAnswerer | None = None) -> list[ScenarioResult]:
    """Every manager sees the identical conversation for a given (noise level, seed)."""
    results: list[ScenarioResult] = []
    for level in noise_levels:
        noise = NoiseGenerator(seed=seed * 7919 + level).generate(level)
        facts = FactPlanter(seed=seed).plant(n_facts)
        messages, noise_contents = build_conversation(noise, facts)
        for name, factory in managers.items():
            results.append(run_scenario(factory, name, messages, noise_contents, facts, budget, level, answerer))
    return results
