"""LLM client abstraction.

Every project runs fully offline against MockLLM so tests are deterministic and free.
Set EVALKIT_LLM=anthropic and ANTHROPIC_API_KEY to run the same code against a real model.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)

    def json(self) -> Any:
        """Parse the first JSON object/array in the text."""
        m = re.search(r"(\{.*\}|\[.*\])", self.text, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON in response: {self.text[:200]!r}")
        return json.loads(m.group(1))


class LLMClient(Protocol):
    model: str

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0,
                 max_tokens: int = 1024, seed: int | None = None) -> LLMResponse: ...


# Approximate public list prices per million tokens, used for cost accounting in reports.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (15.0, 75.0),
    "claude-opus-5-5": (15.0, 75.0),
    "claude-sonnet-5-5": (3.0, 15.0),
    "claude-haiku-5-5": (0.80, 4.0),
    "mock-system-one": (0.25, 1.25),
    "mock-system-two": (5.0, 25.0),
    "mock": (0.0, 0.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = PRICES_PER_MTOK.get(model, (1.0, 5.0))
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


Handler = Callable[[str, str], str]  # (system, prompt) -> text


class MockLLM:
    """Deterministic fake model.

    Behaviour is scripted with `handlers`: an ordered list of (regex, handler-or-string).
    The first regex that matches the prompt wins. A default handler echoes a stable hash so
    outputs are reproducible across runs. Optional `latency_ms` and `fail_rate` let tests
    simulate slow or flaky backends; `seed` makes the failures reproducible.
    """

    def __init__(self, model: str = "mock", handlers: list[tuple[str, Handler | str]] | None = None,
                 latency_ms: float = 0.0, fail_rate: float = 0.0, seed: int = 0,
                 verbosity: float = 1.0):
        self.model = model
        self.handlers = [(re.compile(p, re.DOTALL | re.IGNORECASE), h) for p, h in (handlers or [])]
        self.latency_ms = latency_ms
        self.fail_rate = fail_rate
        self.seed = seed
        self.verbosity = verbosity
        self.calls: list[dict[str, Any]] = []

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0,
                 max_tokens: int = 1024, seed: int | None = None) -> LLMResponse:
        t0 = time.perf_counter()
        digest = hashlib.sha256(f"{self.seed}:{system}:{prompt}".encode()).hexdigest()
        if self.fail_rate and int(digest[:8], 16) / 0xFFFFFFFF < self.fail_rate:
            raise RuntimeError(f"{self.model}: simulated backend failure")
        text = None
        for rx, h in self.handlers:
            if rx.search(prompt) or rx.search(system):
                text = h if isinstance(h, str) else h(system, prompt)
                break
        if text is None:
            text = f"[mock:{digest[:8]}] " + prompt[-80:]
        if self.verbosity != 1.0:
            text = text + (" Furthermore, to elaborate in more detail," * int(max(0, self.verbosity - 1) * 3))
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        it, ot = approx_tokens(system + prompt), approx_tokens(text)
        resp = LLMResponse(text=text, model=self.model, input_tokens=it, output_tokens=ot,
                           latency_ms=(time.perf_counter() - t0) * 1000,
                           cost_usd=estimate_cost(self.model, it, ot))
        self.calls.append({"system": system, "prompt": prompt, "response": text})
        return resp


class AnthropicLLM:
    """Thin wrapper over the Anthropic Messages API. Imported lazily."""

    def __init__(self, model: str = "claude-sonnet-5-5"):
        import anthropic  # type: ignore
        self._client = anthropic.Anthropic()
        self.model = model

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0,
                 max_tokens: int = 1024, seed: int | None = None) -> LLMResponse:
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature,
                                      "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system
        msg = self._client.messages.create(**kwargs)
        text = "".join(getattr(b, "text", "") for b in msg.content)
        it, ot = msg.usage.input_tokens, msg.usage.output_tokens
        return LLMResponse(text=text, model=self.model, input_tokens=it, output_tokens=ot,
                           latency_ms=(time.perf_counter() - t0) * 1000,
                           cost_usd=estimate_cost(self.model, it, ot),
                           raw={"id": msg.id, "stop_reason": msg.stop_reason})


def get_client(model: str | None = None, **mock_kwargs: Any) -> LLMClient:
    """EVALKIT_LLM=anthropic -> real model; anything else -> MockLLM."""
    backend = os.environ.get("EVALKIT_LLM", "mock").lower()
    if backend == "anthropic":
        return AnthropicLLM(model or os.environ.get("EVALKIT_MODEL", "claude-sonnet-5-5"))
    return MockLLM(model=model or "mock", **mock_kwargs)
