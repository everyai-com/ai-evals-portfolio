"""Systems under test: MockLLM-backed functions selectable by name in harness.yaml.

`faq_bot` is a scripted assistant with deliberate, deterministic flaws (number words instead of
digits, dropped first names, an unsupported extra claim, one refusal) so the frameworks have
something to disagree about. The script matches on the *input*, never on the expected answer."""
from __future__ import annotations

from typing import Any, Callable

from evalkit.llm import LLMClient, LLMResponse, MockLLM, get_client

FAQ_SCRIPT: list[tuple[str, str]] = [
    (r"capital of france", "Paris is the capital of France."),
    (r"legs does a spider", "A spider has eight legs."),
    (r"colou?r is the sky", "The sky is blue on a clear day."),
    (r"wrote hamlet", "Hamlet was written by Shakespeare."),
    (r"boiling point of water", "Water boils at 100 degrees Celsius at sea level. Adding salt lowers the boiling point."),
    (r"red planet", "Mars."),
    (r"largest ocean", "I don't know."),
    (r"plants absorb", "Plants absorb carbon dioxide and release oxygen."),
    (r"nimbus 3 (?:router )?warranty", "The Nimbus 3 comes with a 2 year warranty."),
    (r"nimbus mesh network contain", "A Nimbus mesh network supports at most twelve nodes."),
    (r"nimbus cloud plus cost", "Nimbus Cloud Plus costs 4 dollars per month."),
    (r"nimbus cloud data stored", "European data is stored in Frankfurt, and US data in Oregon."),
    (r"chief executive of larkspur", "The CEO of Larkspur Labs is Maren Oduya."),
    (r"nimbus 3 (?:router )?(?:have|has) .*antenna", "It has six internal antennas, one more than the Nimbus 2."),
]

SutFn = Callable[[dict[str, Any]], tuple[str, LLMResponse]]


def faq_bot(client: LLMClient | None = None) -> SutFn:
    client = client or get_client(handlers=[(p, a) for p, a in FAQ_SCRIPT] + [(r".", "I don't know.")])

    def run(example: dict[str, Any]) -> tuple[str, LLMResponse]:
        ctx = example.get("context") or []
        prompt = "Context:\n" + "\n".join(ctx) + f"\n\nQuestion: {example['input']}\nAnswer:" if ctx else example["input"]
        resp = client.complete(prompt, system="Answer briefly from the context.")
        return resp.text.strip(), resp

    return run


def echo_expected(client: LLMClient | None = None) -> SutFn:
    """A cheating SUT (returns the expected answer) - useful to check every adapter reports 100%."""
    def run(example: dict[str, Any]) -> tuple[str, LLMResponse]:
        text = str(example["expected"])
        return text, LLMResponse(text=text, model="mock", input_tokens=1, output_tokens=1, latency_ms=0.0)
    return run


SUTS: dict[str, Callable[..., SutFn]] = {"faq_bot": faq_bot, "echo_expected": echo_expected}


def get_sut(name: str) -> SutFn:
    if name not in SUTS:
        raise KeyError(f"unknown SUT {name!r}; choose from {sorted(SUTS)}")
    return SUTS[name]()
