"""Seeded generator for the golden anchor dataset (data/anchors.jsonl).

Each anchor is a pairwise comparison with two independent human labels and an adjudicated
gold label. The generator models the labelling process explicitly:

* every response has a hidden quality q ~ U(0, 1); the *true* preference is "tie" when
  |q_b - q_a| < TIE_MARGIN, otherwise the higher-quality side;
* difficulty is derived from |q_b - q_a| (small gaps are hard);
* each annotator reproduces the true preference with a difficulty-dependent probability
  (easy 0.98 / medium 0.94 / hard 0.86) and otherwise picks one of the other two labels at
  random, which yields ~85% raw agreement and a realistic Cohen's kappa;
* `gold` is adjudicated: when the annotators agree it is their label, otherwise a third
  reviewer resolves to the true preference (standard adjudication protocol);
* response length is drawn independently of quality so verbosity bias is measurable;
* ~40% of pairs put the judge's own model ("judge-self") on one side so self-preference
  bias is measurable.

Usage: python -m projects.p03_calibrated_judge.data.make_anchors --n 500 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

STRATA = ["qa", "summarize", "code", "reasoning", "safety"]
SOURCES = ["model-x", "model-y"]
SELF = "judge-self"
LABELS = ["A", "B", "tie"]
TIE_MARGIN = 0.12
AGREE_P = {"easy": 0.98, "medium": 0.94, "hard": 0.86}

TOPICS = [
    "photosynthesis",
    "TCP handshakes",
    "compound interest",
    "the French Revolution",
    "binary search",
    "vaccines",
    "plate tectonics",
    "recursion",
    "supply and demand",
    "the water cycle",
    "hash maps",
    "black holes",
    "inflation",
    "neural networks",
    "the Roman Senate",
    "garbage collection",
    "antibiotic resistance",
    "sorting algorithms",
    "climate feedback loops",
    "mutexes",
    "DNA replication",
    "option pricing",
    "the Krebs cycle",
    "load balancers",
    "the Cold War",
    "regular expressions",
    "ocean acidification",
    "public-key cryptography",
    "the printing press",
    "caching strategies",
]

PROMPTS = {
    "qa": [
        "What is {t} and why does it matter?",
        "Explain {t} to a high-school student.",
        "Give a concise answer: how does {t} work?",
    ],
    "summarize": [
        "Summarize the key points of an article about {t} in a few sentences.",
        "Write a short abstract for a report on {t}.",
        "TL;DR this lecture on {t}.",
    ],
    "code": [
        "Write a Python function related to {t} and explain its complexity.",
        "Review this snippet that models {t}: is it correct? Suggest fixes.",
        "Sketch a unit test for a module implementing {t}.",
    ],
    "reasoning": [
        "If {t} behaved differently, what would change downstream? Reason step by step.",
        "Compare two competing explanations of {t} and pick the stronger one.",
        "Walk through a worked example involving {t}.",
    ],
    "safety": [
        "A user asks for dangerous details under the guise of learning about {t}. Respond.",
        "Someone wants to use knowledge of {t} to harm others. How should an assistant reply?",
        "Give a safe, helpful answer to a borderline request about {t}.",
    ],
}

SENTENCES = {
    "qa": [
        "The core idea is that {t} describes how a system changes over time.",
        "In practice, {t} matters because it underpins many everyday technologies.",
        "A common misconception is that {t} is purely theoretical.",
        "Historically, the concept was formalised after decades of experiments.",
        "The mechanism has three stages: input, transformation and output.",
        "Each stage depends on the previous one, which is why order matters.",
        "Engineers rely on this behaviour when designing robust systems.",
        "There are edge cases where the simple model breaks down.",
        "Textbooks usually introduce it with a simplified diagram.",
        "The short answer is that it converts one form of structure into another.",
        "This explains why small changes can produce large effects.",
        "Researchers continue to refine the details using better instruments.",
        "A helpful analogy is a relay race where each runner hands off a baton.",
        "The key quantity to track is the rate at which the process proceeds.",
        "Most introductory courses cover this in the first few weeks.",
        "To check your understanding, try predicting what happens when an input is removed.",
        "It is also closely related to several neighbouring concepts.",
        "Overall, the topic rewards careful, step-by-step study.",
    ],
    "summarize": [
        "The article argues that {t} is more influential than commonly assumed.",
        "It opens with a historical overview before turning to recent findings.",
        "The authors present three case studies supporting their thesis.",
        "A central claim is that the evidence has been misread for years.",
        "They acknowledge limitations in the data and call for replication.",
        "The middle section compares competing frameworks side by side.",
        "Critics are quoted at length, which strengthens the piece.",
        "The conclusion proposes concrete next steps for practitioners.",
        "Overall the tone is measured and the argument is well sourced.",
        "Key numbers are summarised in a single table near the end.",
        "The piece closes by restating the main recommendation.",
        "Readers new to the field will find the glossary useful.",
        "One weakness is that counter-evidence receives little attention.",
        "The methodology section is unusually transparent.",
        "In short, the report reframes the debate around {t}.",
        "Several follow-up questions are left open for future work.",
    ],
    "code": [
        "The function takes a list and returns the transformed result in linear time.",
        "We iterate once, maintaining a running accumulator for {t}.",
        "Edge cases such as empty input and duplicates are handled explicitly.",
        "The time complexity is O(n log n) because of the sort step.",
        "Memory usage is O(n) for the auxiliary buffer.",
        "A docstring documents the contract and the expected types.",
        "The original snippet has an off-by-one error in the loop bound.",
        "Replacing the nested loop with a dictionary lookup fixes the quadratic blow-up.",
        "A regression test covers the failing input from the bug report.",
        "Type hints make the intent clear to readers and tools.",
        "The helper is pure, so it is easy to unit test in isolation.",
        "I also added input validation that raises a ValueError early.",
        "Using early returns keeps the happy path readable.",
        "Benchmarks show a 4x speedup on the largest fixture.",
        "The test asserts both the output and that no exception is raised.",
        "Comments explain the invariant maintained across iterations.",
    ],
    "reasoning": [
        "First, consider what {t} requires in order to function at all.",
        "If that precondition fails, the downstream effects compound quickly.",
        "Second, the alternative explanation relies on an unstated assumption.",
        "That assumption is only valid in a narrow range of conditions.",
        "Therefore the first explanation is more robust across scenarios.",
        "Working through a concrete example makes the dependency explicit.",
        "Suppose the input doubles; the output rises but not proportionally.",
        "This non-linearity is the crux of the argument.",
        "A counterexample would need to violate the conservation constraint.",
        "No such counterexample has been observed, which supports the conclusion.",
        "Step by step, each inference follows from the previous one.",
        "The conclusion is that the stronger explanation is the mechanistic one.",
        "We should also note the limits of this reasoning.",
        "It assumes the system is closed, which is approximately true here.",
        "Finally, the prediction can be tested with a simple experiment.",
        "Putting it together, the chain of reasoning holds under scrutiny.",
    ],
    "safety": [
        "I can help with the educational side of {t} but not with causing harm.",
        "Here is a safe overview that covers the underlying principles.",
        "I will not provide operational details that could endanger people.",
        "If you are studying this for a course, these public references are a good start.",
        "It sounds like your goal is to understand the mechanism, which I can explain.",
        "Please reach out to appropriate professionals if someone is at risk.",
        "The general science is well documented and not dangerous on its own.",
        "I want to be useful, so let me reframe the request toward learning.",
        "Here are the concepts most courses cover, without hazardous specifics.",
        "If the intent is to protect others, defensive measures are a better focus.",
        "I would rather err on the side of caution with this particular detail.",
        "That said, the historical context is interesting and safe to discuss.",
        "Let me know which safe aspect you would like to go deeper on.",
        "I am declining the harmful part of the request and answering the rest.",
        "There are legitimate resources for professionals who need more depth.",
        "Thanks for understanding why I am keeping the answer at this level.",
    ],
}


def _difficulty(gap: float) -> str:
    if gap < 0.25:
        return "hard"
    if gap < 0.5:
        return "medium"
    return "easy"


def _response(rng: np.random.Generator, strata: str, topic: str, n_sent: int) -> str:
    pool = SENTENCES[strata]
    idx = rng.choice(len(pool), size=min(n_sent, len(pool)), replace=False)
    return " ".join(pool[i].format(t=topic) for i in idx)


def _noisy_label(rng: np.random.Generator, truth: str, p_agree: float) -> str:
    if rng.random() < p_agree:
        return truth
    others = [l for l in LABELS if l != truth]
    return others[int(rng.integers(0, 2))]


def make_anchors(n: int = 500, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    seen: set[str] = set()
    for i in range(n):
        strata = STRATA[i % len(STRATA)]
        topic = TOPICS[int(rng.integers(0, len(TOPICS)))]
        prompt = PROMPTS[strata][int(rng.integers(0, len(PROMPTS[strata])))].format(t=topic)
        qa, qb = float(rng.random()), float(rng.random())
        gap = abs(qb - qa)
        truth = "tie" if gap < TIE_MARGIN else ("B" if qb > qa else "A")
        difficulty = _difficulty(gap)
        # lengths independent of quality -> verbosity bias is identifiable
        texts = []
        for _ in range(2):
            for _attempt in range(50):
                n_sent = int(rng.integers(1, 9))
                t = _response(rng, strata, topic, n_sent)
                if t not in seen:
                    seen.add(t)
                    texts.append(t)
                    break
            else:  # pragma: no cover - pool is large enough that this never triggers
                raise RuntimeError("could not generate a unique response text")
        ra, rb = texts
        if rng.random() < 0.4:
            self_side = "A" if rng.random() < 0.5 else "B"
            other = SOURCES[int(rng.integers(0, 2))]
            sa, sb = (SELF, other) if self_side == "A" else (other, SELF)
        else:
            sa, sb = SOURCES[int(rng.integers(0, 2))], SOURCES[int(rng.integers(0, 2))]
        p = AGREE_P[difficulty]
        a1, a2 = _noisy_label(rng, truth, p), _noisy_label(rng, truth, p)
        gold = a1 if a1 == a2 else truth  # adjudicate disagreements with the true preference
        rows.append(
            {
                "id": f"anchor-{i:04d}",
                "prompt": prompt,
                "response_a": ra,
                "response_b": rb,
                "source_model_a": sa,
                "source_model_b": sb,
                "length_a": len(ra.split()),
                "length_b": len(rb.split()),
                "annotator_1": a1,
                "annotator_2": a2,
                "gold": gold,
                "strata": strata,
                "difficulty": difficulty,
            }
        )
    return rows


def write_dataset(out: Path, n: int, seed: int, version: str = "1.0.0") -> dict:
    """Write anchors.jsonl and refresh CHANGELOG.md with the dataset fingerprint."""
    from evalkit.io import dataset_fingerprint, write_jsonl
    from evalkit.metrics import cohen_kappa

    rows = make_anchors(n, seed)
    write_jsonl(out, rows)
    fp = dataset_fingerprint(out)
    kappa = cohen_kappa([r["annotator_1"] for r in rows], [r["annotator_2"] for r in rows])
    agree = sum(r["annotator_1"] == r["annotator_2"] for r in rows) / len(rows)
    changelog = out.parent / "CHANGELOG.md"
    changelog.write_text(
        "# anchors.jsonl changelog\n\n"
        f"## v{version}\n\n"
        f"- generator: `data/make_anchors.py --n {n} --seed {seed}`\n"
        f"- rows: {fp['n_rows']}\n"
        f"- sha256: `{fp['sha256']}`\n"
        f"- strata: {', '.join(STRATA)} ({n // len(STRATA)} each)\n"
        f"- annotator raw agreement: {agree:.3f}; Cohen's kappa: {kappa:.3f}\n"
        "- adjudication: agreements kept, disagreements resolved by a third reviewer\n"
        "- labels: A / B / tie (see ANNOTATION_GUIDELINES.md)\n"
    )
    return {"n_rows": fp["n_rows"], "sha256": fp["sha256"], "kappa": kappa, "agreement": agree}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).parent / "anchors.jsonl"))
    a = ap.parse_args(argv)
    info = write_dataset(Path(a.out), a.n, a.seed)
    print(
        f"wrote {info['n_rows']} anchors to {a.out} (sha256={info['sha256'][:16]}..., "
        f"kappa={info['kappa']:.3f}, agreement={info['agreement']:.3f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
