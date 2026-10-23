"""Seeded generator for the demo corpus with planted leaks.

* train.jsonl  ~5000 rows `{id, text, source}` of template-generated prose (each row is 4-6
  sentences of 8-12 tokens built from random vocabulary fills, so rows are unique).
* eval.jsonl   200 rows `{id, text, strata, planted}` where `planted` is the ground truth used
  by the tests and ignored by the scanner:
    - 10 `exact`      verbatim copies of a train row (some re-cased / re-punctuated, which the
                      scanner's normalisation must see through),
    - 10 `paraphrase` light edits of a train row: sentence order reversed + one synonym swap
                      in the middle of one sentence (no 13-gram survives, char n-gram
                      cosine stays high),
    - 10 `partial`    a clean row with a 15-token span copied from a train row,
    - 170 `null`      clean.
* results.jsonl  `{id, success}` per eval row: leaked rows succeed 95% of the time, clean rows
  70%, so contamination inflates the metric.

Usage: python -m projects.p14_contamination_checker.data.make_corpus --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evalkit.io import write_jsonl

ADJ = [
    "quick",
    "silent",
    "bright",
    "ancient",
    "narrow",
    "gentle",
    "rapid",
    "hollow",
    "clever",
    "humble",
    "frozen",
    "golden",
    "bitter",
    "restless",
    "sturdy",
    "fragile",
    "distant",
    "patient",
    "crimson",
    "vivid",
    "wooden",
    "sleepy",
    "eager",
    "lonely",
    "modest",
    "rugged",
    "subtle",
    "nimble",
    "faded",
    "polished",
]
NOUN = [
    "engineer",
    "harbour",
    "ledger",
    "compiler",
    "orchard",
    "lantern",
    "satellite",
    "archive",
    "glacier",
    "violin",
    "courier",
    "theorem",
    "reactor",
    "monastery",
    "canyon",
    "sculptor",
    "turbine",
    "manuscript",
    "pipeline",
    "beacon",
    "cartographer",
    "furnace",
    "meadow",
    "telescope",
    "locksmith",
    "estuary",
    "almanac",
    "physician",
    "quarry",
    "navigator",
    "parliament",
    "kiln",
    "lighthouse",
    "tribunal",
    "workshop",
    "comet",
    "treaty",
    "harvest",
    "fortress",
    "pendulum",
]
VERB = [
    "inspects",
    "repairs",
    "measures",
    "describes",
    "ignores",
    "protects",
    "balances",
    "records",
    "unlocks",
    "replaces",
    "questions",
    "observes",
    "assembles",
    "escorts",
    "translates",
    "predicts",
    "rebuilds",
    "surveys",
    "finances",
    "stabilises",
    "catalogues",
    "reroutes",
    "summarises",
    "audits",
    "calibrates",
]
PREP = ["beside", "beyond", "under", "across", "within", "behind", "toward", "around"]
ADV = [
    "carefully",
    "rarely",
    "quietly",
    "boldly",
    "promptly",
    "steadily",
    "twice",
    "nightly",
    "slowly",
    "openly",
    "briskly",
    "loosely",
    "neatly",
    "warmly",
    "softly",
    "firmly",
]
SYN = {
    "quick": "fast",
    "silent": "quiet",
    "bright": "luminous",
    "ancient": "old",
    "narrow": "slim",
    "gentle": "mild",
    "rapid": "swift",
    "hollow": "empty",
    "clever": "smart",
    "humble": "modest",
    "frozen": "icy",
    "golden": "gilded",
    "bitter": "sour",
    "restless": "uneasy",
    "sturdy": "solid",
    "fragile": "brittle",
    "distant": "remote",
    "patient": "calm",
    "crimson": "scarlet",
    "vivid": "bright",
    "wooden": "timber",
    "sleepy": "drowsy",
    "eager": "keen",
    "lonely": "solitary",
    "modest": "humble",
    "rugged": "rough",
    "subtle": "faint",
    "nimble": "agile",
    "faded": "pale",
    "polished": "glossy",
    "inspects": "examines",
    "repairs": "fixes",
    "measures": "gauges",
    "describes": "depicts",
    "ignores": "overlooks",
    "protects": "guards",
    "balances": "steadies",
    "records": "logs",
    "unlocks": "opens",
    "replaces": "swaps",
    "questions": "doubts",
    "observes": "watches",
}
TEMPLATES = [
    "the {a} {n} {v} the {a} {n} {p} the {n}",
    "{d} the {n} {v} a {a} {n} {p} the {a} {n}",
    "a {a} {n} {d} {v} the {n} {p} a {a} {n}",
    "the {n} {p} the {a} {n} {d} {v} the {n}",
    "every {a} {n} {v} the {n} {p} the {a} {n} {d}",
    "the {a} {n} and the {a} {n} {v} a {n} {d}",
]
STRATA = ["qa", "code", "math", "summarize"]


def _pick(rng, pool):
    return pool[int(rng.integers(0, len(pool)))]


def sentence(rng: np.random.Generator) -> str:
    t = _pick(rng, TEMPLATES)
    out = []
    for tok in t.split():
        if tok == "{a}":
            out.append(_pick(rng, ADJ))
        elif tok == "{n}":
            out.append(_pick(rng, NOUN))
        elif tok == "{v}":
            out.append(_pick(rng, VERB))
        elif tok == "{p}":
            out.append(_pick(rng, PREP))
        elif tok == "{d}":
            out.append(_pick(rng, ADV))
        else:
            out.append(tok)
    return " ".join(out)


def document(rng: np.random.Generator, n_sent: int | None = None) -> list[str]:
    k = n_sent or int(rng.integers(4, 7))
    return [sentence(rng) for _ in range(k)]


def join(sents: list[str]) -> str:
    return " ".join(s[0].upper() + s[1:] + "." for s in sents)


def paraphrase(rng: np.random.Generator, sents: list[str]) -> list[str]:
    """Reverse sentence order (breaks every original adjacency) and swap one mid-sentence
    word for a synonym in one sentence. Sentences are <= 12 tokens, so no 13-gram survives."""
    out = list(reversed(sents))
    for i in sorted(rng.choice(len(out), size=1, replace=False)):
        toks = out[i].split()
        mid = len(toks) // 2
        cands = [j for j in range(max(1, mid - 2), min(len(toks) - 1, mid + 3)) if toks[j] in SYN]
        j = cands[0] if cands else mid
        toks[j] = SYN.get(toks[j], "notable")
        out[i] = " ".join(toks)
    return out


def recase(rng: np.random.Generator, text: str) -> str:
    """Case / punctuation noise that normalisation should remove."""
    mode = int(rng.integers(0, 3))
    if mode == 0:
        return text
    if mode == 1:
        return text.upper().replace(".", "!")
    return text.lower().replace(". ", " ;  ").replace(".", "")


def make_corpus(
    out_dir: Path,
    seed: int = 0,
    n_train: int = 5000,
    n_eval: int = 200,
    n_exact: int = 10,
    n_para: int = 10,
    n_partial: int = 10,
) -> dict:
    rng = np.random.default_rng(seed)
    train_sents = [document(rng) for _ in range(n_train)]
    train = [
        {
            "id": f"train-{i:05d}",
            "text": join(s),
            "source": _pick(rng, ["web", "forum", "docs", "books"]),
        }
        for i, s in enumerate(train_sents)
    ]
    leak_src = rng.choice(n_train, size=n_exact + n_para + n_partial, replace=False)
    eval_rows = []
    k = 0
    for i in range(n_eval):
        strata = STRATA[i % len(STRATA)]
        if i < n_exact:
            text, planted = recase(rng, train[leak_src[k]]["text"]), "exact"
            k += 1
        elif i < n_exact + n_para:
            text, planted = join(paraphrase(rng, train_sents[leak_src[k]])), "paraphrase"
            k += 1
        elif i < n_exact + n_para + n_partial:
            src_toks = train[leak_src[k]]["text"].lower().replace(".", "").split()
            start = int(rng.integers(0, max(1, len(src_toks) - 15)))
            span = " ".join(src_toks[start : start + 15])
            own = document(rng, 4)
            own.insert(int(rng.integers(1, 4)), span)
            text, planted = join(own), "partial"
            k += 1
        else:
            text, planted = join(document(rng)), None
        eval_rows.append(
            {"id": f"eval-{i:04d}", "text": text, "strata": strata, "planted": planted}
        )
    results = [
        {"id": r["id"], "success": int(rng.random() < (0.95 if r["planted"] else 0.70))}
        for r in eval_rows
    ]
    return {
        "train.jsonl": write_jsonl(out_dir / "train.jsonl", train),
        "eval.jsonl": write_jsonl(out_dir / "eval.jsonl", eval_rows),
        "results.jsonl": write_jsonl(out_dir / "results.jsonl", results),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=5000)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent))
    a = ap.parse_args(argv)
    for name, n in make_corpus(Path(a.out_dir), a.seed, a.n_train, a.n_eval).items():
        print(f"wrote {n:5d} rows -> {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
