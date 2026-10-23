"""Small lexical helpers shared by the fallback adapters. Formulas in docstrings."""
from __future__ import annotations

import re
from collections import Counter

STOPWORDS = frozenset(
    "a an and are as at be by does do did for from has have how in is it its of on or that the this to was "
    "what when where which who whom why will with can could should would there their they them than then".split()
)
_RX = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


def tokenize(text: str) -> list[str]:
    out = []
    for t in _RX.findall(str(text).lower()):
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def normalise(text: str) -> str:
    return " ".join(content_tokens(text))


def exact_match(pred: str, gold: str) -> float:
    return float(normalise(pred) == normalise(gold))


def token_f1(pred: str, gold: str) -> float:
    """SQuAD-style F1 over bags of content tokens: 2PR/(P+R)."""
    p, g = Counter(content_tokens(pred)), Counter(content_tokens(gold))
    if not p or not g:
        return float(p == g)
    common = sum((p & g).values())
    if common == 0:
        return 0.0
    prec, rec = common / sum(p.values()), common / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def coverage(target: str, text: str) -> float:
    """Fraction of `target`'s content tokens that appear in `text` (recall of target by text)."""
    t = content_tokens(target)
    if not t:
        return 1.0
    s = set(tokenize(text))
    return sum(x in s for x in t) / len(t)


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text)) if s.strip()]


def faithfulness(output: str, contexts: list[str], threshold: float = 0.5) -> float:
    """RAGAS-style faithfulness: fraction of answer claims (sentences) supported by the context,
    where support = coverage(sentence, any context chunk) >= threshold."""
    sents = split_sentences(output)
    if not sents:
        return 0.0
    supported = sum(any(coverage(s, c) >= threshold for c in contexts) for s in sents)
    return supported / len(sents)


def context_precision(contexts: list[str], expected: str, threshold: float = 0.5) -> float:
    """RAGAS-style context precision@K: sum_k(precision@k * rel_k) / #relevant, rel_k = chunk k
    covers >= threshold of the expected answer's tokens."""
    if not contexts:
        return 0.0
    rel = [coverage(expected, c) >= threshold for c in contexts]
    if not any(rel):
        return 0.0
    hits, acc = 0, 0.0
    for k, r in enumerate(rel, 1):
        if r:
            hits += 1
            acc += hits / k
    return acc / sum(rel)
