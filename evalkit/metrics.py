"""Classification, agreement, and retrieval metrics. Pure numpy, no sklearn."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence

import numpy as np

# ---------- classification ----------

def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) for binary labels in {0,1}."""
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred, strict=True):
        if t and p:
            tp += 1
        elif not t and p:
            fp += 1
        elif t and not p:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def precision_recall_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float]:
    tp, fp, fn, tn = confusion(y_true, y_pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def accuracy(y_true: Sequence[Hashable], y_pred: Sequence[Hashable]) -> float:
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred, strict=True)) / len(y_true)


# ---------- agreement ----------

def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's kappa between two annotators over the same items."""
    if len(a) != len(b):
        raise ValueError("annotator label lists must have equal length")
    n = len(a)
    if n == 0:
        return 0.0
    labels = set(a) | set(b)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if math.isclose(pe, 1.0):
        return 1.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(ratings: Sequence[Sequence[Hashable]]) -> float:
    """Fleiss' kappa. ratings[i] = labels given to item i by each rater (same count per item)."""
    if not ratings:
        return 0.0
    n_raters = len(ratings[0])
    labels = sorted({l for row in ratings for l in row}, key=str)
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(ratings), len(labels)))
    for i, row in enumerate(ratings):
        if len(row) != n_raters:
            raise ValueError("every item needs the same number of raters")
        for l in row:
            M[i, idx[l]] += 1
    p_j = M.sum(axis=0) / (len(ratings) * n_raters)
    P_i = ((M ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    if math.isclose(P_e, 1.0):
        return 1.0
    return float((P_bar - P_e) / (1 - P_e))


def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977) bands."""
    if k < 0:
        return "poor"
    if k < 0.2:
        return "slight"
    if k < 0.4:
        return "fair"
    if k < 0.6:
        return "moderate"
    if k < 0.8:
        return "substantial"
    return "almost perfect"


# ---------- confidence intervals ----------

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values: Sequence[float], stat=np.mean, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI. Returns (point, lo, hi)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = stat(arr[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(stat(arr)), float(lo), float(hi))


# ---------- retrieval ----------

def hit_rate(retrieved: Sequence[Sequence[Hashable]], relevant: Sequence[set], k: int | None = None) -> float:
    hits = 0
    for r, rel in zip(retrieved, relevant, strict=True):
        top = r[:k] if k else r
        hits += any(d in rel for d in top)
    return hits / len(retrieved) if retrieved else 0.0


def recall_at_k(retrieved: Sequence[Sequence[Hashable]], relevant: Sequence[set], k: int) -> float:
    vals = []
    for r, rel in zip(retrieved, relevant, strict=True):
        if not rel:
            continue
        vals.append(len(set(r[:k]) & rel) / len(rel))
    return float(np.mean(vals)) if vals else 0.0


def mrr(retrieved: Sequence[Sequence[Hashable]], relevant: Sequence[set]) -> float:
    rr = []
    for r, rel in zip(retrieved, relevant, strict=True):
        score = 0.0
        for rank, d in enumerate(r, 1):
            if d in rel:
                score = 1.0 / rank
                break
        rr.append(score)
    return float(np.mean(rr)) if rr else 0.0


def ndcg_at_k(retrieved: Sequence[Sequence[Hashable]], relevant: Sequence[dict], k: int) -> float:
    """relevant[i] maps doc_id -> graded relevance (int >= 0)."""
    vals = []
    for r, rel in zip(retrieved, relevant, strict=True):
        dcg = sum(rel.get(d, 0) / math.log2(i + 2) for i, d in enumerate(r[:k]))
        ideal = sorted(rel.values(), reverse=True)[:k]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        vals.append(dcg / idcg if idcg else 0.0)
    return float(np.mean(vals)) if vals else 0.0
