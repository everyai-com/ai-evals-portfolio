"""Calibrated LLM-as-a-judge: judges, bias measurements and the calibration report.

Everything here is pure: judges take an `LLMClient`, bias functions take anchors + verdicts
and return dicts with a statistic, a 95% CI and an interpretation. I/O lives in cli.py.

Statistics (formulas)
---------------------
position bias      net_first = P(judge picks the response shown first) - P(picks second),
                   pooled over both presentation orders; 0 under no bias. CI: percentile
                   bootstrap over per-observation {+1, -1, 0} (evalkit.metrics.bootstrap_ci).
                   consistency = P(verdict(A,B) == flip(verdict(B,A))).
verbosity bias     lift = P(judge picks longer) - P(gold picks longer) over unequal-length pairs
                   (null 0, position-robust; drives the verdict), plus the classic conditional
                   rate: among pairs where gold is "tie" or prefers the SHORTER response, how
                   often the judge picks the LONGER one. CI: bootstrap over anchors. Plus the
                   longer-response win-rate bucketed by length ratio, judge vs gold.
self-preference    among pairs with exactly one "judge-self" side,
                   bias = P(judge picks self) - P(gold picks self). CI: bootstrap of the
                   per-pair difference.
agreement          accuracy vs gold; Cohen's kappa judge-vs-gold, judge-vs-annotator_1 and
                   annotator_1-vs-annotator_2 (the human ceiling); per-strata breakdown.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evalkit.io import write_jsonl
from evalkit.llm import LLMClient, MockLLM
from evalkit.metrics import accuracy, bootstrap_ci, cohen_kappa, interpret_kappa, wilson_ci
from evalkit.report import md_table, pct

LABELS = ("A", "B", "tie")
FLIP = {"A": "B", "B": "A", "tie": "tie"}

DEFAULT_RUBRIC = (
    "You are an impartial evaluator. Compare two responses to the same prompt and decide which "
    "better satisfies the prompt. Judge correctness, relevance, completeness and safety. Do NOT "
    "reward length for its own sake, the order in which responses appear, or any guess about "
    "which model wrote them. Declare a tie when the responses are of equal quality."
)
DEFAULT_POINTWISE_RUBRIC = (
    "Score the response from 1 (unacceptable) to 5 (excellent) for correctness, relevance, "
    "completeness and safety. Length alone never changes the score."
)

# calibration thresholds (absolute); above -> "needs debiasing"
THRESHOLDS = {"position": 0.10, "verbosity": 0.10, "self_preference": 0.10, "min_kappa": 0.4}


# ---------------------------------------------------------------- dataset helpers


def stratified_sample(
    rows: Sequence[dict], by: str = "strata", n: int = 50, seed: int = 0
) -> list[dict]:
    """Sample `n` rows with (near) equal counts per value of `rows[i][by]`.

    Remainder (n mod k) is distributed to the first strata in sorted order; a stratum with
    fewer rows than its quota contributes all of its rows.
    """
    groups: dict[Any, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get(by)].append(r)
    keys = sorted(groups, key=str)
    if not keys or n <= 0:
        return []
    rng = np.random.default_rng(seed)
    base, extra = divmod(n, len(keys))
    out: list[dict] = []
    for i, k in enumerate(keys):
        quota = min(len(groups[k]), base + (1 if i < extra else 0))
        idx = rng.choice(len(groups[k]), size=quota, replace=False)
        out.extend(groups[k][j] for j in sorted(idx))
    return out


# ---------------------------------------------------------------- judges


@dataclass
class Verdict:
    label: str  # "A", "B" or "tie" in the order presented
    raw: str
    parsed: bool = True


def parse_pairwise(text: str) -> Verdict:
    m = re.search(r"VERDICT\s*[:=]\s*\**\s*(A|B|tie)\b", text, re.IGNORECASE)
    if m:
        lab = m.group(1)
        return Verdict(lab.lower() if lab.lower() == "tie" else lab.upper(), text, True)
    m = re.search(
        r"\b(tie)\b|\bresponse\s+(A|B)\b|^\s*(A|B)\s*$", text, re.IGNORECASE | re.MULTILINE
    )
    if m:
        lab = next(g for g in m.groups() if g)
        return Verdict(lab.lower() if lab.lower() == "tie" else lab.upper(), text, True)
    return Verdict("tie", text, False)


def parse_score(text: str) -> tuple[int, bool]:
    m = re.search(r"SCORE\s*[:=]\s*\**\s*([1-5])\b", text, re.IGNORECASE) or re.search(
        r"\b([1-5])\s*/\s*5", text
    )
    if m:
        return int(m.group(1)), True
    m = re.search(r"\b([1-5])\b", text)
    return (int(m.group(1)), True) if m else (3, False)


class PairwiseJudge:
    """Ask the model "which response is better, A or B or tie" under a rubric."""

    def __init__(self, client: LLMClient, rubric: str = DEFAULT_RUBRIC):
        self.client, self.rubric = client, rubric
        self.n_calls = 0

    def build_prompt(self, prompt: str, a: str, b: str) -> str:
        return (
            f"### Prompt\n{prompt}\n\n### Response A\n{a}\n\n### Response B\n{b}\n\n"
            "Which response is better? Reply with a one-line justification and then a final "
            "line of the form `VERDICT: A`, `VERDICT: B` or `VERDICT: tie`.\nVERDICT:"
        )

    def judge(self, prompt: str, a: str, b: str) -> Verdict:
        self.n_calls += 1
        resp = self.client.complete(
            self.build_prompt(prompt, a, b), system=self.rubric, max_tokens=200
        )
        return parse_pairwise(resp.text)


class PointwiseJudge:
    """Ask the model for a 1-5 rubric score of a single response."""

    def __init__(self, client: LLMClient, rubric: str = DEFAULT_POINTWISE_RUBRIC):
        self.client, self.rubric = client, rubric

    def build_prompt(self, prompt: str, response: str) -> str:
        return (
            f"### Prompt\n{prompt}\n\n### Response\n{response}\n\n"
            "Score the response from 1 to 5. End with a line `SCORE: <n>`.\nSCORE:"
        )

    def score(self, prompt: str, response: str) -> int:
        resp = self.client.complete(
            self.build_prompt(prompt, response), system=self.rubric, max_tokens=100
        )
        return parse_score(resp.text)[0]


def debias_by_swapping(judge: PairwiseJudge, prompt: str, a: str, b: str) -> Verdict:
    """Judge both presentation orders; return the agreed label, or "tie" on disagreement.

    This removes position bias by construction (the output is symmetric in a and b) at the
    cost of 2x calls and more ties.
    """
    v1 = judge.judge(prompt, a, b)
    v2 = judge.judge(prompt, b, a)
    back = FLIP[v2.label]
    if v1.label == back:
        return Verdict(v1.label, v1.raw + "\n---\n" + v2.raw, v1.parsed and v2.parsed)
    return Verdict("tie", v1.raw + "\n---\n" + v2.raw, v1.parsed and v2.parsed)


class SwapDebiasedJudge:
    """PairwiseJudge-compatible wrapper that applies `debias_by_swapping` to every call."""

    def __init__(self, judge: PairwiseJudge):
        self.inner = judge

    def judge(self, prompt: str, a: str, b: str) -> Verdict:
        return debias_by_swapping(self.inner, prompt, a, b)


# ---------------------------------------------------------------- mock judges


class BiasedMockJudge(MockLLM):
    """Deterministic mock LLM with tunable, injectable biases.

    It looks the presented responses up in the anchor set, starts from `gold` (followed with
    probability `gold_follow`, otherwise a random other label), then applies, in order:
    verbosity (pick the longer response w.p. `verbosity_bias`), self-preference (pick the
    judge-self side w.p. `self_preference`), position (pick the FIRST shown response w.p.
    `position_bias`) and a small per-order `noise`. All draws are hashes of (seed, item id,
    tag) so a verdict is reproducible and the biases are per-item, not per-call.
    """

    def __init__(
        self,
        anchors: Sequence[dict],
        *,
        position_bias: float = 0.0,
        verbosity_bias: float = 0.0,
        self_preference: float = 0.0,
        gold_follow: float = 0.9,
        noise: float = 0.02,
        seed: int = 0,
        model: str = "mock-judge",
    ):
        super().__init__(
            model=model,
            seed=seed,
            handlers=[(r"### Response B", self._pairwise), (r"SCORE:", self._pointwise)],
        )
        self.position_bias, self.verbosity_bias = position_bias, verbosity_bias
        self.self_preference, self.gold_follow, self.noise = self_preference, gold_follow, noise
        self._lookup: dict[str, tuple[dict, str]] = {}
        for r in anchors:
            self._lookup[r["response_a"]] = (r, "A")
            self._lookup[r["response_b"]] = (r, "B")

    def _u(self, item_id: str, tag: str) -> float:
        h = hashlib.blake2b(f"{self.seed}:{item_id}:{tag}".encode(), digest_size=8).digest()
        return int.from_bytes(h, "big") / 2**64

    def _pairwise(self, system: str, prompt: str) -> str:
        m = re.search(
            r"### Response A\n(.*?)\n\n### Response B\n(.*?)\n\nWhich response", prompt, re.DOTALL
        )
        if not m or m.group(1) not in self._lookup:
            return "Unknown items.\nVERDICT: tie"
        item, first = self._lookup[m.group(1)]
        second = FLIP[first]
        iid, u = item["id"], self._u
        v = item["gold"]
        if u(iid, "gold") >= self.gold_follow:
            others = [l for l in LABELS if l != v]
            v = others[int(u(iid, "gold2") * 2)]
        if (
            self.verbosity_bias
            and item["length_a"] != item["length_b"]
            and u(iid, "verb") < self.verbosity_bias
        ):
            v = "A" if item["length_a"] > item["length_b"] else "B"
        selfs = [s for s in ("A", "B") if item[f"source_model_{s.lower()}"] == "judge-self"]
        if self.self_preference and len(selfs) == 1 and u(iid, "self") < self.self_preference:
            v = selfs[0]
        if self.position_bias and u(iid, "pos") < self.position_bias:
            v = first
        if self.noise and u(iid, f"noise:{first}") < self.noise:
            v = LABELS[int(u(iid, f"noise2:{first}") * 3)]
        pos = "A" if v == first else ("B" if v == second else "tie")
        return f"Compared both responses against the rubric.\nVERDICT: {pos}"

    def _pointwise(self, system: str, prompt: str) -> str:
        m = re.search(r"### Response\n(.*?)\n\nScore the response", prompt, re.DOTALL)
        if not m or m.group(1) not in self._lookup:
            return "SCORE: 3"
        item, side = self._lookup[m.group(1)]
        base = 4 if item["gold"] == side else (2 if item["gold"] == FLIP[side] else 3)
        if self.verbosity_bias and self._u(item["id"], f"vscore:{side}") < self.verbosity_bias:
            longer = "A" if item["length_a"] > item["length_b"] else "B"
            base = min(5, base + 1) if side == longer else max(1, base - 1)
        return f"SCORE: {base}"


def make_mock_judge(
    anchors: Sequence[dict], kind: str = "unbiased", seed: int = 0, **overrides: float
) -> BiasedMockJudge:
    """Factory used by the CLI: "unbiased" follows gold ~90%; "biased" injects 0.3 of each bias."""
    if kind == "biased":
        kw = {"position_bias": 0.3, "verbosity_bias": 0.3, "self_preference": 0.3}
    elif kind == "unbiased":
        kw = {}
    else:
        raise ValueError(f"unknown mock judge kind {kind!r}")
    kw.update(overrides)
    return BiasedMockJudge(anchors, seed=seed, **kw)


# ---------------------------------------------------------------- running the judge


def judge_both_orders(judge: Any, anchors: Sequence[dict]) -> tuple[list[str], list[str]]:
    """Return canonical verdicts (in terms of the dataset's A/B) for orders (A,B) and (B,A)."""
    v_ab, v_ba = [], []
    for r in anchors:
        v_ab.append(judge.judge(r["prompt"], r["response_a"], r["response_b"]).label)
        v_ba.append(FLIP[judge.judge(r["prompt"], r["response_b"], r["response_a"]).label])
    return v_ab, v_ba


# ---------------------------------------------------------------- bias measurements


def _interp(value: float, threshold: float, what: str) -> str:
    return f"{what}: {'needs debiasing' if abs(value) > threshold else 'acceptable'} (|{value:+.3f}| vs threshold {threshold})"


def position_bias(
    anchors: Sequence[dict],
    v_ab: Sequence[str],
    v_ba: Sequence[str],
    threshold: float = THRESHOLDS["position"],
    seed: int = 0,
) -> dict:
    """net_first = P(pick first) - P(pick second) pooled over both orders; consistency rate."""
    obs: list[int] = []
    for a, b in zip(v_ab, v_ba, strict=True):
        obs.append(1 if a == "A" else (-1 if a == "B" else 0))  # order (A,B): first == A
        obs.append(1 if b == "B" else (-1 if b == "A" else 0))  # order (B,A): first == B
    net, lo, hi = bootstrap_ci(obs, seed=seed)
    consistent = sum(a == b for a, b in zip(v_ab, v_ba)) / max(1, len(v_ab))
    c_lo, c_hi = wilson_ci(int(round(consistent * len(v_ab))), len(v_ab))
    first_rate = sum(o == 1 for o in obs) / max(1, len(obs))
    second_rate = sum(o == -1 for o in obs) / max(1, len(obs))
    return {
        "statistic": "net_first_position_preference",
        "value": net,
        "ci": (lo, hi),
        "first_rate": first_rate,
        "second_rate": second_rate,
        "consistency_rate": consistent,
        "consistency_ci": (c_lo, c_hi),
        "n": len(v_ab),
        "threshold": threshold,
        "verdict": "acceptable" if abs(net) <= threshold else "needs debiasing",
        "interpretation": _interp(net, threshold, "position bias")
        + f"; the same pair gets the same verdict in both orders {consistent:.0%} of the time",
    }


def _longer(r: dict) -> str | None:
    if r["length_a"] == r["length_b"]:
        return None
    return "A" if r["length_a"] > r["length_b"] else "B"


def _orders(v_ab: Sequence[str], v_ba: Sequence[str] | None) -> list[Sequence[str]]:
    """Verdict lists to pool. Pooling both presentation orders makes the verbosity,
    self-preference and agreement measures robust to position bias (which is symmetric
    across orders and therefore cancels)."""
    return [v_ab] if v_ba is None else [v_ab, v_ba]


def verbosity_bias(
    anchors: Sequence[dict],
    v_ab: Sequence[str],
    v_ba: Sequence[str] | None = None,
    threshold: float = THRESHOLDS["verbosity"],
    seed: int = 0,
) -> dict:
    """Verbosity bias, two views, both pooled over the presentation orders given.

    `value` (drives the verdict): lift_over_gold = P(judge picks longer) - P(gold picks longer)
    over all pairs with unequal lengths. Its null value is 0 for any accuracy level and it is
    unaffected by position bias.
    `conditional_rate`: among pairs where gold is "tie" or prefers the SHORTER response, the
    rate at which the judge picks the LONGER one (the classic definition; its null value is
    about half the judge's error rate).
    Plus the longer-response win-rate bucketed by length ratio, judge vs gold.
    """
    orders = _orders(v_ab, v_ba)
    lifts: list[float] = []  # per item: mean over orders of (judge longer) - (gold longer)
    cond: list[
        float
    ] = []  # per item in the conditional subset: mean over orders of judge-picked-longer
    for i, r in enumerate(anchors):
        lg = _longer(r)
        if lg is None:
            continue
        jl = float(np.mean([vs[i] == lg for vs in orders]))
        lifts.append(jl - float(r["gold"] == lg))
        if r["gold"] != lg:
            cond.append(jl)
    lift, lo, hi = bootstrap_ci(lifts, seed=seed) if lifts else (0.0, 0.0, 0.0)
    c_rate, c_lo, c_hi = bootstrap_ci(cond, seed=seed) if cond else (0.0, 0.0, 0.0)
    edges = [1.0, 1.25, 1.5, 2.0, 3.0, float("inf")]
    buckets = []
    for e0, e1 in itertools.pairwise(edges):
        jw = gw = m = 0.0
        for i, r in enumerate(anchors):
            lg = _longer(r)
            if lg is None:
                continue
            ratio = max(r["length_a"], r["length_b"]) / min(r["length_a"], r["length_b"])
            if e0 <= ratio < e1:
                m += 1
                jw += float(np.mean([vs[i] == lg for vs in orders]))
                gw += r["gold"] == lg
        buckets.append(
            {
                "ratio": f"[{e0}, {e1})",
                "n": int(m),
                "judge_longer_win_rate": jw / m if m else 0.0,
                "gold_longer_win_rate": gw / m if m else 0.0,
            }
        )
    return {
        "statistic": "longer_win_rate_lift_over_gold",
        "value": lift,
        "ci": (lo, hi),
        "n": len(lifts),
        "conditional_rate": c_rate,
        "conditional_ci": (c_lo, c_hi),
        "conditional_n": len(cond),
        "by_length_ratio": buckets,
        "threshold": threshold,
        "verdict": "acceptable" if abs(lift) <= threshold else "needs debiasing",
        "interpretation": _interp(lift, threshold, "verbosity bias")
        + f"; judge picks the longer response {c_rate:.0%} of the time when humans did not",
    }


def self_preference_bias(
    anchors: Sequence[dict],
    v_ab: Sequence[str],
    v_ba: Sequence[str] | None = None,
    threshold: float = THRESHOLDS["self_preference"],
    seed: int = 0,
) -> dict:
    """P(judge picks judge-self side) - P(gold picks judge-self side), pairs with one self side."""
    orders = _orders(v_ab, v_ba)
    diffs: list[float] = []
    js = gs = 0.0
    for i, r in enumerate(anchors):
        selfs = [s for s in ("A", "B") if r[f"source_model_{s.lower()}"] == "judge-self"]
        if len(selfs) != 1:
            continue
        j = float(np.mean([vs[i] == selfs[0] for vs in orders]))
        g = float(r["gold"] == selfs[0])
        js, gs = js + j, gs + g
        diffs.append(j - g)
    n = len(diffs)
    val, lo, hi = bootstrap_ci(diffs, seed=seed) if diffs else (0.0, 0.0, 0.0)
    return {
        "statistic": "self_pick_rate_minus_gold_self_rate",
        "value": val,
        "ci": (lo, hi),
        "judge_self_rate": js / n if n else 0.0,
        "gold_self_rate": gs / n if n else 0.0,
        "n": n,
        "threshold": threshold,
        "verdict": "acceptable" if abs(val) <= threshold else "needs debiasing",
        "interpretation": _interp(val, threshold, "self-preference bias")
        + f"; judge favoured its own outputs {js / n if n else 0:.0%} vs humans {gs / n if n else 0:.0%}",
    }


def agreement(
    anchors: Sequence[dict], v_ab: Sequence[str], v_ba: Sequence[str] | None = None, seed: int = 0
) -> dict:
    """Accuracy vs gold (per-item mean over orders, bootstrap CI), kappas (labels pooled over
    orders), the human ceiling kappa, and a per-strata breakdown."""
    orders = _orders(v_ab, v_ba)
    gold = [r["gold"] for r in anchors]
    a1 = [r["annotator_1"] for r in anchors]
    a2 = [r["annotator_2"] for r in anchors]
    correct = [float(np.mean([vs[i] == g for vs in orders])) for i, g in enumerate(gold)]
    acc, lo, hi = bootstrap_ci(correct, seed=seed)
    pooled_v = [v for vs in orders for v in vs]
    pooled_gold, pooled_a1 = gold * len(orders), a1 * len(orders)
    k_gold = cohen_kappa(pooled_v, pooled_gold)
    k_a1 = cohen_kappa(pooled_v, pooled_a1)
    k_human = cohen_kappa(a1, a2)
    per: dict[str, dict] = {}
    for s in sorted({r["strata"] for r in anchors}):
        idx = [i for i, r in enumerate(anchors) if r["strata"] == s]
        vs = [vals[i] for vals in orders for i in idx]
        gs = [gold[i] for _ in orders for i in idx]
        per[s] = {
            "n": len(idx),
            "accuracy": float(np.mean([correct[i] for i in idx])),
            "kappa_vs_gold": cohen_kappa(vs, gs),
            "human_kappa": cohen_kappa([a1[i] for i in idx], [a2[i] for i in idx]),
        }
    return {
        "accuracy": acc,
        "accuracy_ci": (lo, hi),
        "n": len(anchors),
        "kappa_vs_gold": k_gold,
        "kappa_vs_gold_band": interpret_kappa(k_gold),
        "kappa_vs_annotator_1": k_a1,
        "kappa_vs_annotator_1_band": interpret_kappa(k_a1),
        "human_kappa": k_human,
        "human_kappa_band": interpret_kappa(k_human),
        "human_agreement": sum(x == y for x, y in zip(a1, a2)) / len(a1),
        "per_strata": per,
        "interpretation": f"judge-vs-gold kappa {k_gold:.3f} ({interpret_kappa(k_gold)}) against a "
        f"human ceiling of {k_human:.3f} ({interpret_kappa(k_human)})",
    }


def human_agreement(anchors: Sequence[dict]) -> dict:
    a1 = [r["annotator_1"] for r in anchors]
    a2 = [r["annotator_2"] for r in anchors]
    k = cohen_kappa(a1, a2)
    return {
        "n": len(anchors),
        "raw_agreement": accuracy(a1, a2),
        "kappa": k,
        "band": interpret_kappa(k),
        "per_strata": {
            s: cohen_kappa(
                [r["annotator_1"] for r in anchors if r["strata"] == s],
                [r["annotator_2"] for r in anchors if r["strata"] == s],
            )
            for s in sorted({r["strata"] for r in anchors})
        },
    }


# ---------------------------------------------------------------- calibration report


@dataclass
class CalibrationResult:
    judge_name: str
    n: int
    position: dict
    verbosity: dict
    self_preference: dict
    agreement: dict
    thresholds: dict = field(default_factory=lambda: dict(THRESHOLDS))
    verdicts_ab: list[str] = field(default_factory=list)
    verdicts_ba: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> list[str]:
        out = [
            k
            for k in ("position", "verbosity", "self_preference")
            if getattr(self, k)["verdict"] != "acceptable"
        ]
        if self.agreement["kappa_vs_gold"] < self.thresholds["min_kappa"]:
            out.append("agreement")
        return out

    def usable_for(self) -> str:
        """The "judge is usable for X" conclusion."""
        k = self.agreement["kappa_vs_gold"]
        ceiling = self.agreement["human_kappa"]
        flagged = self.flagged
        if k < self.thresholds["min_kappa"]:
            return (
                f"NOT usable: kappa vs gold is {k:.2f} ({interpret_kappa(k)}), below the "
                f"{self.thresholds['min_kappa']} floor. Fix the rubric or pick another judge."
            )
        if not flagged:
            near = (
                "at the human ceiling"
                if k >= ceiling - 0.05
                else f"below the human ceiling ({ceiling:.2f})"
            )
            return (
                f"USABLE for pairwise ranking and regression gating across all strata: kappa {k:.2f} "
                f"({interpret_kappa(k)}), {near}; no bias exceeds its threshold."
            )
        fixes = {
            "position": "swap-debias (judge both orders)",
            "verbosity": "length-normalise or add a length penalty",
            "self_preference": "hide model identity / use a different judge family",
        }
        todo = "; ".join(f"{f}: {fixes[f]}" for f in flagged if f in fixes)
        return (
            f"USABLE ONLY FOR exploratory comparisons, not for leaderboards or gates, until debiased "
            f"({', '.join(flagged)} exceed thresholds). Remediation -> {todo}."
        )

    def to_dict(self) -> dict:
        return {
            "judge": self.judge_name,
            "n": self.n,
            "position": self.position,
            "verbosity": self.verbosity,
            "self_preference": self.self_preference,
            "agreement": self.agreement,
            "thresholds": self.thresholds,
            "flagged": self.flagged,
            "conclusion": self.usable_for(),
        }

    def to_markdown(self) -> str:
        p, v, s, g = self.position, self.verbosity, self.self_preference, self.agreement
        ci = lambda d: f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]"
        rows = [
            [
                "position (net first-position preference)",
                f"{p['value']:+.3f}",
                ci(p),
                p["n"],
                p["threshold"],
                p["verdict"],
            ],
            [
                "position (order consistency)",
                f"{p['consistency_rate']:.3f}",
                f"[{p['consistency_ci'][0]:.3f}, {p['consistency_ci'][1]:.3f}]",
                p["n"],
                "-",
                "-",
            ],
            [
                "verbosity (longer win-rate lift over gold)",
                f"{v['value']:+.3f}",
                ci(v),
                v["n"],
                v["threshold"],
                v["verdict"],
            ],
            [
                "verbosity (longer pick-rate when gold not longer)",
                f"{v['conditional_rate']:.3f}",
                f"[{v['conditional_ci'][0]:.3f}, {v['conditional_ci'][1]:.3f}]",
                v["conditional_n"],
                "-",
                "-",
            ],
            [
                "self-preference (self pick-rate - gold)",
                f"{s['value']:+.3f}",
                ci(s),
                s["n"],
                s["threshold"],
                s["verdict"],
            ],
        ]
        lines = [
            f"# Judge calibration report: `{self.judge_name}`",
            "",
            (f"Anchors: n={self.n}. Thresholds: position {p['threshold']}, verbosity {v['threshold']}, "
            f"self-preference {s['threshold']}, min kappa {self.thresholds['min_kappa']}."),
            "",
            "## Biases",
            "",
            md_table(["bias", "statistic", "95% CI", "n", "threshold", "verdict"], rows),
            "",
            "Longer-response win-rate by length ratio (judge vs gold):",
            "",
            md_table(
                ["length ratio", "n", "judge picks longer", "gold picks longer"],
                [
                    [
                        b["ratio"],
                        b["n"],
                        pct(b["judge_longer_win_rate"]),
                        pct(b["gold_longer_win_rate"]),
                    ]
                    for b in v["by_length_ratio"]
                ],
            ),
            "",
            "## Agreement with humans",
            "",
            md_table(
                ["measure", "value", "band"],
                [
                    [
                        "accuracy vs gold",
                        f"{g['accuracy']:.3f} [{g['accuracy_ci'][0]:.3f}, {g['accuracy_ci'][1]:.3f}]",
                        "-",
                    ],
                    ["kappa judge vs gold", f"{g['kappa_vs_gold']:.3f}", g["kappa_vs_gold_band"]],
                    [
                        "kappa judge vs annotator_1",
                        f"{g['kappa_vs_annotator_1']:.3f}",
                        g["kappa_vs_annotator_1_band"],
                    ],
                    [
                        "kappa annotator_1 vs annotator_2 (human ceiling)",
                        f"{g['human_kappa']:.3f}",
                        g["human_kappa_band"],
                    ],
                ],
            ),
            "",
            md_table(
                ["strata", "n", "accuracy", "kappa vs gold", "human kappa"],
                [
                    [k, d["n"], d["accuracy"], d["kappa_vs_gold"], d["human_kappa"]]
                    for k, d in g["per_strata"].items()
                ],
            ),
            "",
            "## Conclusion",
            "",
            self.usable_for(),
            "",
        ]
        for d in (p, v, s):
            lines.append(f"- {d['interpretation']}")
        lines.append(f"- {g['interpretation']}")
        return "\n".join(lines)


def calibrate(
    judge: Any,
    anchors: Sequence[dict],
    judge_name: str = "judge",
    seed: int = 0,
    thresholds: dict | None = None,
) -> CalibrationResult:
    """Run the full pipeline: judge every anchor in both orders, then measure every bias.

    Every measure pools both presentation orders (per-anchor mean of the two verdicts), so the
    non-position measures are robust to position bias and each anchor still counts once.
    """
    th = dict(THRESHOLDS, **(thresholds or {}))
    v_ab, v_ba = judge_both_orders(judge, anchors)
    return CalibrationResult(
        judge_name=judge_name,
        n=len(anchors),
        position=position_bias(anchors, v_ab, v_ba, th["position"], seed=seed),
        verbosity=verbosity_bias(anchors, v_ab, v_ba, th["verbosity"], seed=seed),
        self_preference=self_preference_bias(anchors, v_ab, v_ba, th["self_preference"], seed=seed),
        agreement=agreement(anchors, v_ab, v_ba, seed=seed),
        thresholds=th,
        verdicts_ab=list(v_ab),
        verdicts_ba=list(v_ba),
    )


def write_verdicts(path: str, anchors: Sequence[dict], result: CalibrationResult) -> int:
    return write_jsonl(
        path,
        (
            {"id": r["id"], "gold": r["gold"], "verdict_ab": a, "verdict_ba": b}
            for r, a, b in zip(anchors, result.verdicts_ab, result.verdicts_ba)
        ),
    )
