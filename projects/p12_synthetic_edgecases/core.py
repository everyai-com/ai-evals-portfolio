"""Pipeline: candidates (rules + metamorphic + LLM) -> dedup -> schema validation -> difficulty
filter -> selection -> coverage. Plus `promote` for merging reviewed cases into the golden set.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Any, Callable

import numpy as np

from evalkit.io import read_jsonl, stable_hash, write_jsonl
from evalkit.llm import LLMClient
from evalkit.report import md_table

from projects.p12_synthetic_edgecases.llm_gen import LLMGenerator
from projects.p12_synthetic_edgecases.metamorphic import generate_pairs
from projects.p12_synthetic_edgecases.operators import OPERATORS, apply_operators
from projects.p12_synthetic_edgecases.spec import TaskSpec, char_ngrams, normalise_text, validate_case
from projects.p12_synthetic_edgecases.sut import baseline_sut

SUT = Callable[[str], Any]


def make_case(inp: str, expected: Any, operator: str, parent: dict[str, Any] | None, generator: str, seed: int,
              rationale: str, category: str | None, spec: TaskSpec) -> dict[str, Any]:
    cid = "syn_" + stable_hash([operator, parent["id"] if parent else None, inp], 12)
    return {"id": cid, "input": inp, "expected": expected, "tags": list(parent["tags"]) if parent else ["llm"],
            "operator": operator, "parent_id": parent["id"] if parent else None, "generator": generator, "seed": seed,
            "category": category if category is not None else spec.category_for(operator), "rationale": rationale}


# ----------------------------------------------------------------------------- dedup

class Deduper:
    """Near-duplicate removal: exact match on normalised text (global) plus character 5-gram
    Jaccard >= threshold against the golden parent and against accepted siblings of the same parent
    (cases without a parent, e.g. LLM output, are compared against the whole golden set).
    Scoping the Jaccard check keeps it O(n * k) instead of O(n^2); cross-parent near-duplicates are
    caught by the exact check because different parents differ in qty/sku/date. A size pre-filter
    skips pairs whose gram-set sizes make the threshold unreachable (J <= min/max)."""

    def __init__(self, golden: list[dict[str, Any]], threshold: float = 0.85, n: int = 5):
        self.threshold, self.n = threshold, n
        self.exact: set[str] = {normalise_text(g["input"]) for g in golden}
        self.golden_grams: dict[str, set[str]] = {g["id"]: char_ngrams(g["input"], n) for g in golden}
        self.by_parent: dict[str | None, list[set[str]]] = defaultdict(list)

    def _near(self, grams: set[str], pool: list[set[str]]) -> bool:
        la = len(grams)
        for g in pool:
            lb = len(g)
            if min(la, lb) / max(la, lb, 1) < self.threshold:
                continue
            if len(grams & g) / len(grams | g) >= self.threshold:
                return True
        return False

    def check(self, case: dict[str, Any]) -> str | None:
        """Returns a reason when `case` is a duplicate, else registers it and returns None."""
        key = normalise_text(case["input"])
        if key in self.exact:
            return "exact_duplicate"
        grams = char_ngrams(case["input"], self.n)
        pid = case.get("parent_id")
        golden_pool = [self.golden_grams[pid]] if pid in self.golden_grams else list(self.golden_grams.values())
        if self._near(grams, golden_pool):
            return "near_duplicate_of_golden"
        if self._near(grams, self.by_parent[pid]):
            return "near_duplicate_of_sibling"
        self.exact.add(key)
        self.by_parent[pid].append(grams)
        return None


# ----------------------------------------------------------------------------- difficulty

@dataclass
class DifficultyEstimator:
    """Run the candidate through a baseline SUT `n_runs` times. A labelled case is 'actually hard'
    when the SUT fails at least once or its outputs disagree across runs; unlabelled cases are
    marked 'unlabelled' (they go to the review queue regardless)."""
    sut: SUT = baseline_sut
    n_runs: int = 3

    def assess(self, case: dict[str, Any]) -> dict[str, Any]:
        outs = []
        for _ in range(self.n_runs):
            try:
                outs.append(self.sut(case["input"]))
            except Exception as e:  # a crash is a failure, and a finding
                outs.append({"error": f"exception: {type(e).__name__}"})
        distinct = {json.dumps(o, sort_keys=True, default=str) for o in outs}
        disagreement = len(distinct) > 1
        if case.get("expected") is None:
            return {"hard": True, "status": "unlabelled", "failures": None, "disagreement": disagreement, "baseline_output": outs[0]}
        failures = sum(1 for o in outs if o != case["expected"])
        return {"hard": failures > 0 or disagreement, "status": "hard" if failures > 0 or disagreement else "easy",
                "failures": failures, "disagreement": disagreement, "baseline_output": outs[0]}


# ----------------------------------------------------------------------------- coverage

@dataclass
class Coverage:
    counts: dict[str, int]                   # "operator|tag|category" -> n
    by_operator: dict[str, int]
    by_category: dict[str, int]
    tag_by_category: dict[str, dict[str, int]]
    uncovered: list[tuple[str, str]]         # (tag, category) cells with zero cases
    operator_category_uncovered: list[str]   # taxonomy categories no operator produced

    def to_dict(self) -> dict[str, Any]:
        return {"counts": self.counts, "by_operator": self.by_operator, "by_category": self.by_category,
                "tag_by_category": self.tag_by_category, "uncovered": [list(u) for u in self.uncovered],
                "operator_category_uncovered": self.operator_category_uncovered}


def coverage_matrix(cases: list[dict[str, Any]], spec: TaskSpec, golden: list[dict[str, Any]]) -> Coverage:
    tags = sorted({t for g in golden for t in g.get("tags", [])})
    cats = list(spec.failure_taxonomy)
    counts: Counter = Counter()
    tbc: dict[str, Counter] = {t: Counter() for t in tags}
    for c in cases:
        cat = c.get("category") or "uncategorised"
        for t in c.get("tags") or ["untagged"]:
            counts[f"{c['operator']}|{t}|{cat}"] += 1
            if t in tbc:
                tbc[t][cat] += 1
    by_op = Counter(c["operator"] for c in cases)
    by_cat = Counter(c.get("category") or "uncategorised" for c in cases)
    uncovered = [(t, k) for t in tags for k in cats if tbc[t][k] == 0]
    return Coverage(dict(counts), dict(by_op), dict(by_cat), {t: dict(v) for t, v in tbc.items()}, uncovered,
                    [k for k in cats if by_cat[k] == 0])


def render_coverage(cov: Coverage, spec: TaskSpec, stats: dict[str, Any]) -> str:
    cats = list(spec.failure_taxonomy)
    lines = ["# Coverage report", "", f"- candidates: {stats['candidates']} -> kept {stats['kept']} "
             f"(dropped: {stats['dropped']})", f"- labelled {stats['labelled']}, review queue {stats['review_queue']}", "",
             "## By operator", "", md_table(["operator", "cases"], sorted(cov.by_operator.items())), "",
             "## By taxonomy category", "", md_table(["category", "cases"], [[k, cov.by_category.get(k, 0)] for k in cats]), "",
             "## Tag x category", "",
             md_table(["tag"] + cats, [[t] + [cov.tag_by_category[t].get(k, 0) for k in cats] for t in sorted(cov.tag_by_category)]),
             "", f"## Uncovered cells ({len(cov.uncovered)})", ""]
    lines += [f"- {t} x {k}" for t, k in cov.uncovered] or ["- none"]
    if cov.operator_category_uncovered:
        lines += ["", "## Categories with no cases at all", ""] + [f"- {k}" for k in cov.operator_category_uncovered]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- pipeline

@dataclass
class GenerationResult:
    kept: list[dict[str, Any]]
    dropped: list[dict[str, Any]]
    review_queue: list[dict[str, Any]]
    coverage: Coverage
    stats: dict[str, Any]
    llm_rejected: list[dict[str, Any]] = field(default_factory=list)


def generate_cases(golden: list[dict[str, Any]], spec: TaskSpec, n: int = 200, seed: int = 0,
                   client: LLMClient | None = None, sut: SUT = baseline_sut, n_runs: int = 3,
                   operators: list[str] | None = None, llm_n: int = 12, dedup_threshold: float = 0.85,
                   keep_easy: bool = False) -> GenerationResult:
    cands: list[dict[str, Any]] = []
    for g in golden:
        for m in apply_operators(g, spec, seed, operators):
            cands.append(make_case(m.mutated_input, m.expected, m.operator_name, g, "rule", seed, m.rationale, None, spec))
        for p in generate_pairs(g, seed):
            cands.append(make_case(p.follow_up_input, p.expected, p.relation, g, "metamorphic", seed, p.rationale, None, spec))
    llm_rejected: list[dict[str, Any]] = []
    if client is not None:
        gen = LLMGenerator(client)
        for r in gen.generate(spec, golden, list(operators or OPERATORS), llm_n, seed):
            cands.append(make_case(r["input"], r["expected"], "llm", None, "llm", seed, r["rationale"], r["category"], spec))
        llm_rejected = gen.rejected
    dedup = Deduper(golden, dedup_threshold)
    est = DifficultyEstimator(sut, n_runs)
    kept, dropped = [], []
    drop_reasons: Counter = Counter()
    for c in cands:
        reason = dedup.check(c)
        if reason is None:
            errs = validate_case(c, spec)
            reason = "schema: " + "; ".join(errs) if errs else None
        if reason is None:
            d = est.assess(c)
            c["difficulty"] = d
            if not d["hard"] and not keep_easy:
                reason = "baseline_always_passes"
        if reason:
            drop_reasons[reason.split(":")[0]] += 1
            dropped.append({"id": c["id"], "input": c["input"], "operator": c["operator"], "reason": reason})
        else:
            kept.append(c)
    kept = select_round_robin(kept, n, seed)
    review = [c for c in kept if c["expected"] is None]
    cov = coverage_matrix(kept, spec, golden)
    stats = {"candidates": len(cands), "kept": len(kept), "dropped": dict(drop_reasons), "labelled": len(kept) - len(review),
             "review_queue": len(review), "llm_rejected": len(llm_rejected), "seed": seed,
             "by_generator": dict(Counter(c["generator"] for c in kept))}
    return GenerationResult(kept, dropped, review, cov, stats, llm_rejected)


def select_round_robin(cases: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """Deterministic selection of at most n cases, cycling over operators so coverage is maximised."""
    if len(cases) <= n:
        return cases
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        buckets[c["operator"]].append(c)
    for b in buckets.values():
        rng.shuffle(b)
    out: list[dict[str, Any]] = []
    while len(out) < n:
        for op in sorted(buckets):
            if buckets[op] and len(out) < n:
                out.append(buckets[op].pop())
    return out


def write_outputs(res: GenerationResult, out_dir: str | Path, spec: TaskSpec) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "generated.jsonl", res.kept)
    write_jsonl(out / "review_queue.jsonl", [{"id": c["id"], "input": c["input"], "operator": c["operator"],
                                               "rationale": c["rationale"], "baseline_output": c["difficulty"]["baseline_output"],
                                               "expected": None, "approved": None} for c in res.review_queue])
    write_jsonl(out / "dropped.jsonl", res.dropped)
    (out / "coverage.md").write_text(render_coverage(res.coverage, spec, res.stats))
    (out / "stats.json").write_text(json.dumps({**res.stats, "coverage": res.coverage.to_dict()}, indent=1))
    return {k: str(out / v) for k, v in {"generated": "generated.jsonl", "review_queue": "review_queue.jsonl",
                                          "dropped": "dropped.jsonl", "coverage": "coverage.md", "stats": "stats.json"}.items()}


# ----------------------------------------------------------------------------- promote

def bump_minor(version: str) -> str:
    major, minor, _patch = (int(x) for x in version.strip().split("."))
    return f"{major}.{minor + 1}.0"


def promote(generated: list[dict[str, Any]], reviewed: list[dict[str, Any]], golden_path: str | Path,
            version_path: str | Path, changelog_path: str | Path, spec: TaskSpec, include_labelled: bool = False,
            today: str | None = None) -> dict[str, Any]:
    """Merge reviewed (approved) cases, and optionally all auto-labelled ones, into the golden set.
    Bumps the minor version in `version_path` and appends a CHANGELOG entry. Idempotent by id."""
    golden_path, version_path, changelog_path = Path(golden_path), Path(version_path), Path(changelog_path)
    golden = read_jsonl(golden_path) if golden_path.exists() else []
    existing = {g["id"] for g in golden}
    by_id = {c["id"]: c for c in generated}
    additions, skipped = [], Counter()
    for r in reviewed:
        c = by_id.get(r.get("id"))
        if c is None or not r.get("approved"):
            skipped["not_approved_or_unknown"] += 1
            continue
        merged = {**c, "expected": r.get("expected", c["expected"])}
        errs = validate_case(merged, spec)
        if merged["expected"] is None or errs:
            skipped["invalid_label"] += 1
            continue
        additions.append(merged)
    if include_labelled:
        additions += [c for c in generated if c["expected"] is not None and c["id"] not in {a["id"] for a in additions}]
    new_rows = []
    for c in additions:
        if c["id"] in existing:
            skipped["already_in_golden"] += 1
            continue
        existing.add(c["id"])
        new_rows.append({"id": c["id"], "input": c["input"], "expected": c["expected"],
                         "tags": sorted(set(c.get("tags", [])) | {"synthetic", c["operator"]} | ({c["category"]} if c.get("category") else set()))})
    if not new_rows:
        return {"added": 0, "skipped": dict(skipped), "version": version_path.read_text().strip() if version_path.exists() else None}
    write_jsonl(golden_path, golden + new_rows)
    old = version_path.read_text().strip() if version_path.exists() else "0.0.0"
    new = bump_minor(old)
    version_path.write_text(new + "\n")
    today = today or _date.today().isoformat()
    ops = Counter(c["operator"] for c in additions if c["id"] in {r["id"] for r in new_rows})
    if not changelog_path.exists():
        changelog_path.write_text("# Golden set changelog\n\n")
    with open(changelog_path, "a", encoding="utf-8") as f:
        f.write(f"- {new} ({today}): +{len(new_rows)} synthetic cases promoted (" +
                ", ".join(f"{k}={v}" for k, v in sorted(ops.items())) + f"); total {len(golden) + len(new_rows)}\n")
    return {"added": len(new_rows), "skipped": dict(skipped), "version": new, "previous_version": old,
            "total": len(golden) + len(new_rows)}
