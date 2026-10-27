"""RefactorBudget: thresholds -> verdict + ranked refactor plan with line estimates.

Line accounting: every action claims a set of (file, line) pairs it would delete; an action's
`lines_removed` is the number of lines it claims that no earlier action already claimed, so the
estimates never double count and `projected_lines == lines_before - sum(lines_removed)` by
construction (tested).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from evalkit.report import md_table, pct
from projects.p23_bloat_detector.core import FunctionInfo, RepoMetrics, diary_line_set
from projects.p23_bloat_detector.velocity import Velocity

DEFAULT_THRESHOLDS = Path(__file__).parent / "data" / "thresholds.yaml"


@dataclass
class Action:
    rank: int
    kind: str
    description: str
    files: list[str]
    lines_removed: int


@dataclass
class Breach:
    metric: str
    value: float
    threshold: float
    rule: str


@dataclass
class RefactorPlan:
    book_the_refactor: bool
    breaches: list[Breach]
    actions: list[Action]
    lines_before: int           # non-blank source lines
    projected_lines: int
    code_lines_before: int
    thresholds: dict[str, Any] = field(default_factory=dict)

    @property
    def lines_removed(self) -> int:
        return sum(a.lines_removed for a in self.actions)

    def to_dict(self) -> dict[str, Any]:
        return {"book_the_refactor": self.book_the_refactor, "breaches": [asdict(b) for b in self.breaches],
                "actions": [asdict(a) for a in self.actions], "lines_before": self.lines_before,
                "lines_removed": self.lines_removed, "projected_lines": self.projected_lines,
                "code_lines_before": self.code_lines_before, "thresholds": self.thresholds}


def load_thresholds(path: str | Path = DEFAULT_THRESHOLDS) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class RefactorBudget:
    """Verdict rules (each key optional; `null` disables the rule):
      comment_ratio_max, duplicate_pct_max, concept_cluster_max (any cluster with >= N members),
      wrapper_depth_max, diary_lines_max, dead_code_max, near_duplicates_max, commits_48h_max.
    """

    def __init__(self, thresholds: dict[str, Any] | None = None):
        self.t = dict(thresholds) if thresholds is not None else load_thresholds()

    # ---- verdict
    def breaches(self, m: RepoMetrics, velocity: Velocity | None = None) -> list[Breach]:
        t = self.t
        checks: list[tuple[str, float, float | None, str, bool]] = [
            ("comment_ratio", m.comment_ratio, t.get("comment_ratio_max"), ">", True),
            ("duplicate_pct", m.duplicate_pct, t.get("duplicate_pct_max"), ">", True),
            ("max_concept_cluster", m.max_concept_cluster, t.get("concept_cluster_max"), ">=", True),
            ("max_wrapper_depth", m.max_wrapper_depth, t.get("wrapper_depth_max"), ">", True),
            ("diary_lines", m.diary_lines, t.get("diary_lines_max"), ">", True),
            ("dead_code_candidates", len(m.dead_code), t.get("dead_code_max"), ">", True),
            ("near_duplicates", len(m.near_duplicates), t.get("near_duplicates_max"), ">", True),
        ]
        if velocity is not None:
            checks.append(("max_commits_48h", velocity.max_commits_48h, t.get("commits_48h_max"), ">=", True))
        out = []
        for metric, value, thr, op, _ in checks:
            if thr is None:
                continue
            hit = value > thr if op == ">" else value >= thr
            if hit:
                out.append(Breach(metric, round(float(value), 4), float(thr), f"{metric} {op} {thr}"))
        return out

    # ---- plan
    def plan(self, m: RepoMetrics, velocity: Velocity | None = None) -> RefactorPlan:
        fns = {f.ref: f for f in m.functions()}
        claimed: set[tuple[str, int]] = set()
        actions: list[Action] = []

        def claim(fn: FunctionInfo) -> int:
            new = {(fn.file, ln) for ln in range(fn.lineno, fn.end_lineno + 1)} - claimed
            claimed.update(new)
            return len(new)

        def add(kind: str, desc: str, files: list[str], lines: int) -> None:
            actions.append(Action(0, kind, desc, sorted(set(files)), lines))

        for c in m.duplicate_clusters:
            lines = sum(claim(fns[r]) for r in c.members if r != c.keep)
            names = ", ".join(f"`{fns[r].qualname}`" for r in c.members)
            add("merge_duplicates", f"merge {len(c.members)} identical implementations ({names}) -> keep `{c.keep}`",
                [fns[r].file for r in c.members], lines)
        for d in m.near_duplicates:
            a, b = fns[d.a], fns[d.b]
            smaller = b if b.n_lines <= a.n_lines else a
            lines = claim(smaller)
            add("merge_near_duplicates",
                f"fold `{smaller.qualname}` into `{(a if smaller is b else b).qualname}` (5-gram Jaccard {d.jaccard:.2f}); parameterise the difference",
                [a.file, b.file], lines)
        for w in m.wrapper_chains:
            wrappers = [fns[r] for r in w.chain[:-1] if r in fns]
            lines = sum(claim(f) for f in wrappers)
            add("collapse_wrappers",
                f"collapse wrapper chain {' -> '.join(f'`{fns[r].qualname}`' if r in fns else f'`{r}`' for r in w.chain)} (depth {w.depth}); call the target directly",
                [f.file for f in wrappers], lines)
        for c in m.concept_clusters:
            members = [fns[r] for r in c.members]
            diary_in = {f.path: len(f.diary) for f in m.files}
            # keep: the one no earlier action deletes, then most referenced, documented, in the
            # least diary-ridden file, then the longest (most complete) implementation
            keep = max(members, key=lambda f: ((f.file, f.lineno) not in claimed, f.ref_count, f.has_docstring,
                                               -diary_in.get(f.file, 0), f.n_lines, -len(f.qualname)))
            lines = sum(claim(f) for f in members if f is not keep)
            if lines == 0:
                continue  # fully covered by the duplicate / wrapper actions above
            add("unify_concept",
                f"merge {len(members)} implementations of `{c.stem}` ({', '.join('`' + f.name + '`' for f in members)}) -> keep `{keep.ref}`",
                [f.file for f in members], lines)
        for f in m.files:
            if not f.diary:
                continue
            new = {(f.path, ln) for ln in diary_line_set(f.diary)} - claimed
            claimed.update(new)
            if new:
                kinds = sorted({d.kind for d in f.diary})
                add("delete_diary", f"delete {len(new)} lines of diary comments in {f.path} ({', '.join(kinds)})", [f.path], len(new))
        for d in m.dead_code:
            lines = claim(fns[d.ref])
            if lines:
                add("delete_dead_code", f"delete unused `{d.name}` in {d.file} (never referenced)", [d.file], lines)
        for o in m.file_outliers:
            add("split_file", f"split {o.path}: {o.reason}", [o.path], 0)

        actions.sort(key=lambda a: (-a.lines_removed, a.kind, a.description))
        for i, a in enumerate(actions, 1):
            a.rank = i
        breaches = self.breaches(m, velocity)
        before = m.source_lines
        removed = sum(a.lines_removed for a in actions)
        return RefactorPlan(bool(breaches), breaches, actions, before, before - removed, m.code_lines, dict(self.t))


# ---------------------------------------------------------------- rendering


def render_scan(m: RepoMetrics, plan: RefactorPlan, velocity: Velocity | None = None, max_rows: int = 12) -> str:
    s = m.summary()
    out = [f"## bloat scan: `{m.root}`", "",
           md_table(["files", "code", "comment", "docstring", "blank", "comment ratio", "diary lines", "dup lines", "dup %"],
                    [[s["files"], s["code_lines"], s["comment_lines"], s["docstring_lines"], s["blank_lines"],
                      f"{s['comment_ratio']:.2f}", s["diary_lines"], s["duplicated_lines"], pct(s["duplicate_pct"])]]), ""]
    out.append(md_table(["duplicate clusters", "near-duplicates", "wrapper chains (max depth)", "concept clusters (max)", "dead code", "file outliers"],
                        [[s["duplicate_clusters"], s["near_duplicates"], f"{s['wrapper_chains']} ({s['max_wrapper_depth']})",
                          f"{s['concept_clusters']} ({s['max_concept_cluster']})", s["dead_code_candidates"], s["file_outliers"]]]))
    if m.duplicate_clusters:
        out += ["", "### Duplicate function bodies (identifiers normalised)", "",
                md_table(["members", "lines each", "removable"],
                         [[", ".join(c.members), c.lines_each, c.duplicated_lines] for c in m.duplicate_clusters[:max_rows]])]
    if m.near_duplicates:
        out += ["", "### Near-duplicate implementations (5-gram Jaccard)", "",
                md_table(["a", "b", "jaccard", "lines"], [[d.a, d.b, d.jaccard, f"{d.lines_a}/{d.lines_b}"] for d in m.near_duplicates[:max_rows]])]
    if m.wrapper_chains:
        out += ["", "### Wrapper chains", "",
                md_table(["chain", "depth"], [[" -> ".join(w.chain), w.depth] for w in m.wrapper_chains[:max_rows]])]
    if m.concept_clusters:
        out += ["", "### Concept redundancy (same stem, different names)", "",
                md_table(["stem", "members"], [[c.stem, ", ".join(c.members)] for c in m.concept_clusters[:max_rows]])]
    if m.dead_code:
        out += ["", "### Dead code candidates (top-level, never referenced)", "",
                md_table(["function", "lines"], [[d.ref, d.n_lines] for d in m.dead_code[:max_rows]])]
    diary = [d for f in m.files for d in f.diary]
    if diary:
        out += ["", f"### Diary comments ({len(diary)} findings, {m.diary_lines} lines)", "",
                md_table(["file:line", "kind", "text"],
                         [[f"{d.file}:{d.line}" + (f"-{d.end_line}" if d.end_line != d.line else ""), d.kind, d.text.replace("|", "\\|")]
                          for d in diary[:max_rows]])]
    if m.file_outliers:
        out += ["", "### File size outliers", "", md_table(["file", "code lines", "reason"], [[o.path, o.code_lines, o.reason] for o in m.file_outliers])]
    if velocity is not None:
        out += ["", render_velocity(velocity)]
    out += ["", render_plan(plan)]
    return "\n".join(out)


def render_plan(plan: RefactorPlan, max_rows: int = 20) -> str:
    out = ["## refactor plan", "",
           f"**book_the_refactor: {'true' if plan.book_the_refactor else 'false'}**"
           + (" — " + "; ".join(f"{b.metric}={b.value:g} ({b.rule})" for b in plan.breaches) if plan.breaches else " — every threshold respected"), ""]
    if plan.actions:
        out.append(md_table(["#", "action", "lines removed"],
                            [[a.rank, a.description.replace("|", "\\|"), a.lines_removed] for a in plan.actions[:max_rows]]))
        if len(plan.actions) > max_rows:
            out.append(f"\n... {len(plan.actions) - max_rows} more actions")
    else:
        out.append("no actions: nothing to merge, collapse or delete.")
    out += ["", (f"source lines before: {plan.lines_before} -> projected after refactor: {plan.projected_lines} "
                f"(-{plan.lines_removed}, {pct(plan.lines_removed / max(1, plan.lines_before))})")]
    return "\n".join(out)


def render_velocity(v: Velocity) -> str:
    rows = [[v.days, v.n_commits, v.max_commits_48h, v.busiest_window_start or "-", v.lines_added, v.lines_deleted, v.add_delete_ratio]]
    s = ["### git velocity", "",
         md_table(["days", "commits", "max in 48h", "busiest window start", "added", "deleted", "add/delete"], rows)]
    if v.busiest_days:
        s.append("\nbusiest days: " + ", ".join(f"{d} ({n})" for d, n in v.busiest_days))
    return "\n".join(s)
