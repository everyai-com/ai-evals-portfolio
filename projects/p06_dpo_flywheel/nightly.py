"""NightlyJob: ingest -> build pairs -> validate -> train (dry-run) -> post-train eval -> promote.

State lives in `state/`: feedback store files, `paired_event_ids.json`, `nightly_state.json`
(watermark + history), `adapters/<date>/`, `reports/eval_diff_<date>.md`, `promoted.json`.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable

from evalkit.io import read_jsonl
from evalkit.llm import LLMClient
from evalkit.report import md_table, pct
from evalkit.stats import paired_bootstrap

from projects.p06_dpo_flywheel.core import (FeedbackStore, FilterConfig, Judge, PairBuilder, append_golden,
                                            heuristic_judge, normalise, write_pairs)
from projects.p06_dpo_flywheel.trainer import train_lora

HERE = Path(__file__).parent

# eval_hook(adapter_dir, previous_adapter_dir_or_None, golden_cases) -> (before_scores, after_scores)
EvalHook = Callable[[Path, Path | None, list[dict[str, Any]]], tuple[list[float], list[float]]]


@dataclass
class NightlyConfig:
    min_pairs: int = 20
    max_tenant_share: float = 0.7
    min_judge_pass_rate: float = 0.6
    val_frac: float = 0.2
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    train: dict[str, Any] = field(default_factory=dict)
    promote_tolerance: float = 0.0     # promote iff delta >= -tolerance and not significantly worse
    seed: int = 0


@dataclass
class NightlyReport:
    date: str
    status: str                       # "promoted" | "not_promoted" | "skipped"
    ingest: dict[str, Any]
    new_pairs: int
    pairs_by_source: dict[str, int]
    validation: dict[str, Any]
    manifest: dict[str, Any] | None = None
    eval: dict[str, Any] | None = None
    adapter_dir: str | None = None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def simulated_eval_hook(adapter_dir: Path, previous: Path | None, golden: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    """Dry-run stand-in for 'run the model on the golden set before/after'.

    A dry-run adapter 'knows' the prompts it was trained on (adapter_memory.jsonl) and answers them
    with the chosen behaviour; everything else it answers like the base model (the rejected
    behaviour, since each golden case *is* a captured base-model failure). Pass = normalised output
    equals expected_behavior. Replace with a real inference hook when EVALKIT_TRAIN=1."""
    def memory(d: Path | None) -> dict[str, str]:
        p = d / "adapter_memory.jsonl" if d else None
        return {normalise(r["prompt"]): r["chosen"] for r in read_jsonl(p)} if p and p.exists() else {}

    def run(mem: dict[str, str]) -> list[float]:
        out = []
        for c in golden:
            answer = mem.get(normalise(c["input"]), c["rejected_behavior"])
            out.append(1.0 if normalise(answer) == normalise(c["expected_behavior"]) else 0.0)
        return out
    return run(memory(previous)), run(memory(adapter_dir))


class NightlyJob:
    def __init__(self, state_dir: str | Path, events_path: str | Path, client: LLMClient | None = None,
                 judge: Judge = heuristic_judge, eval_hook: EvalHook = simulated_eval_hook,
                 golden_path: str | Path | None = None, changelog_path: str | Path | None = None,
                 config: NightlyConfig | None = None, filter_cfg: FilterConfig | None = None):
        self.state = Path(state_dir)
        self.state.mkdir(parents=True, exist_ok=True)
        self.events_path = Path(events_path)
        self.client, self.judge, self.eval_hook = client, judge, eval_hook
        self.golden_path = Path(golden_path) if golden_path else HERE / "data" / "golden_additions.jsonl"
        self.changelog_path = Path(changelog_path) if changelog_path else HERE / "data" / "CHANGELOG.md"
        self.cfg = config or NightlyConfig()
        self.filter_cfg = filter_cfg
        self.state_file = self.state / "nightly_state.json"
        self.paired_file = self.state / "paired_event_ids.json"
        self.promoted_file = self.state / "promoted.json"

    # ------------------------------------------------------------------ helpers
    def _load(self, p: Path, default):
        return json.loads(p.read_text()) if p.exists() else default

    def promoted(self) -> dict[str, Any] | None:
        return self._load(self.promoted_file, None)

    def validate(self, pairs: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
        n = len(pairs)
        tenants = Counter(p.get("tenant") for p in pairs)
        top_share = (tenants.most_common(1)[0][1] / n) if n else 0.0
        passed = sum(1 for p in pairs if self.judge(p["prompt"], p["chosen"]) > self.judge(p["prompt"], p["rejected"]))
        rate = passed / n if n else 0.0
        checks = {"min_pairs": n >= self.cfg.min_pairs,
                  "max_tenant_share": top_share <= self.cfg.max_tenant_share if n else True,
                  "judge_pass_rate": rate >= self.cfg.min_judge_pass_rate if n else True}
        info = {"n_pairs": n, "top_tenant": tenants.most_common(1)[0][0] if n else None, "top_tenant_share": round(top_share, 3),
                "judge_pass_rate": round(rate, 3), "checks": checks,
                "thresholds": {"min_pairs": self.cfg.min_pairs, "max_tenant_share": self.cfg.max_tenant_share,
                               "min_judge_pass_rate": self.cfg.min_judge_pass_rate}}
        return all(checks.values()), info

    # ------------------------------------------------------------------ run
    def run(self, date: str) -> NightlyReport:
        st = self._load(self.state_file, {"history": [], "watermark": None})
        store = FeedbackStore(self.state, self.filter_cfg)
        ingest = store.ingest(read_jsonl(self.events_path)).to_dict()
        events = store.events()
        paired: set[str] = set(self._load(self.paired_file, []))
        builder = PairBuilder(client=self.client, judge=self.judge)
        pairs = builder.build(events, exclude_event_ids=paired)
        by_source = Counter(p["source"] for p in pairs)
        report = NightlyReport(date=date, status="skipped", ingest=ingest, new_pairs=len(pairs),
                               pairs_by_source=dict(by_source), validation={})
        ok, info = self.validate(pairs)
        report.validation = info
        if not ok:
            report.reasons = [f"validation failed: {k}" for k, v in info["checks"].items() if not v]
            return self._finish(st, report)
        # pairs are consumed exactly once; the golden set grows with every pair (the flywheel)
        pairs_dir = self.state / "pairs" / date
        counts = write_pairs(pairs, pairs_dir, self.cfg.val_frac, self.cfg.seed)
        added = append_golden(pairs, self.golden_path, self.changelog_path, today=date)
        self.paired_file.write_text(json.dumps(sorted(paired | {p["event_id"] for p in pairs})))
        adapter_dir = self.state / "adapters" / date
        manifest = train_lora(pairs_dir / "train.jsonl", self.cfg.base_model, adapter_dir, self.cfg.train)
        report.manifest, report.adapter_dir = manifest, str(adapter_dir)
        # post-train eval on the golden additions, paired before/after
        golden = read_jsonl(self.golden_path) if self.golden_path.exists() else []
        prev = self.promoted()
        prev_dir = Path(prev["adapter_dir"]) if prev else None
        before, after = self.eval_hook(adapter_dir, prev_dir, golden)
        rep = paired_bootstrap(before, after, metric="golden_pass_rate", n_boot=5000, seed=self.cfg.seed)
        worse = rep.delta < -self.cfg.promote_tolerance and (rep.significant or rep.delta < -0.02)
        report.eval = {**rep.to_dict(), "n_golden": len(golden), "golden_added": added, "split": counts, "worse": worse}
        (self.state / "reports").mkdir(exist_ok=True)
        (self.state / "reports" / f"eval_diff_{date}.md").write_text(self.render_diff(report, rep))
        if worse:
            report.status, report.reasons = "not_promoted", [f"post-train eval worse: {rep.headline()}"]
        else:
            report.status = "promoted"
            self.promoted_file.write_text(json.dumps({"adapter_dir": str(adapter_dir), "date": date,
                                                      "delta": rep.delta, "golden_pass_rate": rep.b_mean}, indent=2))
        return self._finish(st, report)

    def _finish(self, st: dict[str, Any], report: NightlyReport) -> NightlyReport:
        st["watermark"] = report.ingest.get("watermark")
        st["history"] = (st.get("history", []) + [{"date": report.date, "status": report.status, "new_pairs": report.new_pairs,
                                                   "reasons": report.reasons}])[-100:]
        self.state_file.write_text(json.dumps(st, indent=2))
        return report

    @staticmethod
    def render_diff(report: NightlyReport, rep) -> str:
        rows = [["golden pass rate", pct(rep.a_mean), pct(rep.b_mean), f"{rep.delta * 100:+.1f} pts",
                 f"[{rep.ci_lo * 100:+.1f}, {rep.ci_hi * 100:+.1f}]", f"{rep.p_value:.3g}", rep.n]]
        return (f"# Post-train eval diff — {report.date}\n\n{rep.headline()}\n\n"
                + md_table(["metric", "before", "after", "Δ", "95% CI (pts)", "p", "n"], rows)
                + f"\n\npromote: {'no' if report.eval and report.eval.get('worse') else 'yes'}\n")


def render_report(r: NightlyReport) -> str:
    lines = [f"## Nightly DPO flywheel — {r.date}: **{r.status.upper()}**", "",
             f"- ingest: {r.ingest['received']} received, {r.ingest['duplicates']} duplicates, "
             f"{r.ingest['accepted']} accepted, rejected {r.ingest['rejected']}, PII scrubbed {r.ingest['pii_scrubbed']}",
             f"- new pairs: {r.new_pairs} {dict(r.pairs_by_source)}",
             f"- validation: {r.validation.get('checks')} (judge pass rate {r.validation.get('judge_pass_rate')}, "
             f"top tenant share {r.validation.get('top_tenant_share')})"]
    if r.manifest:
        d = r.manifest["dataset"]
        lines.append(f"- train: {'dry-run' if r.manifest['dry_run'] else 'real'} on {d['n_pairs']} pairs, "
                     f"{d['total_tokens']} tokens, est. {d['estimated_gpu_hours']} GPU-h -> {r.adapter_dir}")
    if r.eval:
        lines.append(f"- post-train eval on {r.eval['n_golden']} golden cases: {r.eval['headline']}")
    for reason in r.reasons:
        lines.append(f"- reason: {reason}")
    return "\n".join(lines) + "\n"
