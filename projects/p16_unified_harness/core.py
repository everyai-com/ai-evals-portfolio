"""Unified Eval Harness: run several eval frameworks behind one CLI and produce one report.

Agreement between frameworks is Cohen's kappa (evalkit.metrics.cohen_kappa) over their pass/fail
decisions on the same example ids; the disagreement list names every example where at least two
frameworks disagree, because the scaffolding lies: the same output can "pass" one framework's
definition and "fail" another's."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import yaml

from evalkit.io import write_jsonl
from evalkit.metrics import cohen_kappa, interpret_kappa
from evalkit.report import md_table

from .adapters import ADAPTERS, UnifiedResult, load_dataset
from .sut import get_sut

UNIFIED = ["correctness", "faithfulness", "relevancy", "pass_rate"]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    for d in cfg.get("datasets", []):
        p = Path(d["path"])
        d["path"] = str(p if p.is_absolute() else (path.parent / p))
    cfg.setdefault("options", {})
    cfg.setdefault("metric_map", {})
    cfg.setdefault("unified_metrics", UNIFIED)
    cfg["_config_dir"] = str(path.parent)
    return cfg


def normalise_metrics(framework: str, metrics: dict[str, float], metric_map: dict[str, dict[str, str]]) -> dict[str, float | None]:
    """Map framework-native metric names to the unified vocabulary; unmapped unified names -> None."""
    table = metric_map.get(framework, {})
    out: dict[str, float | None] = {u: None for u in UNIFIED}
    for native, unified in table.items():
        if native in metrics:
            out[unified] = float(metrics[native])
    return out


def run_harness(cfg: dict[str, Any], frameworks: list[str] | None = None,
                adapters: dict[str, Any] | None = None) -> dict[str, Any]:
    names = frameworks or cfg.get("frameworks", list(ADAPTERS))
    unknown = [n for n in names if n not in (adapters or ADAPTERS)]
    if unknown:
        raise KeyError(f"unknown frameworks {unknown}; choose from {sorted(ADAPTERS)}")
    sut_fn = get_sut(cfg.get("sut", "faq_bot"))
    run_cfg = {**cfg, "sut_fn": sut_fn}
    rows, results = [], {}
    for ds in cfg["datasets"]:
        examples = load_dataset(ds["path"])
        for name in names:
            adapter = (adapters or {}).get(name) or ADAPTERS[name]()
            res: UnifiedResult = adapter.run(ds["path"], run_cfg)
            results[(ds["name"], name)] = res
            rows.append({"dataset": ds["name"], "framework": name, "backend": res.backend,
                         "unified": normalise_metrics(name, res.metrics, cfg["metric_map"]),
                         "raw": res.metrics, "cost_usd": res.cost_usd, "duration_s": res.duration_s,
                         "warnings": res.warnings, "n": len(res.per_example)})
        results[(ds["name"], "_examples")] = examples
    agreement = agreement_analysis(cfg, names, results)
    return {"frameworks": names, "rows": rows, "agreement": agreement["pairs"],
            "disagreements": agreement["disagreements"], "results": results}


def agreement_analysis(cfg: dict[str, Any], names: list[str], results: dict) -> dict[str, Any]:
    pairs, disagreements = [], []
    for ds in cfg["datasets"]:
        verdicts = {n: {p["id"]: p for p in results[(ds["name"], n)].per_example} for n in names}
        ids = [e["id"] for e in results[(ds["name"], "_examples")]]
        for a, b in itertools.combinations(names, 2):
            va = [int(verdicts[a][i]["passed"]) for i in ids]
            vb = [int(verdicts[b][i]["passed"]) for i in ids]
            k = cohen_kappa(va, vb)
            pairs.append({"dataset": ds["name"], "a": a, "b": b, "kappa": k, "interpretation": interpret_kappa(k),
                          "n": len(ids), "n_disagree": sum(x != y for x, y in zip(va, vb))})
        for ex in results[(ds["name"], "_examples")]:
            vs = {n: bool(verdicts[n][ex["id"]]["passed"]) for n in names}
            if len(set(vs.values())) > 1:
                out = next((verdicts[n][ex["id"]]["details"].get("output") for n in names
                            if verdicts[n][ex["id"]]["details"].get("output")), "")
                disagreements.append({"dataset": ds["name"], "id": ex["id"], "input": ex["input"],
                                      "expected": ex["expected"], "output": out, "verdicts": vs,
                                      "scores": {n: verdicts[n][ex["id"]]["score"] for n in names}})
    return {"pairs": pairs, "disagreements": disagreements}


def export_jsonl(report: dict[str, Any], cfg: dict[str, Any], path: str | Path) -> int:
    """evalkit pytest-plugin-compatible rows: id/input/expected/tags plus output, per-framework
    verdicts and a majority `passed` so `pytest --eval-dataset results.jsonl` can replay them."""
    rows = []
    names = report["frameworks"]
    for ds in cfg["datasets"]:
        examples = report["results"][(ds["name"], "_examples")]
        per = {n: {p["id"]: p for p in report["results"][(ds["name"], n)].per_example} for n in names}
        for ex in examples:
            verdicts = {n: bool(per[n][ex["id"]]["passed"]) for n in names}
            out = next((per[n][ex["id"]]["details"].get("output") for n in names
                        if per[n][ex["id"]]["details"].get("output")), "")
            rows.append({"id": ex["id"], "input": ex["input"], "expected": ex["expected"],
                         "tags": list(ex.get("tags", [])) + [f"dataset:{ds['name']}"], "output": out,
                         "verdicts": verdicts, "scores": {n: per[n][ex["id"]]["score"] for n in names},
                         "passed": sum(verdicts.values()) * 2 > len(verdicts)})
    return write_jsonl(path, rows)


def _f(v: Any) -> str:
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def render_markdown(report: dict[str, Any], cfg: dict[str, Any]) -> str:
    out = ["# Unified eval harness report", "", f"SUT: `{cfg.get('sut')}`  |  frameworks: {', '.join(report['frameworks'])}", ""]
    out.append("## Dataset x framework x unified metric")
    rows = [[r["dataset"], r["framework"], r["backend"], r["n"]] + [_f(r["unified"][u]) for u in UNIFIED]
            + [f"{r['cost_usd']:.5f}", f"{r['duration_s'] * 1000:.1f}ms"] for r in report["rows"]]
    out.append(md_table(["dataset", "framework", "backend", "n"] + UNIFIED + ["cost_usd", "runtime"], rows))
    out.append("")
    out.append("## Agreement between frameworks (Cohen's kappa on pass/fail)")
    out.append(md_table(["dataset", "A", "B", "kappa", "interpretation", "disagree/n"],
                        [[p["dataset"], p["a"], p["b"], f"{p['kappa']:.3f}", p["interpretation"], f"{p['n_disagree']}/{p['n']}"]
                         for p in report["agreement"]]))
    out.append("")
    out.append("## Disagreements (the scaffolding lies)")
    if report["disagreements"]:
        names = report["frameworks"]
        out.append(md_table(["dataset", "id", "expected", "output"] + names,
                            [[d["dataset"], d["id"], str(d["expected"])[:30], d["output"][:60].replace("|", "/")]
                             + [("pass" if d["verdicts"][n] else "FAIL") + f" ({d['scores'][n]:.2f})" for n in names]
                             for d in report["disagreements"]]))
    else:
        out.append("none")
    out.append("")
    warns = [(r["dataset"], r["framework"], w) for r in report["rows"] for w in r["warnings"]]
    out.append("## Warnings")
    out.append(md_table(["dataset", "framework", "warning"], warns) if warns else "none")
    return "\n".join(out)


def to_json(report: dict[str, Any]) -> dict[str, Any]:
    return {"frameworks": report["frameworks"], "rows": report["rows"], "agreement": report["agreement"],
            "disagreements": report["disagreements"],
            "per_example": {f"{ds}/{fw}": res.to_dict() for (ds, fw), res in report["results"].items() if fw != "_examples"}}
