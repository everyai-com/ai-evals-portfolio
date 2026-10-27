"""Markdown / JSON rendering for the harness result."""
from __future__ import annotations

from typing import Any

from evalkit.report import md_table


def _f(v: Any) -> str:
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def render_markdown(result: dict[str, Any], top_failures: int = 5) -> str:
    out = ["# RAG adversarial harness", "",
           (f"strict grounding: **{'on' if result['strict_grounding'] else 'off'}**  |  k={result['k']}  |  "
           f"seed={result['seed']}  |  overall: **{'PASS' if result['passed'] else 'FAIL'}**"), ""]
    rows = []
    for v in result["variants"]:
        rm, gm = v["retrieval"]["metrics"], v["generation"]["metrics"]
        rows.append([v["variant"], v["n"], _f(rm["hit_rate_at_k"]), _f(rm["recall_at_k"]), _f(rm["mrr"]),
                     _f(rm["ndcg_at_k"]), _f(gm["faithfulness"]), _f(gm["citation_precision"]),
                     _f(gm["correctness"]), _f(gm["correctness_f1"]), _f(gm["over_abstain_rate"]), _f(gm["abstention_precision"]), _f(gm["abstention_recall"]),
                     _f(gm["confident_lie_rate"]), _f(gm["injection_follow_rate"]),
                     "PASS" if v["retrieval"]["gate"]["passed"] else "FAIL",
                     "PASS" if v["generation"]["gate"]["passed"] else "FAIL"])
    out.append("## Per-variant gates")
    out.append(md_table(["variant", "n", "hit@k", "recall@k", "MRR", "NDCG@k", "faith", "cite-P", "correct", "corr-F1", "over-abst",
                         "abst-P", "abst-R", "lie-rate", "inj-follow", "retrieval", "generation"], rows))
    out.append("")
    out.append("## Failed checks")
    fails = []
    for v in result["variants"]:
        for gate in ("retrieval", "generation"):
            for c in v[gate]["gate"]["checks"]:
                if not c["ok"]:
                    bound = ("<= " if c["metric"] in ("confident_lie_rate", "injection_follow_rate", "over_abstain_rate") else ">= ") + _f(c["bound"])
                    fails.append([v["variant"], gate, c["metric"], _f(c["value"]), bound])
    out.append(md_table(["variant", "gate", "metric", "value", "required"], fails) if fails else "none")
    out.append("")
    out.append("## Retrieval misses")
    misses = [[v["variant"], m["id"], m["why"], ", ".join(m["top_k"][:3])] for v in result["variants"]
              for m in v["retrieval"]["misses"]]
    out.append(md_table(["variant", "query", "why", "top-3"], misses) if misses else "none")
    out.append("")
    out.append("## Top failing generation examples")
    ex = []
    for v in result["variants"]:
        bad = [r for r in v["generation"]["rows"] if r["confident_lie"] or r["injection_followed"]
               or (r["faithfulness"] is not None and r["faithfulness"] < 1.0)
               or (r["contains_gold"] is not None and r["contains_gold"] < 1.0)
               or (r["should_abstain"] is False and r["abstained"])]
        for r in bad[:top_failures]:
            why = ("confident lie" if r["confident_lie"] else "followed injection" if r["injection_followed"]
                   else "unsupported sentence" if (r["faithfulness"] or 1.0) < 1.0
                   else "wrong answer" if r["contains_gold"] is not None else "over-abstained")
            ex.append([v["variant"], r["id"].split("/")[0], why, r["output"][:90].replace("|", "/")])
    out.append(md_table(["variant", "query", "failure", "model output"], ex) if ex else "none")
    return "\n".join(out)


def to_json(result: dict[str, Any]) -> dict[str, Any]:
    """Compact JSON: drop per-row retrieval tables but keep generation rows (they hold model outputs)."""
    slim = {**result, "variants": []}
    for v in result["variants"]:
        slim["variants"].append({"variant": v["variant"], "n": v["n"],
                                 "retrieval": {k: x for k, x in v["retrieval"].items() if k != "rows"},
                                 "generation": v["generation"]})
    return slim
