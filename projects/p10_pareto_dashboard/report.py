"""Markdown report per tenant: frontier table, marginal costs, recommendations."""
from __future__ import annotations

from typing import Any

from evalkit.report import md_table, pct
from projects.p10_pareto_dashboard.core import GLOBAL


def tenant_name(t: str) -> str:
    return "All tenants (pooled)" if t == GLOBAL else t


def render_markdown(metrics: dict[str, Any]) -> str:
    out = [f"# Cost-quality Pareto report (System-Two price scale {metrics['price_scale']:g}x, "
           f"{metrics['n_rows']} eval rows)", ""]
    for t in metrics["tenant_order"]:
        blk = metrics["tenants"][t]
        pols = sorted(blk["policies"], key=lambda p: p["mean_cost"])
        out += [f"## {tenant_name(t)}", "",
                md_table(["policy", "n", "success (95% CI)", "cost/task", "cost/success", "p95 ms", "tokens", "frontier", "move to", "save/task", "save %"],
                         [[p["policy"], p["n"], f"{pct(p['success_rate'])} ({pct(p['ci_lo'])}-{pct(p['ci_hi'])})",
                           f"${p['mean_cost']:.6f}", f"${p['cost_per_success']:.6f}" if p["cost_per_success"] is not None else "-",
                           f"{p['p95_latency_ms']:.0f}", f"{p['mean_tokens']:.0f}", "yes" if p["on_frontier"] else "no",
                           p["target_policy"] or "-", f"${p['savings_per_task']:.6f}" if not p["on_frontier"] else "-",
                           pct(p["savings_pct"]) if not p["on_frontier"] else "-"] for p in pols]),
                "", f"Frontier (cheapest first): {' -> '.join(blk['frontier'])}", ""]
        if blk["marginal"]:
            out += ["Marginal cost between adjacent frontier points:", "",
                    md_table(["from", "to", "+success pts", "+cost/task", "cost per +1 success pt"],
                             [[m["from"], m["to"], f"{m['delta_success_pts']:+.1f}", f"${m['delta_cost']:+.6f}",
                               f"${m['cost_per_success_point']:.6f}" if m["cost_per_success_point"] is not None else "-"]
                              for m in blk["marginal"]]), ""]
        recs = [p for p in pols if not p["on_frontier"] and p["savings_per_task"] > 0]
        out.append("Recommendations:")
        if recs:
            for p in sorted(recs, key=lambda p: -p["savings_per_task"]):
                n = blk["n_tasks"]
                out.append(f"- `{p['policy']}` is dominated: switch to `{p['target_policy']}` for "
                           f"{pct(p['savings_pct'])} lower cost (${p['savings_per_task'] * 1000:.2f} per 1k tasks) "
                           f"and {p['success_gain'] * 100:+.1f} success points (n={n} tasks).")
        else:
            out.append("- every policy is on the frontier; pick by success target vs budget.")
        out.append("")
    sens = metrics.get("sensitivity", {})
    if sens:
        scales = list(sens)
        out += ["## Price-drop sensitivity (frontier per System-Two price scale)", "",
                md_table(["tenant"] + [f"{s}x" for s in scales],
                         [[tenant_name(t)] + [", ".join(sens[s].get(t, [])) for s in scales]
                          for t in metrics["tenant_order"]]), ""]
    return "\n".join(out)
