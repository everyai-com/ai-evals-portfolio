"""Self-contained dashboard.html: inline SVG scatter per tenant + global, frontier line, CI error
bars, a vanilla-JS price-sensitivity slider that re-draws from the embedded JSON, and tables.
No external resources; light/dark via CSS variables and prefers-color-scheme.

The Python renderer draws the initial SVG children (so the page is complete without JS); the JS
redraws them from the same numbers whenever the slider moves.
"""
from __future__ import annotations

import html
import json
import math
from typing import Any

from projects.p10_pareto_dashboard.core import GLOBAL, pareto_frontier

W, H = 640, 360
M = {"l": 64, "r": 20, "t": 16, "b": 48}
PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2", "#db2777", "#65a30d"]


def _points(block: dict[str, Any], scale: float) -> list[dict[str, Any]]:
    pts = [{"policy": p["policy"], "cost": p["mean_cost_one"] + p["mean_cost_two"] * scale,
            "succ": p["success_rate"], "lo": p["ci_lo"], "hi": p["ci_hi"], "n": p["n"]} for p in block["policies"]]
    front = set(pareto_frontier([(p["cost"], p["succ"]) for p in pts]))
    for i, p in enumerate(pts):
        p["front"] = i in front
        if p["front"]:
            p["target"], p["save"], p["savePct"] = None, 0.0, 0.0
        else:
            t = min((pts[j] for j in front if pts[j]["succ"] >= p["succ"]), key=lambda q: q["cost"])
            p["target"], p["save"] = t["policy"], p["cost"] - t["cost"]
            p["savePct"] = p["save"] / p["cost"] if p["cost"] > 0 else 0.0
    return pts


def svg_children(pts: list[dict[str, Any]], colors: dict[str, str]) -> str:
    """Mirror of the JS `draw()`; returns the inner markup of the chart <svg>."""
    costs = [max(p["cost"], 1e-9) for p in pts]
    x0, x1 = math.log10(min(costs) / 1.4), math.log10(max(costs) * 1.4)
    y0 = max(0.0, math.floor((min(p["lo"] for p in pts) - 0.05) * 10) / 10)
    y1 = 1.0
    iw, ih = W - M["l"] - M["r"], H - M["t"] - M["b"]
    sx = lambda c: M["l"] + (math.log10(max(c, 1e-9)) - x0) / (x1 - x0) * iw
    sy = lambda s: M["t"] + (y1 - s) / (y1 - y0) * ih
    s = [f'<rect x="{M["l"]}" y="{M["t"]}" width="{iw}" height="{ih}" class="plot"/>']
    k = math.floor(x0)
    while k <= math.ceil(x1):
        for m in (1, 2, 5):
            v = m * 10 ** k
            lv = math.log10(v)
            if x0 <= lv <= x1:
                x = sx(v)
                s.append(f'<line x1="{x:.1f}" y1="{M["t"]}" x2="{x:.1f}" y2="{M["t"] + ih}" class="grid{" major" if m == 1 else ""}"/>')
                if m == 1:
                    s.append(f'<text x="{x:.1f}" y="{M["t"] + ih + 16}" class="tick" text-anchor="middle">${v:g}</text>')
        k += 1
    yt = y0
    while yt <= y1 + 1e-9:
        y = sy(yt)
        s.append(f'<line x1="{M["l"]}" y1="{y:.1f}" x2="{M["l"] + iw}" y2="{y:.1f}" class="grid"/>')
        s.append(f'<text x="{M["l"] - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{yt * 100:.0f}%</text>')
        yt = round(yt + 0.1, 10)
    s.append(f'<text x="{M["l"] + iw / 2:.1f}" y="{H - 8}" class="axis" text-anchor="middle">cost per task (USD, log scale)</text>')
    s.append(f'<text transform="translate(14,{M["t"] + ih / 2:.1f}) rotate(-90)" class="axis" text-anchor="middle">task success rate</text>')
    front = sorted((p for p in pts if p["front"]), key=lambda p: p["cost"])
    if len(front) > 1:
        s.append('<polyline class="frontier" points="' + " ".join(f"{sx(p['cost']):.1f},{sy(p['succ']):.1f}" for p in front) + '"/>')
    for p in pts:
        x, y, c = sx(p["cost"]), sy(p["succ"]), colors[p["policy"]]
        s.append(f'<line x1="{x:.1f}" y1="{sy(p["lo"]):.1f}" x2="{x:.1f}" y2="{sy(p["hi"]):.1f}" stroke="{c}" class="ci"/>')
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" stroke="{c}" fill="{c if p["front"] else "var(--bg)"}" class="pt">'
                 f'<title>{html.escape(p["policy"])}: success {p["succ"] * 100:.1f}% [{p["lo"] * 100:.1f}-{p["hi"] * 100:.1f}], '
                 f'cost ${p["cost"]:.6f}/task{"" if p["front"] else " (dominated)"}</title></circle>')
        s.append(f'<text x="{x + 9:.1f}" y="{y - 8 if p["front"] else y + 16:.1f}" class="label" fill="{c}">{html.escape(p["policy"])}</text>')
    return "".join(s)


def table_rows(pts: list[dict[str, Any]]) -> str:
    rows = []
    for p in sorted(pts, key=lambda p: p["cost"]):
        rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in [
            html.escape(p["policy"]), p["n"], f"{p['succ'] * 100:.1f}% ({p['lo'] * 100:.1f}-{p['hi'] * 100:.1f})",
            f"${p['cost']:.6f}", f"${p['cost'] / p['succ']:.6f}" if p["succ"] > 0 else "-",
            "yes" if p["front"] else "no", html.escape(p["target"] or "-"),
            f"${p['save']:.6f} ({p['savePct'] * 100:.0f}%)" if not p["front"] else "-"]) + "</tr>")
    return "".join(rows)


JS = r"""
(function () {
  const DATA = JSON.parse(document.getElementById('pareto-data').textContent);
  const PALETTE = __PALETTE__;
  const W = __W__, H = __H__, M = __M__;
  const colors = {};
  (DATA.tenants[DATA.tenant_order[0]].policies).forEach((p, i) => { colors[p.policy] = PALETTE[i % PALETTE.length]; });
  function esc(s) { return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
  function frontier(pts) {
    const idx = pts.map((p, i) => i).sort((a, b) => (pts[a].cost - pts[b].cost) || (pts[b].succ - pts[a].succ));
    let best = -Infinity; const out = [];
    for (const i of idx) { if (pts[i].succ > best) { out.push(i); best = pts[i].succ; } }
    return out;
  }
  function points(tenant, scale) {
    const pts = DATA.tenants[tenant].policies.map(p => ({ policy: p.policy, cost: p.mean_cost_one + p.mean_cost_two * scale,
      succ: p.success_rate, lo: p.ci_lo, hi: p.ci_hi, n: p.n }));
    const f = new Set(frontier(pts));
    pts.forEach((p, i) => {
      p.front = f.has(i);
      if (p.front) { p.target = null; p.save = 0; p.savePct = 0; return; }
      const cands = pts.filter((q, j) => f.has(j) && q.succ >= p.succ);
      const t = cands.reduce((a, b) => (a.cost <= b.cost ? a : b));
      p.target = t.policy; p.save = p.cost - t.cost; p.savePct = p.cost > 0 ? p.save / p.cost : 0;
    });
    return pts;
  }
  function draw(svg, pts) {
    const costs = pts.map(p => Math.max(p.cost, 1e-9));
    const x0 = Math.log10(Math.min(...costs) / 1.4), x1 = Math.log10(Math.max(...costs) * 1.4);
    const y0 = Math.max(0, Math.floor((Math.min(...pts.map(p => p.lo)) - 0.05) * 10) / 10), y1 = 1.0;
    const iw = W - M.l - M.r, ih = H - M.t - M.b;
    const sx = c => M.l + (Math.log10(Math.max(c, 1e-9)) - x0) / (x1 - x0) * iw;
    const sy = s => M.t + (y1 - s) / (y1 - y0) * ih;
    const f1 = v => v.toFixed(1);
    let s = `<rect x="${M.l}" y="${M.t}" width="${iw}" height="${ih}" class="plot"/>`;
    for (let k = Math.floor(x0); k <= Math.ceil(x1); k++) {
      for (const m of [1, 2, 5]) {
        const v = m * Math.pow(10, k), lv = Math.log10(v);
        if (lv < x0 || lv > x1) continue;
        const x = sx(v);
        s += `<line x1="${f1(x)}" y1="${M.t}" x2="${f1(x)}" y2="${M.t + ih}" class="grid${m === 1 ? ' major' : ''}"/>`;
        if (m === 1) s += `<text x="${f1(x)}" y="${M.t + ih + 16}" class="tick" text-anchor="middle">$${v}</text>`;
      }
    }
    for (let yt = y0; yt <= y1 + 1e-9; yt = Math.round((yt + 0.1) * 1e6) / 1e6) {
      const y = sy(yt);
      s += `<line x1="${M.l}" y1="${f1(y)}" x2="${M.l + iw}" y2="${f1(y)}" class="grid"/>`;
      s += `<text x="${M.l - 6}" y="${f1(y + 4)}" class="tick" text-anchor="end">${Math.round(yt * 100)}%</text>`;
    }
    s += `<text x="${f1(M.l + iw / 2)}" y="${H - 8}" class="axis" text-anchor="middle">cost per task (USD, log scale)</text>`;
    s += `<text transform="translate(14,${f1(M.t + ih / 2)}) rotate(-90)" class="axis" text-anchor="middle">task success rate</text>`;
    const front = pts.filter(p => p.front).sort((a, b) => a.cost - b.cost);
    if (front.length > 1) s += `<polyline class="frontier" points="${front.map(p => `${f1(sx(p.cost))},${f1(sy(p.succ))}`).join(' ')}"/>`;
    for (const p of pts) {
      const x = sx(p.cost), y = sy(p.succ), c = colors[p.policy];
      s += `<line x1="${f1(x)}" y1="${f1(sy(p.lo))}" x2="${f1(x)}" y2="${f1(sy(p.hi))}" stroke="${c}" class="ci"/>`;
      s += `<circle cx="${f1(x)}" cy="${f1(y)}" r="6" stroke="${c}" fill="${p.front ? c : 'var(--bg)'}" class="pt"><title>${esc(p.policy)}: success ${(p.succ * 100).toFixed(1)}% [${(p.lo * 100).toFixed(1)}-${(p.hi * 100).toFixed(1)}], cost $${p.cost.toFixed(6)}/task${p.front ? '' : ' (dominated)'}</title></circle>`;
      s += `<text x="${f1(x + 9)}" y="${f1(p.front ? y - 8 : y + 16)}" class="label" fill="${c}">${esc(p.policy)}</text>`;
    }
    svg.innerHTML = s;
  }
  function table(tbody, pts) {
    tbody.innerHTML = pts.slice().sort((a, b) => a.cost - b.cost).map(p => '<tr>' + [
      esc(p.policy), p.n, `${(p.succ * 100).toFixed(1)}% (${(p.lo * 100).toFixed(1)}-${(p.hi * 100).toFixed(1)})`,
      `$${p.cost.toFixed(6)}`, p.succ > 0 ? `$${(p.cost / p.succ).toFixed(6)}` : '-', p.front ? 'yes' : 'no',
      esc(p.target || '-'), p.front ? '-' : `$${p.save.toFixed(6)} (${Math.round(p.savePct * 100)}%)`
    ].map(c => `<td>${c}</td>`).join('') + '</tr>').join('');
  }
  function render(scale) {
    for (const t of DATA.tenant_order) {
      const sec = document.querySelector(`section[data-tenant="${CSS.escape(t)}"]`);
      const pts = points(t, scale);
      draw(sec.querySelector('svg'), pts);
      table(sec.querySelector('tbody'), pts);
      sec.querySelector('.frontier-list').textContent = pts.filter(p => p.front).sort((a, b) => a.cost - b.cost).map(p => p.policy).join(' -> ');
    }
    document.getElementById('scale-label').textContent = scale.toFixed(2) + 'x';
  }
  const slider = document.getElementById('scale');
  slider.addEventListener('input', () => render(parseFloat(slider.value)));
  render(parseFloat(slider.value));
})();
"""

CSS = """
:root { --bg: #ffffff; --fg: #111827; --muted: #6b7280; --grid: #e5e7eb; --grid-major: #d1d5db; --plot: #f9fafb;
        --frontier: #111827; --accent: #2563eb; --card: #ffffff; --border: #e5e7eb; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0f172a; --fg: #e5e7eb; --muted: #94a3b8; --grid: #1f2937; --grid-major: #334155; --plot: #111827;
          --frontier: #f1f5f9; --accent: #60a5fa; --card: #0b1220; --border: #1f2937; }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; background: var(--bg); color: var(--fg);
       font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
h1 { font-size: 22px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 0 0 8px; }
.sub { color: var(--muted); margin-bottom: 16px; }
.controls { display: flex; gap: 12px; align-items: center; padding: 12px 16px; border: 1px solid var(--border);
            border-radius: 10px; background: var(--card); margin-bottom: 20px; flex-wrap: wrap; }
.controls input[type=range] { width: 320px; accent-color: var(--accent); }
#scale-label { font-variant-numeric: tabular-nums; font-weight: 600; min-width: 56px; }
.grid-wrap { display: grid; grid-template-columns: repeat(auto-fit, minmax(640px, 1fr)); gap: 20px; }
section.panel { border: 1px solid var(--border); border-radius: 10px; background: var(--card); padding: 14px; }
svg.chart { width: 100%; height: auto; display: block; }
.plot { fill: var(--plot); stroke: none; }
.grid { stroke: var(--grid); stroke-width: 1; } .grid.major { stroke: var(--grid-major); }
.tick { fill: var(--muted); font-size: 11px; } .axis { fill: var(--muted); font-size: 12px; }
.label { font-size: 11px; } .ci { stroke-width: 2; opacity: 0.8; } .pt { stroke-width: 2; }
.frontier { fill: none; stroke: var(--frontier); stroke-width: 1.5; stroke-dasharray: 4 3; }
table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12.5px; }
th, td { text-align: left; padding: 5px 6px; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { color: var(--muted); font-weight: 600; }
.frontier-line { color: var(--muted); margin: 8px 0 0; font-size: 12.5px; }
.legend { color: var(--muted); font-size: 12px; margin-top: 6px; }
.sens { margin-top: 20px; }
"""


def render_dashboard(metrics: dict[str, Any]) -> str:
    order = metrics["tenant_order"]
    first = metrics["tenants"][order[0]]["policies"]
    colors = {p["policy"]: PALETTE[i % len(PALETTE)] for i, p in enumerate(first)}
    scale = float(metrics["price_scale"])
    panels = []
    for t in order:
        blk = metrics["tenants"][t]
        pts = _points(blk, scale)
        name = "All tenants (pooled)" if t == GLOBAL else t
        panels.append(f"""
<section class="panel" data-tenant="{html.escape(t, quote=True)}">
  <h2>{html.escape(name)} <span class="legend">({blk['n_tasks']} tasks per policy)</span></h2>
  <svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="cost vs success for {html.escape(name)}">{svg_children(pts, colors)}</svg>
  <p class="frontier-line">Frontier (cheapest first): <span class="frontier-list">{html.escape(' -> '.join(p['policy'] for p in sorted((p for p in pts if p['front']), key=lambda p: p['cost'])))}</span></p>
  <table><thead><tr><th>policy</th><th>n</th><th>success (95% CI)</th><th>cost/task</th><th>cost/success</th><th>frontier</th><th>move to</th><th>savings/task</th></tr></thead>
  <tbody>{table_rows(pts)}</tbody></table>
</section>""")
    sens = metrics.get("sensitivity", {})
    sens_html = ""
    if sens:
        scales = list(sens)
        rows = "".join("<tr><td>" + html.escape("All tenants (pooled)" if t == GLOBAL else t) + "</td>" +
                       "".join(f"<td>{html.escape(', '.join(sens[s].get(t, [])))}</td>" for s in scales) + "</tr>" for t in order)
        sens_html = (f'<section class="panel sens"><h2>Frontier vs System-Two price scale (precomputed)</h2>'
                     f'<table><thead><tr><th>tenant</th>{"".join(f"<th>{html.escape(s)}x</th>" for s in scales)}</tr></thead>'
                     f'<tbody>{rows}</tbody></table></section>')
    data_json = json.dumps(metrics, separators=(",", ":")).replace("</", "<\\/")
    js = (JS.replace("__PALETTE__", json.dumps(PALETTE)).replace("__W__", str(W)).replace("__H__", str(H))
          .replace("__M__", json.dumps(M)))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cost-Quality Pareto Dashboard</title><style>{CSS}</style></head>
<body>
<h1>Cost-Quality Pareto Dashboard</h1>
<div class="sub">Task success vs inference cost per routing policy. Filled points are on the Pareto frontier (no policy is both cheaper and better); hollow points are dominated. Error bars are Wilson 95% CIs. {metrics['n_rows']} eval rows.</div>
<div class="controls">
  <label for="scale">System-Two price scale</label>
  <input id="scale" type="range" min="0.05" max="1.5" step="0.05" value="{scale:g}">
  <span id="scale-label">{scale:.2f}x</span>
  <span class="legend">Drag to see how the frontier moves if the expensive model's price changes (re-computed in the browser from the embedded JSON).</span>
</div>
<div class="grid-wrap">{''.join(panels)}</div>
{sens_html}
<script id="pareto-data" type="application/json">{data_json}</script>
<script>{js}</script>
</body></html>
"""
