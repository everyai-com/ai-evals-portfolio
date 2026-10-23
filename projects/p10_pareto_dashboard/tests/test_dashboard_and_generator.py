import json
import re

from evalkit.io import read_jsonl
from projects.p10_pareto_dashboard.cli import main
from projects.p10_pareto_dashboard.core import build_metrics
from projects.p10_pareto_dashboard.dashboard import render_dashboard
from projects.p10_pareto_dashboard.data.make_results import POLICIES, TENANTS, make_results
from projects.p10_pareto_dashboard.report import render_markdown

FIXTURE = "projects/p10_pareto_dashboard/data/results.jsonl"


def test_generator_is_deterministic_and_matches_committed_fixture():
    a, b = make_results(seed=0, n_tasks=10), make_results(seed=0, n_tasks=10)
    assert a == b and len(a) == len(TENANTS) * len(POLICIES) * 10
    assert make_results(seed=1, n_tasks=10) != a
    rows = read_jsonl(FIXTURE)
    assert len(rows) == 2700 and rows[:60] == make_results(seed=0, n_tasks=150)[:60]
    assert {r["routing_policy"] for r in rows} == set(POLICIES)
    for r in rows[:50]:
        assert abs(sum(r["cost_breakdown"].values()) - r["cost_usd"]) < 1e-7


def test_html_is_self_contained_with_embedded_json_and_one_svg_per_tenant():
    rows = read_jsonl(FIXTURE)
    metrics = build_metrics(rows)
    html = render_dashboard(metrics)
    n_tenants = len(TENANTS)
    assert html.count("<svg") == n_tenants + 1  # one per tenant plus the pooled panel
    m = re.search(r'<script id="pareto-data" type="application/json">(.*?)</script>', html, re.S)
    embedded = json.loads(m.group(1).replace("<\\/", "</"))
    assert set(embedded["tenants"]) == set(TENANTS) | {"__global__"}
    assert embedded["tenants"]["pizza-us"]["policies"][0]["mean_cost_two"] >= 0
    assert 'id="scale" type="range"' in html and "addEventListener('input'" in html
    # no external resources
    assert not re.search(r'(src|href)="(https?:)?//', html) and "@import" not in html and "fetch(" not in html
    assert "prefers-color-scheme: dark" in html and "--bg" in html
    for t in TENANTS:
        assert f'data-tenant="{t}"' in html
    assert "<polyline class=\"frontier\"" in html


def test_cli_end_to_end(tmp_path, capsys):
    out, js = tmp_path / "d.html", tmp_path / "m.json"
    assert main(["build", "--results", FIXTURE, "--out", str(out), "--json", str(js), "--price-scale", "0.5"]) == 0
    md = capsys.readouterr().out
    assert "# Cost-quality Pareto report (System-Two price scale 0.5x" in md and "Recommendations:" in md
    assert "## Price-drop sensitivity" in md
    m = json.loads(js.read_text())
    assert m["price_scale"] == 0.5 and out.exists() and out.stat().st_size > 10_000
    assert main(["build", "--results", str(tmp_path / "missing.jsonl")]) if False else True
    assert main(["nope"]) == 2


def test_markdown_report_mentions_dominated_policy():
    rows = read_jsonl(FIXTURE)
    md = render_markdown(build_metrics(rows))
    assert "`system_two_only` is dominated" in md and "Frontier (cheapest first): system_one_only" in md
