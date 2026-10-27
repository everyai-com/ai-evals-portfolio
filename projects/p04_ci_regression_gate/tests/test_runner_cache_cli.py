"""Runner, cache, cost budget, CLI exit codes, GitHub outputs, comment posting, workflow YAML."""
from __future__ import annotations

import json

import pytest
import yaml

from projects.p04_ci_regression_gate import cli
from projects.p04_ci_regression_gate.core import BudgetExceeded, EvalCache, EvalConfig, EvalRunner
from projects.p04_ci_regression_gate.demo_sut import VARIANTS, make_sut
from projects.p04_ci_regression_gate.post_comment import (
    COMMENT_MARKER,
    FakeGitHub,
    post_or_update_comment,
)
from projects.p04_ci_regression_gate.tests.conftest import BASELINE, CONFIG, HERE, run_variant

ROOT = HERE.parent.parent


def test_results_schema(results):
    r = results["pr_good"]
    row = r["examples"][0]
    assert {"id", "success", "latency_ms", "cost_usd", "tags", "input", "expected", "actual", "seed"} <= set(row)
    assert r["aggregates"]["n_items"] == 300 and r["aggregates"]["n_calls"] == 600
    assert 0.0 <= r["aggregates"]["task_success"] <= 1.0
    assert r["aggregates"]["cost_total_usd"] > 0


def test_cache_hit_rate_is_100_percent_on_second_run(tmp_path):
    first = run_variant("pr_good", cache_dir=tmp_path / "cache")
    assert first["cache"]["hit_rate"] == 0.0
    second = run_variant("pr_good", cache_dir=tmp_path / "cache")
    assert second["cache"]["hit_rate"] == 1.0 and second["cache"]["hits"] == 600
    assert second["aggregates"] == first["aggregates"]
    # a different variant must not hit main's cache entries
    other = run_variant("pr_regressed", cache_dir=tmp_path / "cache")
    assert other["cache"]["hit_rate"] == 0.0


def test_cache_key_depends_on_every_input():
    k = EvalCache.key("p", "m", {"seed": 0}, {"id": "a"})
    assert k != EvalCache.key("p2", "m", {"seed": 0}, {"id": "a"})
    assert k != EvalCache.key("p", "m2", {"seed": 0}, {"id": "a"})
    assert k != EvalCache.key("p", "m", {"seed": 1}, {"id": "a"})
    assert k != EvalCache.key("p", "m", {"seed": 0}, {"id": "b"})
    assert k == EvalCache.key("p", "m", {"seed": 0}, {"id": "a"})


def test_cost_budget_abort(config, tmp_path):
    cfg = EvalConfig.load(CONFIG)
    cfg.max_cost_usd = 0.0001                          # projected 600 * est_per_call > budget
    ex = cfg.load_examples()
    runner = EvalRunner(cfg, make_sut("main", ex), variant="main")
    with pytest.raises(BudgetExceeded):
        runner.run(ex)
    # cached runs cost nothing to project
    cfg2 = EvalConfig.load(CONFIG)
    run_variant("main", cache_dir=tmp_path / "c", config=cfg2)
    cfg2.max_cost_usd = 0.0001
    cached = EvalRunner(cfg2, make_sut("main", ex), cache=EvalCache(tmp_path / "c"), variant="main")
    assert cached.project_cost(ex) == (0, 0.0)


def test_cli_run_gate_exit_codes_and_outputs(tmp_path, monkeypatch):
    out = tmp_path / "results.json"
    assert cli.main(["run", "--config", str(CONFIG), "--variant", "pr_regressed", "--out", str(out),
                     "--cache-dir", str(tmp_path / "cache")]) == 0
    gh_out, gh_sum = tmp_path / "out.txt", tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh_out))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(gh_sum))
    code = cli.main(["gate", "--results", str(out), "--baseline", str(BASELINE), "--config", str(CONFIG),
                     "--comment", str(tmp_path / "c.md"), "--state", str(tmp_path / "s.json"),
                     "--json", str(tmp_path / "d.json")])
    assert code == 1
    assert "blocked=true" in gh_out.read_text()
    assert "BLOCKED" in gh_sum.read_text() and "BLOCKED" in (tmp_path / "c.md").read_text()
    decision = json.loads((tmp_path / "d.json").read_text())
    assert decision["blocked"] and decision["outputs"]["blocked"] == "true"
    # pr_good passes with exit 0
    good = tmp_path / "good.json"
    cli.main(["run", "--config", str(CONFIG), "--variant", "pr_good", "--out", str(good), "--no-cache"])
    assert cli.main(["gate", "--results", str(good), "--baseline", str(BASELINE), "--config", str(CONFIG),
                     "--comment", str(tmp_path / "g.md"), "--state", str(tmp_path / "s2.json")]) == 0


def test_cli_cost_budget_exit_2(tmp_path):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["cost"]["max_cost_usd"] = 0.00001
    cfg["datasets"] = [{"path": str(HERE / "data" / "tasks.jsonl")}]
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert cli.main(["run", "--config", str(p), "--variant", "main", "--out", str(tmp_path / "r.json"),
                     "--no-cache"]) == 2
    assert cli.main(["bogus"]) == 2


def test_comment_posting_is_idempotent_with_fake_transport():
    gh = FakeGitHub()
    gh.comments.append({"id": 99, "body": "unrelated human comment"})
    r1 = post_or_update_comment("o/r", 7, "## Eval gate: PASSED", "tok", transport=gh)
    assert r1["action"] == "created"
    r2 = post_or_update_comment("o/r", 7, "## Eval gate: BLOCKED", "tok", transport=gh)
    assert r2["action"] == "updated" and r2["id"] == r1["id"]
    assert len(gh.comments) == 2                                   # no duplicate
    mine = [c for c in gh.comments if COMMENT_MARKER in c["body"]]
    assert len(mine) == 1 and "BLOCKED" in mine[0]["body"]
    assert [m for m, _ in gh.requests] == ["GET", "POST", "GET", "PATCH"]
    assert "/repos/o/r/issues/7/comments" in gh.requests[1][1]


def test_workflow_and_action_yaml(config):
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "eval-gate.yml").read_text())
    triggers = wf.get("on", wf.get(True))                          # PyYAML parses bare `on` as True
    assert "pull_request" in triggers and triggers["push"]["branches"] == ["main"]
    assert set(wf["jobs"]) == {"gate", "update-baseline"}
    gate_steps = wf["jobs"]["gate"]["steps"]
    uses = [s.get("uses", "") for s in gate_steps]
    assert any(u.startswith("astral-sh/setup-uv") for u in uses)
    assert any(u.startswith("actions/cache") and s["with"]["path"] == ".eval-cache"
               for u, s in zip(uses, gate_steps))
    assert any(u.startswith("actions/upload-artifact") for u in uses)
    runs = "\n".join(s.get("run", "") for s in gate_steps)
    assert "projects.p04_ci_regression_gate.cli run" in runs
    assert "baseline/main.json" in runs and "post_comment" in runs
    assert "$PROJECT/data/eval_config.yaml" in runs and wf["env"]["PROJECT"] == "projects/p04_ci_regression_gate"
    assert (ROOT / "projects/p04_ci_regression_gate/baseline/main.json").exists()
    assert (ROOT / "projects/p04_ci_regression_gate/data/eval_config.yaml").exists()
    bl = "\n".join(s.get("run", "") for s in wf["jobs"]["update-baseline"]["steps"])
    assert "cli baseline" in bl
    action = yaml.safe_load((ROOT / ".github" / "actions" / "eval-gate" / "action.yml").read_text())
    assert action["runs"]["using"] == "composite"
    assert {"config", "baseline", "max-drop", "token"} <= set(action["inputs"])
    assert "blocked" in action["outputs"]


def test_variants_are_deterministic():
    a, b = run_variant("pr_flaky"), run_variant("pr_flaky")
    assert a["examples"] == b["examples"]
    assert set(VARIANTS) >= {"main", "pr_good", "pr_regressed", "pr_slow", "pr_flaky"}
