"""Verdict thresholds, plan line accounting, markdown/JSON output, fake-git velocity, CLI."""
from __future__ import annotations

import json
from pathlib import Path

from projects.p23_bloat_detector.budget import (
    RefactorBudget,
    load_thresholds,
    render_plan,
    render_scan,
)
from projects.p23_bloat_detector.cli import main
from projects.p23_bloat_detector.core import Analyzer
from projects.p23_bloat_detector.velocity import (
    compute_velocity,
    git_velocity,
    max_in_window,
    parse_log,
)

DATA = Path(__file__).resolve().parent.parent / "data"
BLOATED = Analyzer().scan(DATA / "bloated_repo")
CLEAN = Analyzer().scan(DATA / "clean_repo")
BUDGET = RefactorBudget(load_thresholds())


def test_thresholds_book_the_refactor_on_bloated_not_on_clean():
    bp, cp = BUDGET.plan(BLOATED), BUDGET.plan(CLEAN)
    assert bp.book_the_refactor is True
    assert {b.metric for b in bp.breaches} >= {"duplicate_pct", "max_concept_cluster", "max_wrapper_depth"}
    assert cp.book_the_refactor is False and cp.breaches == []
    assert cp.actions == [] and cp.projected_lines == cp.lines_before


def test_each_rule_can_fire_alone():
    assert RefactorBudget({"comment_ratio_max": 0.0}).breaches(BLOATED)[0].metric == "comment_ratio"
    assert RefactorBudget({"diary_lines_max": 0}).breaches(BLOATED)[0].metric == "diary_lines"
    assert RefactorBudget({"dead_code_max": 0}).breaches(BLOATED)[0].metric == "dead_code_candidates"
    assert RefactorBudget({"near_duplicates_max": 0}).breaches(BLOATED)[0].metric == "near_duplicates"
    assert RefactorBudget({}).breaches(BLOATED) == []  # null / missing rules never fire


def test_refactor_plan_line_estimates_sum_correctly_and_never_double_count():
    plan = BUDGET.plan(BLOATED)
    assert plan.projected_lines == plan.lines_before - sum(a.lines_removed for a in plan.actions)
    assert plan.lines_before == BLOATED.source_lines
    kinds = {a.kind for a in plan.actions}
    assert {"merge_duplicates", "merge_near_duplicates", "collapse_wrappers", "unify_concept", "delete_diary", "delete_dead_code"} <= kinds
    # ranked by lines removed, ranks are 1..n
    assert [a.rank for a in plan.actions] == list(range(1, len(plan.actions) + 1))
    assert all(a.lines_removed >= b.lines_removed for a, b in zip(plan.actions, plan.actions[1:]))
    # the duplicate cluster and the concept cluster share members: only one action removes clean_sku
    sku_actions = [a for a in plan.actions if "clean_sku" in a.description]
    assert len(sku_actions) == 1 and sku_actions[0].kind == "merge_duplicates"
    assert sku_actions[0].lines_removed == BLOATED.duplicate_clusters[0].duplicated_lines
    concept = next(a for a in plan.actions if a.kind == "unify_concept")
    assert "keep `shopkit/core/dates.py:parse_date`" in concept.description
    dead = next(a for a in plan.actions if a.kind == "delete_dead_code")
    assert dead.lines_removed == BLOATED.dead_code[0].n_lines
    diary_actions = [a for a in plan.actions if a.kind == "delete_diary"]
    # diary lines inside functions another action already deletes are not counted twice
    assert 0 < sum(a.lines_removed for a in diary_actions) <= BLOATED.diary_lines
    assert diary_actions[0].files == ["shopkit/utils/helpers.py"] and diary_actions[0].lines_removed == 9


def test_render_and_json_roundtrip(tmp_path):
    plan = BUDGET.plan(BLOATED)
    md = render_scan(BLOATED, plan)
    assert "book_the_refactor: true" in md and "### Wrapper chains" in md and "parse:date" in md
    assert "book_the_refactor: false" in render_plan(BUDGET.plan(CLEAN))
    blob = json.dumps({**BLOATED.to_dict(), "plan": plan.to_dict()})
    back = json.loads(blob)
    assert back["summary"]["files"] == BLOATED.n_files and back["plan"]["book_the_refactor"] is True


FAKE_LOG = """\
aaa|1700000000
10\t2\tsrc/a.py
5\t0\tsrc/b.py

bbb|1700003600
-\t-\tbin/blob.png
100\t20\tsrc/a.py

ccc|1700090000
3\t300\tsrc/c.py

ddd|1700600000
1\t1\tREADME.md
"""


def _fake_runner(args, cwd):
    if args[0] == "rev-parse":
        return "true\n"
    assert args[0] == "log" and "--numstat" in args
    return FAKE_LOG


def test_fake_git_runner_velocity_numbers():
    commits = parse_log(FAKE_LOG)
    assert [c.sha for c in commits] == ["aaa", "bbb", "ccc", "ddd"]
    assert (commits[0].added, commits[0].deleted) == (15, 2)
    assert (commits[1].added, commits[1].deleted) == (100, 20)   # binary '-' lines ignored
    v = git_velocity("/any", days=30, runner=_fake_runner)
    assert v is not None and v.n_commits == 4
    assert v.max_commits_48h == 3           # aaa, bbb, ccc are within 90 000 s; ddd is 6 days later
    assert v.lines_added == 119 and v.lines_deleted == 323
    assert v.add_delete_ratio == round(119 / 323, 3)
    assert sum(v.commits_per_day.values()) == 4 and v.busiest_days[0][1] == 2
    assert max_in_window([], 10) == (0, None)
    assert git_velocity("/any", runner=lambda a, c: "false\n") is None
    # the velocity rule only fires with a velocity
    assert RefactorBudget({"commits_48h_max": 3}).breaches(BLOATED, v)[0].metric == "max_commits_48h"
    assert RefactorBudget({"commits_48h_max": 3}).breaches(BLOATED, None) == []


def test_synthetic_157_commit_days():
    ts = [1700000000 + i * 500 for i in range(157)] + [1700000000 + 86400 + i * 500 for i in range(157)]
    v = compute_velocity([type("C", (), {"sha": str(i), "ts": t, "added": 10, "deleted": 1})() for i, t in enumerate(ts)], 7)
    assert v.max_commits_48h == 314 and v.busiest_days[0][1] >= 157 and v.add_delete_ratio == 10.0


def test_cli_scan_exit_codes_and_outputs(tmp_path, capsys):
    out_json, out_plan = tmp_path / "o.json", tmp_path / "plan.md"
    assert main(["scan", "--dir", str(DATA / "bloated_repo"), "--json", str(out_json), "--plan", str(out_plan)]) == 1
    assert "## refactor plan" in capsys.readouterr().out
    assert json.loads(out_json.read_text())["plan"]["book_the_refactor"] is True
    assert "book_the_refactor: true" in out_plan.read_text()
    assert main(["scan", "--dir", str(DATA / "clean_repo")]) == 0
    assert main(["scan", "--dir", str(tmp_path / "missing")]) == 2
    assert main(["bogus"]) == 2
    assert main(["velocity", "--dir", str(tmp_path)]) == 2  # not a git repo
