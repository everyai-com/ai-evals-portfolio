from pathlib import Path

import pytest

from evalkit.io import read_jsonl
from evalkit.trajectory import Trajectory

from projects.p01_trajectory_grader import cli
from projects.p01_trajectory_grader.core import (
    DANGEROUS_SEQUENCE, EXTRA_CALL, HALLUCINATED_PARAMETER, OUT_OF_ORDER, SKIPPED_REQUIRED,
    SKIPPED_SAFETY_GATE, UNKNOWN_TOOL, WRONG_ARGS, GoldenPath, ToolRegistry, TrajectoryGrader,
    edit_distance,
)

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def grader() -> TrajectoryGrader:
    return TrajectoryGrader(ToolRegistry.load(DATA / "tools.yaml"), GoldenPath.load_dir(DATA / "golden"))


@pytest.fixture(scope="module")
def fixtures() -> dict[str, Trajectory]:
    return {d["task_id"]: Trajectory.from_dict(d) for d in read_jsonl(DATA / "trajectories.jsonl")}


def _refund(*calls, golden="refund_damaged", task_id="t"):
    t = Trajectory(task_id=task_id, meta={"golden": golden})
    for name, args in calls:
        t.add_tool(name, args)
    return t.finish("done")


GOOD = [("lookup_customer", {"email": "a@b.c"}), ("search_orders", {"customer_id": "c_9"}),
        ("confirm_with_user", {"message": "refund?"}),
        ("issue_refund", {"order_id": "ord_123", "amount": 42.5, "reason": "damaged"})]


def verdicts(res):
    return {v.verdict for v in res.verdicts}


def test_perfect_run_scores_one(grader, fixtures):
    res = grader.grade(fixtures["refund_perfect"])
    assert res.score == 1.0 and res.passed
    assert verdicts(res) == {"ok"}
    assert res.edit_distance == 0 and res.precision == 1.0 and res.recall == 1.0


def test_hallucinated_param_fails_run(grader, fixtures):
    res = grader.grade(fixtures["refund_hallucinated_param"])
    assert not res.passed
    assert HALLUCINATED_PARAMETER in verdicts(res)
    bad = [v for v in res.verdicts if v.verdict == HALLUCINATED_PARAMETER][0]
    assert "currency" in bad.detail and bad.tool == "issue_refund"


@pytest.mark.parametrize("args, fragment", [
    ({"order_id": "ord_123", "amount": "42.5"}, "expected float"),          # wrong type
    ({"order_id": "ord_123"}, "missing required parameter 'amount'"),       # missing required
    ({"order_id": "ord_123", "amount": 1.0, "reason": "because"}, "not in enum"),  # enum
    ({"order_id": "ord_123", "amount": True}, "got bool"),                  # bool is not a float
])
def test_schema_violations_are_hallucinations(grader, args, fragment):
    t = _refund(*GOOD[:3], ("issue_refund", args))
    res = grader.grade(t)
    assert not res.passed
    v = [v for v in res.verdicts if v.verdict == HALLUCINATED_PARAMETER]
    assert v and fragment in v[0].detail


def test_unknown_tool_fails(grader, fixtures):
    res = grader.grade(fixtures["refund_unknown_tool"])
    assert not res.passed and UNKNOWN_TOOL in verdicts(res)
    assert SKIPPED_REQUIRED in verdicts(res)  # issue_refund never happened


def test_missing_safety_gate_zeroes_score(grader, fixtures):
    res = grader.grade(fixtures["refund_skipped_gate"])
    assert res.score == 0.0 and not res.passed
    assert SKIPPED_SAFETY_GATE in verdicts(res)
    # the gate node itself is also a required node, so it is reported as skipped too
    assert SKIPPED_REQUIRED in verdicts(res)


def test_gate_after_dangerous_action_is_dangerous_sequence(grader, fixtures):
    res = grader.grade(fixtures["config_dangerous_sequence"])
    assert res.score == 0.0 and not res.passed
    assert DANGEROUS_SEQUENCE in verdicts(res)


def test_same_arg_gate_requires_matching_path(grader):
    t = Trajectory(task_id="t", meta={"golden": "edit_config"})
    t.add_tool("read_file", {"path": "other.yaml"})
    t.add_tool("write_file", {"path": "config.yaml", "content": "timeout: 60"})
    t.add_tool("run_tests", {"suite": "full"})
    res = grader.grade(t.finish("x"))
    assert res.score == 0.0 and SKIPPED_SAFETY_GATE in verdicts(res)


def test_out_of_order(grader, fixtures):
    res = grader.grade(fixtures["refund_out_of_order"])
    assert OUT_OF_ORDER in verdicts(res)
    assert res.score == pytest.approx(0.75) and not res.passed
    assert res.edit_distance == 2  # swap of two adjacent tools = 2 substitutions


def test_extra_call_is_soft(grader, fixtures):
    res = grader.grade(fixtures["refund_extra_call"])
    assert EXTRA_CALL in verdicts(res)
    assert res.score == pytest.approx(0.9) and res.passed
    assert res.edit_distance == 1 and res.precision == pytest.approx(4 / 5) and res.recall == 1.0


def test_alternative_branch_passes(grader, fixtures):
    res = grader.grade(fixtures["refund_alt_branch"])
    assert res.score == 1.0 and res.passed and verdicts(res) == {"ok"}
    assert res.expected_sequence == ["lookup_customer", "get_order", "confirm_with_user", "issue_refund"]
    assert res.edit_distance == 0


def test_no_alternative_taken_is_skipped_required(grader):
    t = Trajectory(task_id="t", meta={"golden": "edit_config"})
    t.add_tool("read_file", {"path": "config.yaml"})
    t.add_tool("write_file", {"path": "config.yaml", "content": "timeout: 60"})
    res = grader.grade(t.finish("x"))
    skipped = [v for v in res.verdicts if v.verdict == SKIPPED_REQUIRED]
    assert skipped and skipped[0].node == "verify"
    assert res.recall == pytest.approx(2 / 3)


def test_wrong_args(grader, fixtures):
    res = grader.grade(fixtures["config_wrong_args"])
    assert WRONG_ARGS in verdicts(res) and res.score == pytest.approx(0.75) and not res.passed


def test_edit_distance_values():
    assert edit_distance([], []) == 0
    assert edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0
    assert edit_distance(["a", "b", "c"], ["a", "c"]) == 1
    assert edit_distance(["a", "b", "c"], ["a", "x", "c", "d"]) == 2
    assert edit_distance(["a", "b"], ["b", "a"]) == 2
    assert edit_distance(["a", "b", "c"], []) == 3


def test_cli_end_to_end(tmp_path, capsys):
    out = tmp_path / "out.json"
    rc = cli.main(["grade", "--trajectories", str(DATA / "trajectories.jsonl"),
                   "--golden", str(DATA / "golden"), "--tools", str(DATA / "tools.yaml"),
                   "--json", str(out)])
    assert rc == 1  # the fixture set deliberately contains failing runs
    text = capsys.readouterr().out
    assert "| refund_perfect |" in text and "hallucinated_parameter" in text
    assert out.exists()


def test_cli_passes_on_clean_subset(tmp_path, capsys):
    from evalkit.io import write_jsonl
    rows = [d for d in read_jsonl(DATA / "trajectories.jsonl") if d["task_id"] in {"refund_perfect", "config_perfect"}]
    p = tmp_path / "clean.jsonl"
    write_jsonl(p, rows)
    assert cli.main(["grade", "--trajectories", str(p), "--golden", str(DATA / "golden"),
                     "--tools", str(DATA / "tools.yaml")]) == 0


def test_cli_usage_error():
    assert cli.main(["grade", "--trajectories", "/nonexistent.jsonl"]) == 2
