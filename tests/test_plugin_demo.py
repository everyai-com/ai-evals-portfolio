"""Run with: pytest tests/test_plugin_demo.py --eval-dataset tests/data/plugin_demo.jsonl"""


def tiny_model(expr: str) -> str:
    try:
        return str(eval(expr, {"__builtins__": {}}))
    except Exception:  # any bad expression is a wrong answer  # noqa: BLE001
        return "?"


def test_case(eval_case, eval_recorder):
    pred = tiny_model(eval_case["input"])
    eval_recorder.record(eval_case, pred == eval_case["expected"], pred=pred)
    assert pred == eval_case["expected"]
