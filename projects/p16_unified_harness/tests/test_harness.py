"""Tests prove the README claims: fallbacks compute correct metrics, names normalise, disagreements
surface, one row per dataset x framework, unavailable backends do not crash, promptfoo config is YAML."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from evalkit.io import read_jsonl, write_jsonl
from projects.p16_unified_harness import cli
from projects.p16_unified_harness.adapters import (
    ADAPTERS,
    DeepEvalAdapter,
    Framework,
    NativeAdapter,
    PromptfooAdapter,
    RagasAdapter,
    UnifiedResult,
)
from projects.p16_unified_harness.adapters.promptfoo_adapter import generate_promptfoo_config
from projects.p16_unified_harness.core import (
    export_jsonl,
    load_config,
    normalise_metrics,
    render_markdown,
    run_harness,
)
from projects.p16_unified_harness.sut import get_sut
from projects.p16_unified_harness.textmetrics import context_precision, faithfulness, token_f1

DATA = Path(__file__).resolve().parents[1] / "data"

HAND = [
    {"id": "h1", "input": "What is the capital of France?", "expected": "Paris",
     "context": ["Paris is the capital of France.", "Lyon is a city in France."], "tags": ["t"]},
    {"id": "h2", "input": "How many legs does a spider have?", "expected": "8",
     "context": ["Spiders have eight legs."], "tags": ["t"]},
    {"id": "h3", "input": "Which planet is the Red Planet?", "expected": "Mars",
     "context": ["Mars is the Red Planet."], "tags": ["t"],
     "asserts": [{"type": "equals", "value": "Mars."}, {"type": "regex", "value": "^M"}]},
]
OUTPUTS = {"h1": "Paris is the capital of France.", "h2": "A spider has eight legs.", "h3": "Mars."}


def scripted_sut(ex):
    from evalkit.llm import LLMResponse
    t = OUTPUTS[ex["id"]]
    return t, LLMResponse(text=t, model="mock", input_tokens=1, output_tokens=1, latency_ms=0, cost_usd=0.001)


@pytest.fixture
def hand_path(tmp_path):
    p = tmp_path / "hand.jsonl"
    write_jsonl(p, HAND)
    return str(p)


@pytest.fixture
def cfg():
    c = load_config(DATA / "harness.yaml")
    c["sut_fn"] = scripted_sut
    return c


# ---------------------------------------------------------------- fallbacks compute correct metrics

def test_promptfoo_fallback_assertions(hand_path, cfg):
    cfg["options"]["promptfoo"]["force_fallback"] = True
    r = PromptfooAdapter().run(hand_path, cfg)
    assert isinstance(r, UnifiedResult) and r.backend == "fallback"
    by = {p["id"]: p for p in r.per_example}
    assert by["h1"]["passed"] is True                       # icontains Paris + rubric covered
    assert by["h2"]["passed"] is False                      # "8" not in "eight legs"
    assert by["h3"]["passed"] is True and [a["type"] for a in by["h3"]["details"]["asserts"]] == ["equals", "regex"]
    assert r.metrics["pass_rate"] == pytest.approx(2 / 3)
    assert r.cost_usd == pytest.approx(0.003, abs=1e-6)     # SUT cost accounted


def test_promptfoo_javascript_assert_warns(tmp_path, cfg):
    p = tmp_path / "js.jsonl"
    write_jsonl(p, [{**HAND[0], "asserts": [{"type": "javascript", "value": "output.length > 0"}]}])
    r = PromptfooAdapter().run(str(p), cfg)
    assert r.warnings and "javascript" in r.warnings[0]
    assert r.per_example[0]["passed"] is False               # nothing evaluable => not a pass


def test_deepeval_fallback_rubric_and_relevancy(hand_path, cfg):
    r = DeepEvalAdapter().run(hand_path, cfg)
    assert r.backend == "fallback"
    by = {p["id"]: p for p in r.per_example}
    assert by["h1"]["details"]["geval_correctness"] == 1.0
    assert by["h2"]["details"]["geval_correctness"] == 0.0
    assert by["h1"]["details"]["answer_relevancy"] == 1.0                    # {capital, france} both echoed
    assert by["h2"]["details"]["answer_relevancy"] == 0.5                    # half of the question tokens echoed
    assert r.metrics["pass_rate"] == pytest.approx(1 / 3)


def test_ragas_fallback_definitions(hand_path, cfg):
    r = RagasAdapter().run(hand_path, cfg)
    assert r.backend == "fallback"
    by = {p["id"]: p for p in r.per_example}
    assert by["h1"]["details"]["faithfulness"] == 1.0
    assert by["h1"]["details"]["context_precision"] == 1.0            # relevant chunk ranked first
    assert context_precision(["Lyon is in France.", "Paris is the capital of France."], "Paris") == 0.5
    assert faithfulness("Paris is the capital. It has 50 million people.", ["Paris is the capital of France."]) == 0.5


def test_native_adapter_uses_evalkit_metrics(hand_path, cfg):
    r = NativeAdapter().run(hand_path, cfg)
    assert r.backend == "native" and NativeAdapter().available()
    by = {p["id"]: p for p in r.per_example}
    assert by["h3"]["score"] == 1.0 and by["h3"]["details"]["exact_match"] == 1.0
    assert by["h1"]["score"] == pytest.approx(token_f1("Paris is the capital of France.", "Paris"))
    assert r.metrics["pass_rate_ci_lo"] <= r.metrics["pass_rate"] <= r.metrics["pass_rate_ci_hi"]


# ---------------------------------------------------------------- normalisation, report shape, agreement

def test_metric_name_normalisation(cfg):
    u = normalise_metrics("ragas", {"faithfulness": 0.9, "answer_relevancy": 0.4, "context_precision": 1.0,
                                    "pass_rate": 0.5, "unmapped": 3.0}, cfg["metric_map"])
    assert u == {"correctness": None, "faithfulness": 0.9, "relevancy": 0.4, "pass_rate": 0.5,
                 "context_precision": 1.0}                     # extra mapped names are carried through
    u = normalise_metrics("native", {"token_f1": 0.7, "faithfulness": 1.0, "pass_rate": 1.0}, cfg["metric_map"])
    assert u["correctness"] == 0.7 and u["relevancy"] is None
    assert normalise_metrics("unknown_fw", {"x": 1.0}, cfg["metric_map"]) == {k: None for k in u if k != "context_precision"}


def test_one_row_per_dataset_x_framework_and_disagreements(cfg):
    cfg["sut_fn"] = get_sut(cfg["sut"])
    rep = run_harness(cfg)
    assert len(rep["rows"]) == len(cfg["datasets"]) * len(cfg["frameworks"])
    assert {(r["dataset"], r["framework"]) for r in rep["rows"]} == {
        (d["name"], f) for d in cfg["datasets"] for f in cfg["frameworks"]}
    assert all(r["backend"] == ("native" if r["framework"] == "native" else "fallback") for r in rep["rows"])
    # crafted disagreement: "A spider has eight legs." vs expected "8"
    d = next(x for x in rep["disagreements"] if x["id"] == "qa-02")
    assert d["verdicts"]["ragas"] is True and d["verdicts"]["native"] is False and d["verdicts"]["promptfoo"] is False
    assert len(rep["agreement"]) == len(cfg["datasets"]) * 6          # C(4,2) pairs per dataset
    assert all(-1 <= p["kappa"] <= 1 and p["interpretation"] for p in rep["agreement"])
    md = render_markdown(rep, cfg)
    assert "qa-02" in md and "Cohen's kappa" in md and "fallback" in md


def test_disagreement_list_empty_when_frameworks_agree(hand_path, cfg):
    """A cheating SUT makes every framework pass everything (except relevancy-gated ones on 'Mars.')."""
    cfg["datasets"] = [{"name": "hand", "path": hand_path}]
    cfg["sut_fn"] = get_sut("echo_expected")
    rep = run_harness(cfg, ["promptfoo", "native"])
    assert rep["disagreements"] == [] and rep["agreement"][0]["kappa"] == 1.0


def test_unavailable_backend_does_not_crash(hand_path, cfg):
    for cls in (PromptfooAdapter, DeepEvalAdapter, RagasAdapter):
        a = cls()
        assert isinstance(a, Framework)
        if not a.available():
            r = a.run(hand_path, cfg)
            assert r.backend == "fallback" and r.per_example
    assert set(ADAPTERS) == {"promptfoo", "deepeval", "ragas", "native"}


def test_promptfoo_config_generation_is_valid_yaml(cfg):
    outs = [OUTPUTS[e["id"]] for e in HAND]
    c = generate_promptfoo_config(HAND, outs, cfg["options"]["promptfoo"])
    text = yaml.safe_dump(c, sort_keys=False)
    back = yaml.safe_load(text)
    assert back["providers"] == ["echo"] and back["prompts"] == ["{{output}}"]
    assert len(back["tests"]) == 3 and back["tests"][0]["vars"]["output"] == OUTPUTS["h1"]
    assert back["tests"][0]["assert"][0] == {"type": "icontains", "value": "Paris"}       # {{expected}} filled
    assert back["tests"][2]["assert"] == [{"type": "equals", "value": "Mars."}, {"type": "regex", "value": "^M"}]


def test_jsonl_export_is_pytest_plugin_compatible(tmp_path, cfg):
    cfg["sut_fn"] = get_sut(cfg["sut"])
    rep = run_harness(cfg, ["promptfoo", "native"])
    n = export_jsonl(rep, cfg, tmp_path / "r.jsonl")
    rows = read_jsonl(tmp_path / "r.jsonl")
    assert n == len(rows) == 14
    assert all({"id", "input", "expected", "tags", "output", "verdicts", "passed"} <= set(r) for r in rows)


def test_cli_run_and_list(tmp_path, capsys):
    assert cli.main(["list"]) == 0
    assert "native" in capsys.readouterr().out
    rc = cli.main(["run", "--config", str(DATA / "harness.yaml"), "--frameworks", "ragas,native",
                   "--json", str(tmp_path / "o.json"), "--out", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Dataset x framework" in out and (tmp_path / "results.jsonl").exists()
    assert set(json.loads((tmp_path / "o.json").read_text())) >= {"rows", "agreement", "disagreements", "per_example"}
    assert cli.main(["run", "--config", str(DATA / "harness.yaml"), "--frameworks", "native",
                     "--out", str(tmp_path), "--min-pass-rate", "0.99"]) == 1
    assert cli.main(["run", "--config", str(DATA / "harness.yaml"), "--frameworks", "nope", "--out", str(tmp_path)]) == 2
