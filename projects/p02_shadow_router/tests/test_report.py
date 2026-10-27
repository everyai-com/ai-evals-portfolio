import json
import math

import pytest

from evalkit.metrics import wilson_ci
from projects.p02_shadow_router.cli import main
from projects.p02_shadow_router.report import (
    DEFAULT_THRESHOLDS,
    ReportBuilder,
    load_thresholds,
    recommend,
)


def rec(i, p_text, s_text, p_cost, s_cost, p_lat, s_lat, shadowed=True, err=None, expected=None, ts=0.0):
    return {"request_id": f"r{i}", "kind": "completion", "prompt": "q", "system": "", "ts": ts,
            "meta": {"expected": expected} if expected else {}, "shadowed": shadowed,
            "primary": {"text": p_text, "model": "p", "input_tokens": 5, "output_tokens": 5, "tokens": 10,
                        "latency_ms": p_lat, "cost_usd": p_cost},
            "shadow": None if err else {"text": s_text, "model": "s", "input_tokens": 5, "output_tokens": 7,
                                        "tokens": 12, "latency_ms": s_lat, "cost_usd": s_cost},
            "shadow_error": err}


def hand_built_store():
    rows = []
    # 10 shadowed: 7 agree, 2 disagree, 1 error; 10 unsampled
    for i in range(7):
        rows.append(rec(i, "a", "a", 0.001, 0.002, 100, 150, ts=i * 3600))
    rows.append(rec(7, "a", "b", 0.001, 0.002, 100, 150, ts=7 * 3600))
    rows.append(rec(8, "a", "b", 0.001, 0.002, 100, 150, ts=8 * 3600))
    rows.append(rec(9, "a", None, 0.001, 0.0, 100, 0, err="RuntimeError: down", ts=9 * 3600))
    for i in range(10, 20):
        rows.append(rec(i, "a", None, 0.001, 0.0, 100, 0, shadowed=False, ts=(i + 20) * 3600))
    return rows


def test_report_numbers_on_hand_built_store():
    s = ReportBuilder(hand_built_store(), n_boot=500).build()
    assert s["n_requests"] == 20 and s["n_shadowed"] == 10 and s["n_pairs"] == 9
    assert s["observed_sample_rate"] == 0.5
    assert math.isclose(s["agreement_rate"], 7 / 9)
    assert s["agreement_ci"] == pytest.approx(list(wilson_ci(7, 9)))
    assert s["classes"] == {"same": 7, "shadow_better": 0, "shadow_worse": 0, "different_unknown": 2, "shadow_error": 1}
    assert math.isclose(s["error_rate"], 0.1)
    assert s["cost"]["delta"] == pytest.approx(0.001) and s["cost"]["ci_lo"] == pytest.approx(0.001)
    assert s["latency"]["delta"] == pytest.approx(50.0)
    assert s["latency_primary"]["p95"] == 100 and s["latency_shadow"]["p50"] == 150
    assert s["cost_ratio"] == pytest.approx(2.0)
    # span = (39h - 0h) floored at 1 day -> 1.625 days; 20 requests -> 12.3/day
    assert s["traffic"]["span_days"] == pytest.approx(39 / 24)
    rpd = 20 / (39 / 24)
    assert s["projected_monthly_cost"]["primary"] == pytest.approx(rpd * 30 * 0.001)
    assert s["projected_monthly_cost"]["shadow"] == pytest.approx(rpd * 30 * 0.002)
    assert s["tokens"]["delta"] == pytest.approx(2.0)


def test_empty_store_does_not_crash():
    s = ReportBuilder([]).build()
    assert s["n_requests"] == 0 and s["recommendation"] == "hold"
    assert "Recommendation" in ReportBuilder([]).to_markdown(s)


def base_summary(**over):
    s = {"n_pairs": 100, "n_shadowed": 100, "agreement_ci": [0.95, 0.99], "error_rate": 0.0,
         "shadow_worse_rate": 0.0, "cost_ratio": 0.8, "latency_p95_ratio": 1.0}
    s.update(over)
    return s


def test_recommendation_thresholds():
    th = load_thresholds(None)
    assert recommend(base_summary(), th)[0] == "promote"
    assert recommend(base_summary(n_pairs=10), th)[0] == "hold"
    assert recommend(base_summary(cost_ratio=1.5), th)[0] == "hold"
    assert recommend(base_summary(agreement_ci=[0.85, 0.95]), th)[0] == "hold"
    assert recommend(base_summary(latency_p95_ratio=2.0), th)[0] == "hold"
    assert recommend(base_summary(cost_ratio=12.0), th)[0] == "reject"
    assert recommend(base_summary(error_rate=0.2), th)[0] == "reject"
    assert recommend(base_summary(shadow_worse_rate=0.2), th)[0] == "reject"
    assert recommend(base_summary(agreement_ci=[0.5, 0.7]), th)[0] == "reject"


def test_thresholds_yaml_overrides(tmp_path):
    cfg = tmp_path / "t.yaml"
    cfg.write_text("min_pairs: 5\npromote:\n  max_cost_ratio: 30\nreject:\n  min_cost_ratio: 50\n")
    th = load_thresholds(cfg)
    assert th["min_pairs"] == 5 and th["promote"]["max_cost_ratio"] == 30
    assert th["promote"]["min_agreement_lo"] == DEFAULT_THRESHOLDS["promote"]["min_agreement_lo"]
    assert recommend(base_summary(n_pairs=6, cost_ratio=20.0), th)[0] == "promote"


def test_cli_end_to_end(tmp_path, capsys):
    out = tmp_path / "shadow.jsonl"
    assert main(["run", "--rate", "0.5", "--out", str(out)]) == 0
    js = tmp_path / "m.json"
    rc = main(["report", "--store", str(out), "--json", str(js)])
    md = capsys.readouterr().out
    assert rc in (0, 1) and "# Shadow routing report" in md and "Projected monthly" in md
    m = json.loads(js.read_text())
    assert 70 < m["n_shadowed"] < 130 and m["n_requests"] == 200
    assert m["recommendation"] in ("promote", "hold", "reject")
    assert main(["bogus"]) == 2
