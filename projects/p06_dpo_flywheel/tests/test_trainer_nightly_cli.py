"""Trainer dry-run manifest, nightly idempotency + promotion gate, CLI exit codes."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalkit.io import read_jsonl, write_jsonl
from evalkit.llm import MockLLM
from projects.p06_dpo_flywheel import cli
from projects.p06_dpo_flywheel.nightly import NightlyConfig, NightlyJob
from projects.p06_dpo_flywheel.trainer import DataValidationError, train_lora

HERE = Path(__file__).resolve().parent.parent
EVENTS = HERE / "data" / "feedback.jsonl"


def test_dry_run_manifest_fields(tmp_path, monkeypatch):
    monkeypatch.delenv("EVALKIT_TRAIN", raising=False)
    rows = [{"prompt": f"q{i} about order", "chosen": "a good long answer " * 5, "rejected": "no", "source": "correction",
             "confidence": 0.9} for i in range(10)]
    pp = tmp_path / "train.jsonl"
    write_jsonl(pp, rows)
    m = train_lora(pp, "base/model", tmp_path / "adapter", {"lora_r": 8, "epochs": 3})
    assert m["dry_run"] is True and m["status"] == "dry_run_ok"
    hp = m["hyperparams"]
    assert hp["lora_r"] == 8 and hp["lora_alpha"] == 32 and hp["lora_dropout"] == 0.05 and hp["beta"] == 0.1
    assert hp["learning_rate"] == 5e-6 and hp["epochs"] == 3
    d = m["dataset"]
    assert d["n_pairs"] == 10 and d["total_tokens"] == d["prompt_tokens"] + d["chosen_tokens"] + d["rejected_tokens"]
    assert d["estimated_gpu_hours"] > 0 and d["sources"] == {"correction": 10}
    assert json.loads((tmp_path / "adapter" / "training_manifest.json").read_text()) == json.loads(json.dumps(m))
    ac = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert ac["peft_type"] == "LORA" and ac["r"] == 8 and ac["base_model_name_or_path"] == "base/model"
    with pytest.raises(DataValidationError):
        write_jsonl(pp, [{"prompt": "x", "chosen": "same", "rejected": "same"}])
        train_lora(pp, "b", tmp_path / "a2", None, dry_run=True)


def test_nightly_is_idempotent(tmp_path):
    job = NightlyJob(tmp_path / "state", EVENTS, client=None, golden_path=tmp_path / "g.jsonl",
                     changelog_path=tmp_path / "C.md")
    r1 = job.run("2026-10-23")
    assert r1.status == "promoted" and r1.new_pairs >= 60 and r1.ingest["accepted"] > 300
    assert (tmp_path / "state" / "adapters" / "2026-10-23" / "adapter_config.json").exists()
    assert (tmp_path / "state" / "reports" / "eval_diff_2026-10-23.md").exists()
    assert len(read_jsonl(tmp_path / "g.jsonl")) == r1.new_pairs
    r2 = NightlyJob(tmp_path / "state", EVENTS, client=None, golden_path=tmp_path / "g.jsonl",
                    changelog_path=tmp_path / "C.md").run("2026-10-24")
    assert r2.new_pairs == 0 and r2.status == "skipped" and r2.ingest["accepted"] == 0
    assert r2.reasons == ["validation failed: min_pairs"]
    assert len(read_jsonl(tmp_path / "g.jsonl")) == r1.new_pairs      # golden set unchanged
    st = json.loads((tmp_path / "state" / "nightly_state.json").read_text())
    assert [h["status"] for h in st["history"]] == ["promoted", "skipped"]
    assert json.loads((tmp_path / "state" / "promoted.json").read_text())["date"] == "2026-10-23"


def test_promotion_blocked_when_post_train_eval_is_worse(tmp_path):
    def worse_hook(adapter_dir, previous, golden):
        n = len(golden)
        return [1.0] * n, [1.0 if i % 3 else 0.0 for i in range(n)]    # after loses a third
    job = NightlyJob(tmp_path / "state", EVENTS, client=None, eval_hook=worse_hook,
                     golden_path=tmp_path / "g.jsonl", changelog_path=tmp_path / "C.md")
    r = job.run("2026-10-23")
    assert r.status == "not_promoted" and r.eval["worse"] and r.eval["delta"] < 0
    assert r.reasons and "worse" in r.reasons[0]
    assert not (tmp_path / "state" / "promoted.json").exists()
    assert (tmp_path / "state" / "adapters" / "2026-10-23" / "training_manifest.json").exists()


def test_validation_blocks_single_tenant_dominance(tmp_path):
    cfg = NightlyConfig(max_tenant_share=0.3)
    r = NightlyJob(tmp_path / "state", EVENTS, client=None, config=cfg, golden_path=tmp_path / "g.jsonl",
                   changelog_path=tmp_path / "C.md").run("2026-10-23")
    assert r.status == "skipped" and r.reasons == ["validation failed: max_tenant_share"]


def test_regen_client_adds_pairs_in_nightly(tmp_path):
    client = MockLLM(handlers=[(r".", cli.demo_regen_handler)])
    r = NightlyJob(tmp_path / "state", EVENTS, client=client, golden_path=tmp_path / "g.jsonl",
                   changelog_path=tmp_path / "C.md").run("2026-10-23")
    assert r.pairs_by_source.get("regenerated", 0) > 0 and r.status == "promoted"


def test_cli_end_to_end(tmp_path):
    state, pairs, g, c = tmp_path / "state", tmp_path / "pairs", tmp_path / "g.jsonl", tmp_path / "C.md"
    assert cli.main(["ingest", "--events", str(EVENTS), "--state", str(state)]) == 0
    assert cli.main(["build-pairs", "--state", str(state), "--out", str(pairs), "--golden", str(g), "--changelog", str(c)]) == 0
    assert (pairs / "train.jsonl").exists() and (pairs / "val.jsonl").exists()
    row = read_jsonl(pairs / "train.jsonl")[0]
    assert {"prompt", "chosen", "rejected"} <= set(row)
    assert cli.main(["train", "--pairs", str(pairs / "train.jsonl"), "--out", str(tmp_path / "ad"), "--dry-run"]) == 0
    assert cli.main(["nightly", "--events", str(EVENTS), "--state", str(tmp_path / "s2"), "--date", "2026-10-23",
                     "--golden", str(tmp_path / "g2.jsonl"), "--changelog", str(tmp_path / "C2.md"),
                     "--json", str(tmp_path / "n.json")]) == 0
    assert json.loads((tmp_path / "n.json").read_text())["status"] == "promoted"
    assert cli.main(["build-pairs", "--state", str(tmp_path / "empty")]) == 2
    assert cli.main(["nope"]) == 2
