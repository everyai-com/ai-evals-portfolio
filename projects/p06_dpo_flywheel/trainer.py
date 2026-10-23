"""LoRA + DPO training step. Real training runs only when EVALKIT_TRAIN=1 and peft/trl import;
otherwise a dry-run validates the data, computes dataset stats, writes `training_manifest.json`
and a fake adapter directory with `adapter_config.json` so downstream steps are testable offline.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from evalkit.io import read_jsonl, write_jsonl
from evalkit.llm import approx_tokens


@dataclass
class TrainConfig:
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    beta: float = 0.1            # DPO temperature
    learning_rate: float = 5e-6
    epochs: int = 2
    max_length: int = 1024
    batch_size: int = 8
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    throughput_tok_per_s: float = 2500.0   # 7B model, LoRA, bf16, one A100-80GB (rough)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "TrainConfig":
        d = dict(d or {})
        if "target_modules" in d:
            d["target_modules"] = tuple(d["target_modules"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class DataValidationError(ValueError):
    pass


def validate_pairs(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise DataValidationError("no pairs")
    for i, r in enumerate(rows):
        for k in ("prompt", "chosen", "rejected"):
            if not isinstance(r.get(k), str) or not r[k].strip():
                raise DataValidationError(f"row {i}: missing or empty {k!r}")
        if r["chosen"].strip() == r["rejected"].strip():
            raise DataValidationError(f"row {i}: chosen == rejected")


def dataset_stats(rows: list[dict[str, Any]], cfg: TrainConfig) -> dict[str, Any]:
    """Token counts use evalkit.llm.approx_tokens (len/4). GPU-hours estimate:
    each pair costs ~2 sequences (chosen, rejected) x 2 passes (policy forward+backward, reference
    forward) => 4 x pair_tokens per epoch; hours = 4 * tokens * epochs / throughput / 3600."""
    p = np.array([approx_tokens(r["prompt"]) for r in rows])
    c = np.array([approx_tokens(r["chosen"]) for r in rows])
    j = np.array([approx_tokens(r["rejected"]) for r in rows])
    pair_tokens = int((p + c + j).sum())
    srcs: dict[str, int] = {}
    for r in rows:
        srcs[r.get("source", "unknown")] = srcs.get(r.get("source", "unknown"), 0) + 1
    return {"n_pairs": len(rows), "prompt_tokens": int(p.sum()), "chosen_tokens": int(c.sum()),
            "rejected_tokens": int(j.sum()), "total_tokens": pair_tokens,
            "mean_chosen_len": float(c.mean()), "mean_rejected_len": float(j.mean()),
            "truncated_pairs": int(((p + np.maximum(c, j)) > cfg.max_length).sum()),
            "sources": srcs, "mean_confidence": float(np.mean([r.get("confidence", 0.0) for r in rows])),
            "estimated_gpu_hours": round(4 * pair_tokens * cfg.epochs / cfg.throughput_tok_per_s / 3600, 4)}


def _real_available() -> bool:
    try:
        import peft  # noqa: F401
        import trl  # noqa: F401
        return True
    except Exception:
        return False


def train_lora(pairs_path: str | Path, base_model: str, out_dir: str | Path,
               config: dict[str, Any] | TrainConfig | None = None, dry_run: bool | None = None) -> dict[str, Any]:
    """Train (or dry-run) a LoRA adapter with DPO on `pairs_path`. Returns the manifest.

    dry_run=None -> real training iff EVALKIT_TRAIN=1 and peft+trl are importable."""
    cfg = config if isinstance(config, TrainConfig) else TrainConfig.from_dict(config)
    rows = read_jsonl(pairs_path)
    validate_pairs(rows)
    stats = dataset_stats(rows, cfg)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if dry_run is None:
        dry_run = not (os.environ.get("EVALKIT_TRAIN") == "1" and _real_available())
    manifest = {"base_model": base_model, "pairs_path": str(pairs_path), "adapter_dir": str(out),
                "hyperparams": asdict(cfg), "dataset": stats, "dry_run": bool(dry_run)}
    if dry_run:
        adapter_cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": base_model,
                       "r": cfg.lora_r, "lora_alpha": cfg.lora_alpha, "lora_dropout": cfg.lora_dropout,
                       "target_modules": list(cfg.target_modules), "dry_run": True}
        (out / "adapter_config.json").write_text(json.dumps(adapter_cfg, indent=2))
        # Stand-in for the learned weights: the prompts the adapter "saw", used by the simulated
        # post-train eval in nightly.py. A real adapter writes adapter_model.safetensors instead.
        write_jsonl(out / "adapter_memory.jsonl", ({"prompt": r["prompt"], "chosen": r["chosen"]} for r in rows))
        manifest["status"] = "dry_run_ok"
    else:
        manifest["status"] = _train_real(rows, base_model, out, cfg)
    (out / "training_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _train_real(rows: list[dict[str, Any]], base_model: str, out: Path, cfg: TrainConfig) -> str:  # pragma: no cover
    """peft + trl.DPOTrainer path (not exercised in tests; needs GPU + EVALKIT_TRAIN=1)."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
    ds = Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in rows])
    peft_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                          target_modules=list(cfg.target_modules), task_type="CAUSAL_LM")
    args = DPOConfig(output_dir=str(out), beta=cfg.beta, learning_rate=cfg.learning_rate,
                     num_train_epochs=cfg.epochs, per_device_train_batch_size=cfg.batch_size,
                     max_length=cfg.max_length, logging_steps=10, save_strategy="no", report_to=[])
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.model.save_pretrained(str(out))
    return "trained"
