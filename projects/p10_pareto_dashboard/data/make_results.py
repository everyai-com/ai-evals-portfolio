"""Seeded generator for data/results.jsonl: 3 tenants x 6 routing policies x 150 shared tasks.

Each tenant has its own difficulty distribution (Beta). For every task we pre-draw one uniform per
model so that `system_two_only` and the System-Two leg of a cascade agree on the same task (paired
design). Costs come from evalkit.llm PRICES_PER_MTOK and are stored per model in `cost_breakdown`
so the dashboard can re-price System-Two without re-running anything.

Run: .venv/bin/python -m projects.p10_pareto_dashboard.data.make_results [--out path] [--seed 0]
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from evalkit.io import write_jsonl
from evalkit.llm import estimate_cost

ONE, TWO = "mock-system-one", "mock-system-two"
TENANTS = {"pizza-us": (2.0, 3.0), "pizza-uk": (2.5, 2.5), "wings-ca": (3.0, 2.0)}  # Beta(a, b) difficulty
CASCADE_THRESHOLDS = (0.5, 0.7, 0.9)
POLICIES = ["system_one_only", "system_two_only"] + [f"cascade_confidence_{t}" for t in CASCADE_THRESHOLDS] + ["router_classifier"]


def _leg(rng: np.random.Generator, model: str, d: float, u: float) -> dict[str, Any]:
    if model == ONE:
        p_ok = 0.97 - 0.75 * d
        tin, tout = int(rng.normal(600, 100)), int(rng.normal(120, 40))
        lat = float(rng.gamma(4, 120))
    else:
        p_ok = 0.99 - 0.25 * d
        tin, tout = int(rng.normal(600, 100)), int(rng.normal(180, 50))
        lat = float(rng.gamma(4, 450))
    tin, tout = max(50, tin), max(10, tout)
    ok = u < p_ok
    conf = float(np.clip(0.5 + 0.45 * (1 - d) + rng.normal(0, 0.12) + (0.1 if ok else -0.12), 0.0, 1.0))
    return {"model": model, "success": bool(ok), "tokens_in": tin, "tokens_out": tout,
            "cost": estimate_cost(model, tin, tout), "latency_ms": lat, "confidence": conf}


def make_results(seed: int = 0, n_tasks: int = 150) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for tenant, (a, b) in TENANTS.items():
        for i in range(n_tasks):
            task_id = f"{tenant}-t{i:03d}"
            d = float(rng.beta(a, b))
            u_one, u_two = float(rng.random()), float(rng.random())
            one = _leg(rng, ONE, d, u_one)
            two = _leg(rng, TWO, d, u_two)
            d_hat = d + float(rng.normal(0, 0.2))
            for policy in POLICIES:
                if policy == "system_one_only":
                    legs, model = [one], ONE
                elif policy == "system_two_only":
                    legs, model = [two], TWO
                elif policy.startswith("cascade_confidence_"):
                    t = float(policy.rsplit("_", 1)[1])
                    legs = [one, two] if one["confidence"] < t else [one]
                    model = f"{ONE}->{TWO}@{t}"
                else:
                    legs = [two] if d_hat > 0.5 else [one]
                    model = f"router({ONE}|{TWO})"
                breakdown = {ONE: 0.0, TWO: 0.0}
                for leg in legs:
                    breakdown[leg["model"]] += leg["cost"]
                rows.append({
                    "tenant": tenant, "routing_policy": policy, "model_or_cascade": model, "task_id": task_id,
                    "success": legs[-1]["success"], "cost_usd": round(sum(breakdown.values()), 8),
                    "latency_ms": round(sum(leg["latency_ms"] for leg in legs), 1),
                    "tokens": {"input": sum(leg["tokens_in"] for leg in legs), "output": sum(leg["tokens_out"] for leg in legs)},
                    "cost_breakdown": {k: round(v, 8) for k, v in breakdown.items()},
                    "escalated": len(legs) > 1 or (policy == "router_classifier" and model and legs[0]["model"] == TWO),
                    "difficulty": round(d, 4),
                })
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(Path(__file__).parent / "results.jsonl"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-tasks", type=int, default=150)
    a = p.parse_args(argv)
    n = write_jsonl(a.out, make_results(a.seed, a.n_tasks))
    print(f"wrote {n} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
