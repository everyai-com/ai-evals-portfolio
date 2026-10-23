"""Seeded generator for the demo results files.

* model_a.jsonl / model_b.jsonl  n=400 paired per-example results `{id, strata, success, score}`.
  Model A succeeds with a strata-dependent base rate (~0.70 overall). Model B is A with
  P(1->0)=0.003 and P(0->1)=0.110, i.e. a TRUE +3.0% success delta with few discordant pairs;
  `score` is a continuous quality in [0,1] with B truly +0.03.
* model_c.jsonl  a NULL candidate: A with 20% of outcomes re-drawn at A's own rate (no true delta).
* small_a.jsonl / small_b.jsonl  n=50 from the same process: the same effect, too few samples.
* runs.jsonl  3 seeds x 2 models over the 400 ids `{id, model, seed, strata, success, score}`.
  Each (model, seed) run gets a seed offset ~ N(0, 0.04) on its success probability, so the
  within-model seed std is comparable to the +3% between-model delta.

Usage: python -m projects.p07_significance_engine.data.make_results --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from evalkit.io import write_jsonl

STRATA = {"qa": 0.80, "code": 0.60, "math": 0.65, "summarize": 0.75}
P_BREAK, P_FIX = 0.003, 0.110  # B relative to A on the success metric
SCORE_DELTA, SEED_SD = 0.03, 0.04


def _base(rng: np.random.Generator, n: int, prefix: str) -> list[dict]:
    names = list(STRATA)
    rows = []
    for i in range(n):
        s = names[i % len(names)]
        succ = int(rng.random() < STRATA[s])
        score = float(np.clip(rng.normal(0.45 + 0.3 * succ + 0.1 * (STRATA[s] - 0.7), 0.12), 0, 1))
        rows.append(
            {"id": f"{prefix}-{i:04d}", "strata": s, "success": succ, "score": round(score, 4)}
        )
    return rows


def _b_from_a(rng: np.random.Generator, a_rows: list[dict]) -> list[dict]:
    out = []
    for r in a_rows:
        u = rng.random()
        succ = (0 if u < P_BREAK else 1) if r["success"] else (1 if u < P_FIX else 0)
        score = float(np.clip(r["score"] + SCORE_DELTA + rng.normal(0, 0.05), 0, 1))
        out.append({**r, "success": succ, "score": round(score, 4)})
    return out


def _c_from_a(rng: np.random.Generator, a_rows: list[dict]) -> list[dict]:
    out = []
    for r in a_rows:
        succ = r["success"] if rng.random() < 0.8 else int(rng.random() < STRATA[r["strata"]])
        score = float(np.clip(r["score"] + rng.normal(0, 0.05), 0, 1))
        out.append({**r, "success": succ, "score": round(score, 4)})
    return out


def _runs(
    rng: np.random.Generator, a_rows: list[dict], b_rows: list[dict], n_seeds: int
) -> list[dict]:
    rows = []
    for model, base in (("model_a", a_rows), ("model_b", b_rows)):
        for seed in range(n_seeds):
            offset = float(rng.normal(0, SEED_SD))
            for r in base:
                p = np.clip(
                    STRATA[r["strata"]] + (0.03 if model == "model_b" else 0.0) + offset, 0.02, 0.98
                )
                # keep ~70% of the per-example outcome, re-draw the rest around the shifted rate
                succ = r["success"] if rng.random() < 0.7 else int(rng.random() < p)
                score = float(np.clip(r["score"] + offset + rng.normal(0, 0.03), 0, 1))
                rows.append(
                    {
                        "id": r["id"],
                        "model": model,
                        "seed": seed,
                        "strata": r["strata"],
                        "success": succ,
                        "score": round(score, 4),
                    }
                )
    return rows


def make_all(
    out_dir: Path, seed: int = 0, n: int = 400, n_small: int = 50, n_seeds: int = 3
) -> dict:
    rng = np.random.default_rng(seed)
    a = _base(rng, n, "ex")
    b = _b_from_a(rng, a)
    c = _c_from_a(rng, a)
    sa = _base(rng, n_small, "sm")
    sb = _b_from_a(rng, sa)
    runs = _runs(rng, a, b, n_seeds)
    counts = {
        "model_a.jsonl": write_jsonl(out_dir / "model_a.jsonl", a),
        "model_b.jsonl": write_jsonl(out_dir / "model_b.jsonl", b),
        "model_c.jsonl": write_jsonl(out_dir / "model_c.jsonl", c),
        "small_a.jsonl": write_jsonl(out_dir / "small_a.jsonl", sa),
        "small_b.jsonl": write_jsonl(out_dir / "small_b.jsonl", sb),
        "runs.jsonl": write_jsonl(out_dir / "runs.jsonl", runs),
    }
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(Path(__file__).parent))
    a = ap.parse_args(argv)
    for name, n in make_all(Path(a.out_dir), a.seed).items():
        print(f"wrote {n:5d} rows -> {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
