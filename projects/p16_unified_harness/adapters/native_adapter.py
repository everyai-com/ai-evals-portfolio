"""Native adapter: evalkit metrics directly (no external framework, so never a fallback)."""
from __future__ import annotations

from typing import Any

from evalkit.metrics import wilson_ci

from ..textmetrics import exact_match, faithfulness, token_f1
from .base import Timer, UnifiedResult, load_dataset, run_sut


class NativeAdapter:
    name = "native"

    def available(self) -> bool:
        return True

    def run(self, dataset_path: str, config: dict[str, Any]) -> UnifiedResult:
        options = config.get("options", {}).get("native", {})
        thr = options.get("pass_threshold", 0.5)
        examples = load_dataset(dataset_path)
        with Timer() as t:
            outputs, cost = run_sut(examples, config)
            per = []
            for ex, out in zip(examples, outputs):
                f1 = token_f1(out, str(ex["expected"]))
                per.append({"id": ex["id"], "score": f1, "passed": bool(f1 >= thr),
                            "details": {"exact_match": exact_match(out, str(ex["expected"])), "token_f1": f1,
                                        "faithfulness": faithfulness(out, ex.get("context") or []), "output": out}})
        n = len(per)
        k = sum(p["passed"] for p in per)
        lo, hi = wilson_ci(k, n)
        m = {"token_f1": sum(p["score"] for p in per) / n if n else 0.0,
             "exact_match": sum(p["details"]["exact_match"] for p in per) / n if n else 0.0,
             "faithfulness": sum(p["details"]["faithfulness"] for p in per) / n if n else 0.0,
             "pass_rate": k / n if n else 0.0, "pass_rate_ci_lo": lo, "pass_rate_ci_hi": hi}
        return UnifiedResult(self.name, m, per, cost_usd=cost, duration_s=t.elapsed, backend="native")
