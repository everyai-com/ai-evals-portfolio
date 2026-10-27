"""RAGAS adapter. Real path: `ragas.evaluate` with faithfulness / answer_relevancy /
context_precision when importable (needs an LLM + embeddings, so failures fall back). Fallback:
native token-overlap implementations of the same three definitions (see textmetrics)."""
from __future__ import annotations

from typing import Any

from ..textmetrics import context_precision, faithfulness
from .base import Timer, UnifiedResult, load_dataset, run_sut
from .deepeval_adapter import lexical_relevancy

try:                                                        # pragma: no cover - optional heavy dep
    from ragas import evaluate as _ragas_evaluate  # type: ignore
    from ragas.metrics import answer_relevancy as _ar  # type: ignore
    from ragas.metrics import context_precision as _cp
    from ragas.metrics import faithfulness as _f
    _HAVE_RAGAS = True
except Exception:  # a missing or broken optional install falls back  # noqa: BLE001
    _HAVE_RAGAS = False


class RagasAdapter:
    name = "ragas"

    def available(self) -> bool:
        return _HAVE_RAGAS

    def run(self, dataset_path: str, config: dict[str, Any]) -> UnifiedResult:
        options = config.get("options", {}).get("ragas", {})
        examples = load_dataset(dataset_path)
        with Timer() as t:
            outputs, cost = run_sut(examples, config)
            res = None
            if self.available() and not options.get("force_fallback"):
                try:
                    res = self._real(examples, outputs, options)
                except Exception as e:  # any framework failure falls back, with a warning  # noqa: BLE001
                    res = self._fallback(examples, outputs, options)
                    res.warnings.insert(0, f"ragas evaluate failed ({e!r}); using fallback")
            if res is None:
                res = self._fallback(examples, outputs, options)
        res.cost_usd += cost
        res.duration_s = t.elapsed
        return res

    def _real(self, examples, outputs, options) -> UnifiedResult:          # pragma: no cover
        from datasets import Dataset  # type: ignore
        ds = Dataset.from_dict({"question": [e["input"] for e in examples], "answer": outputs,
                                "contexts": [e.get("context") or [""] for e in examples],
                                "ground_truth": [str(e["expected"]) for e in examples]})
        scores = _ragas_evaluate(ds, metrics=[_f, _ar, _cp]).to_pandas()
        per = []
        for ex, (_, row) in zip(examples, scores.iterrows()):
            f, r, c = float(row["faithfulness"]), float(row["answer_relevancy"]), float(row["context_precision"])
            per.append({"id": ex["id"], "score": f, "passed": self._passed(f, r, options),
                        "details": {"faithfulness": f, "answer_relevancy": r, "context_precision": c}})
        return self._aggregate(per, "real")

    def _fallback(self, examples, outputs, options) -> UnifiedResult:
        ov = options.get("support_overlap", 0.5)
        per = []
        for ex, out in zip(examples, outputs):
            ctx = ex.get("context") or []
            f = faithfulness(out, ctx, ov)
            r = lexical_relevancy(ex["input"], out)
            c = context_precision(ctx, str(ex["expected"]), ov)
            per.append({"id": ex["id"], "score": f, "passed": self._passed(f, r, options),
                        "details": {"faithfulness": f, "answer_relevancy": r, "context_precision": c, "output": out}})
        return self._aggregate(per, "fallback")

    @staticmethod
    def _passed(f: float, r: float, options: dict[str, Any]) -> bool:
        return bool(f >= options.get("faithfulness_threshold", 0.6) and r >= options.get("relevancy_threshold", 0.3))

    def _aggregate(self, per, backend) -> UnifiedResult:
        n = len(per)
        keys = ("faithfulness", "answer_relevancy", "context_precision")
        m = {k: (sum(p["details"][k] for p in per) / n if n else 0.0) for k in keys}
        m["pass_rate"] = sum(p["passed"] for p in per) / n if n else 0.0
        return UnifiedResult(self.name, m, per, backend=backend)
