"""DeepEval adapter. Real path: `deepeval.metrics.GEval` + `AnswerRelevancyMetric` when importable
(they need a model API, so any failure falls back). Fallback: a native rubric (coverage of the
expected answer by the output, GEval-style 0..1) and lexical answer relevancy (share of the
question's content tokens echoed in the answer)."""
from __future__ import annotations

from typing import Any

from ..textmetrics import content_tokens, coverage, tokenize
from .base import Timer, UnifiedResult, load_dataset, run_sut

try:                                                        # pragma: no cover - optional heavy dep
    from deepeval.metrics import AnswerRelevancyMetric, GEval  # type: ignore
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams  # type: ignore
    _HAVE_DEEPEVAL = True
except Exception:  # a missing or broken optional install falls back  # noqa: BLE001
    _HAVE_DEEPEVAL = False


def lexical_relevancy(question: str, answer: str) -> float:
    """|Q ∩ A| / |Q| over content tokens: how much of the question the answer addresses."""
    q = set(content_tokens(question))
    if not q:
        return 1.0
    a = set(tokenize(answer))
    return len(q & a) / len(q)


class DeepEvalAdapter:
    name = "deepeval"

    def available(self) -> bool:
        return _HAVE_DEEPEVAL

    def run(self, dataset_path: str, config: dict[str, Any]) -> UnifiedResult:
        options = config.get("options", {}).get("deepeval", {})
        examples = load_dataset(dataset_path)
        with Timer() as t:
            outputs, cost = run_sut(examples, config)
            res = None
            if self.available() and not options.get("force_fallback"):
                try:
                    res = self._real(examples, outputs, options)
                except Exception as e:  # any framework failure falls back, with a warning  # noqa: BLE001
                    res = self._fallback(examples, outputs, options)
                    res.warnings.insert(0, f"deepeval metrics failed ({e!r}); using fallback")
            if res is None:
                res = self._fallback(examples, outputs, options)
        res.cost_usd += cost
        res.duration_s = t.elapsed
        return res

    def _real(self, examples, outputs, options) -> UnifiedResult:          # pragma: no cover
        geval = GEval(name="Correctness", criteria="Is the actual output correct given the expected output?",
                      evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT])
        rel = AnswerRelevancyMetric(threshold=options.get("relevancy_threshold", 0.3))
        per = []
        for ex, out in zip(examples, outputs):
            tc = LLMTestCase(input=ex["input"], actual_output=out, expected_output=str(ex["expected"]),
                             retrieval_context=ex.get("context") or None)
            geval.measure(tc)
            rel.measure(tc)
            passed = geval.score >= options.get("rubric_threshold", 0.5) and rel.score >= options.get("relevancy_threshold", 0.3)
            per.append({"id": ex["id"], "score": float(geval.score), "passed": bool(passed),
                        "details": {"geval": geval.score, "relevancy": rel.score, "reason": geval.reason}})
        return self._aggregate(per, backend="real")

    def _fallback(self, examples, outputs, options) -> UnifiedResult:
        rt, relt = options.get("rubric_threshold", 0.5), options.get("relevancy_threshold", 0.3)
        per = []
        for ex, out in zip(examples, outputs):
            g = coverage(str(ex["expected"]), out)
            r = lexical_relevancy(ex["input"], out)
            per.append({"id": ex["id"], "score": g, "passed": bool(g >= rt and r >= relt),
                        "details": {"geval_correctness": g, "answer_relevancy": r, "output": out}})
        return self._aggregate(per, backend="fallback")

    def _aggregate(self, per, backend) -> UnifiedResult:
        n = len(per)
        m = {"geval_correctness": sum(p["details"].get("geval_correctness", p["score"]) for p in per) / n if n else 0.0,
             "answer_relevancy": sum(p["details"].get("answer_relevancy", p["details"].get("relevancy", 0.0)) for p in per) / n if n else 0.0,
             "pass_rate": sum(p["passed"] for p in per) / n if n else 0.0}
        return UnifiedResult(self.name, m, per, backend=backend)
