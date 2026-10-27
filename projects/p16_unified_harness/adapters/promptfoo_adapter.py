"""promptfoo adapter. Real path: the `promptfoo` CLI on PATH, fed a generated config that grades the
SUT's pre-computed outputs via the `echo` provider. Fallback: a native implementation of its
assertion types (equals, contains, icontains, regex, llm-rubric via a MockLLM judge); `javascript`
asserts are not supported offline and produce a warning."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from evalkit.llm import get_client

from ..textmetrics import coverage
from .base import Timer, UnifiedResult, load_dataset, run_sut

SUPPORTED = {"equals", "contains", "icontains", "regex", "llm-rubric"}


def _rubric_judge_handlers() -> list[tuple[str, Any]]:
    """Deterministic rubric judge: PASS when the expected answer's tokens are covered by the output."""
    def judge(system: str, prompt: str) -> str:
        m = re.search(r"EXPECTED: (.*)\nOUTPUT: (.*)\nRUBRIC", prompt, re.DOTALL)
        if not m:
            return '{"pass": false, "score": 0.0}'
        cov = coverage(m.group(1), m.group(2))
        return json.dumps({"pass": cov >= 0.5, "score": round(cov, 3)})
    return [(r"RUBRIC", judge)]


def generate_promptfoo_config(examples: list[dict[str, Any]], outputs: list[str],
                              options: dict[str, Any]) -> dict[str, Any]:
    """A valid promptfooconfig: the `echo` provider returns the pre-computed SUT output so promptfoo
    only grades. Asserts come from the example (`asserts`) or `options.default_asserts`."""
    tests = []
    for ex, out in zip(examples, outputs):
        asserts = []
        for a in resolve_asserts(ex, options):
            asserts.append({"type": a["type"], "value": a["value"]})
        tests.append({"description": ex["id"], "vars": {"input": ex["input"], "expected": str(ex["expected"]),
                                                          "output": out}, "assert": asserts})
    return {"description": "unified-harness generated config", "prompts": ["{{output}}"],
            "providers": ["echo"], "tests": tests}


def resolve_asserts(ex: dict[str, Any], options: dict[str, Any]) -> list[dict[str, Any]]:
    raw = ex.get("asserts") or options.get("default_asserts") or [{"type": "icontains", "value": "{{expected}}"}]
    out = []
    for a in raw:
        val = str(a.get("value", "")).replace("{{expected}}", str(ex["expected"]))
        out.append({"type": a["type"], "value": val})
    return out


class PromptfooAdapter:
    name = "promptfoo"

    def __init__(self, judge=None):
        self._judge = judge

    def available(self) -> bool:
        return shutil.which("promptfoo") is not None

    def run(self, dataset_path: str, config: dict[str, Any]) -> UnifiedResult:
        options = config.get("options", {}).get("promptfoo", {})
        examples = load_dataset(dataset_path)
        with Timer() as t:
            outputs, cost = run_sut(examples, config)
            if self.available() and not options.get("force_fallback"):
                try:
                    res = self._run_cli(examples, outputs, options)
                    res.cost_usd += cost
                    res.duration_s = t.elapsed if hasattr(t, "elapsed") else 0.0
                    return res
                except Exception as e:                       # fall back, but say so  # noqa: BLE001
                    warn = f"promptfoo CLI failed ({e!r}); using fallback"
                    res = self._fallback(examples, outputs, options)
                    res.warnings.insert(0, warn)
            else:
                res = self._fallback(examples, outputs, options)
        res.cost_usd += cost
        res.duration_s = t.elapsed
        return res

    # ---- real path
    def _run_cli(self, examples, outputs, options) -> UnifiedResult:
        cfg = generate_promptfoo_config(examples, outputs, options)
        with tempfile.TemporaryDirectory() as d:
            cfg_path, out_path = Path(d) / "promptfooconfig.yaml", Path(d) / "out.json"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            subprocess.run(["promptfoo", "eval", "-c", str(cfg_path), "-o", str(out_path), "--no-cache"],
                           check=True, capture_output=True, timeout=600)
            data = json.loads(out_path.read_text())
        rows = data.get("results", {}).get("results", data.get("results", []))
        per = []
        for ex, r in zip(examples, rows):
            ok = bool(r.get("success"))
            score = float(r.get("score", 1.0 if ok else 0.0))
            per.append({"id": ex["id"], "score": score, "passed": ok, "details": r.get("gradingResult", {})})
        n = len(per)
        return UnifiedResult(self.name, {"pass_rate": sum(p["passed"] for p in per) / n if n else 0.0,
                                         "rubric_score": sum(p["score"] for p in per) / n if n else 0.0},
                             per, backend="real")

    # ---- fallback
    def _fallback(self, examples, outputs, options) -> UnifiedResult:
        judge = self._judge or get_client(model=options.get("judge_model", "mock"), handlers=_rubric_judge_handlers())
        warnings: list[str] = []
        per, judge_cost = [], 0.0
        for ex, out in zip(examples, outputs):
            results, rubric_scores = [], []
            for a in resolve_asserts(ex, options):
                typ, val = a["type"], a["value"]
                if typ == "equals":
                    ok = out == val
                elif typ == "contains":
                    ok = val in out
                elif typ == "icontains":
                    ok = val.lower() in out.lower()
                elif typ == "regex":
                    ok = re.search(val, out) is not None
                elif typ == "llm-rubric":
                    resp = judge.complete(f"EXPECTED: {ex['expected']}\nOUTPUT: {out}\nRUBRIC: {val}\n"
                                          "Reply JSON {\"pass\": bool, \"score\": float}")
                    judge_cost += resp.cost_usd
                    try:
                        verdict = resp.json()
                        ok, sc = bool(verdict.get("pass")), float(verdict.get("score", 0.0))
                    except (ValueError, TypeError, AttributeError):
                        ok, sc = False, 0.0
                    rubric_scores.append(sc)
                else:
                    warnings.append(f"{ex['id']}: assertion type {typ!r} not supported in fallback; skipped")
                    continue
                results.append({"type": typ, "value": val, "pass": bool(ok)})
            passed = all(r["pass"] for r in results) if results else False
            score = sum(rubric_scores) / len(rubric_scores) if rubric_scores else float(passed)
            per.append({"id": ex["id"], "score": score, "passed": passed, "details": {"asserts": results}})
        n = len(per)
        return UnifiedResult(self.name, {"pass_rate": sum(p["passed"] for p in per) / n if n else 0.0,
                                         "rubric_score": sum(p["score"] for p in per) / n if n else 0.0},
                             per, cost_usd=judge_cost, warnings=sorted(set(warnings)), backend="fallback")
