"""Shadow routing engine.

A `ShadowProxy` sits in front of the production model. Every call is answered by the primary
model exactly as before; a deterministic hash of the request id decides whether the same prompt
is *also* sent to a candidate ("shadow") model on a background thread. The shadow result never
reaches the caller and shadow failures never raise; both responses are written to a store so a
report can later diff cost, latency and answer quality.

Sampling
--------
``should_shadow(request_id, rate, seed)`` maps ``sha256(f"{seed}:{request_id}")`` to a uniform
number in [0, 1) and shadows when it is below ``rate``. The same request id is therefore always
(or never) shadowed for a given seed, which makes replays reproducible and keeps the sampled
share at ``rate`` in expectation (the first 32 hex digits are uniform).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

from evalkit.io import read_jsonl
from evalkit.llm import LLMClient, LLMResponse
from evalkit.trajectory import Step, Trajectory

log = logging.getLogger("p02_shadow_router")

# --------------------------------------------------------------------------- sampling

def should_shadow(request_id: str, sample_rate: float, seed: int = 0) -> bool:
    """Deterministic hash-based Bernoulli(sample_rate) draw keyed on (seed, request_id)."""
    if sample_rate <= 0:
        return False
    if sample_rate >= 1:
        return True
    digest = hashlib.sha256(f"{seed}:{request_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0x1_0000_0000 < sample_rate


# --------------------------------------------------------------------------- stores

class MemoryStore:
    """In-memory store with the same interface as JsonlStore (used by tests)."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def append(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.rows.append(json.loads(json.dumps(row, default=str)))

    def read(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.rows)


class JsonlStore:
    """Append-only JSONL store; appends are serialised with a lock so worker threads can share it."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def append(self, row: dict[str, Any]) -> None:
        line = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def read(self) -> list[dict[str, Any]]:
        return read_jsonl(self.path) if self.path.exists() else []

    def clear(self) -> None:
        with self._lock:
            self.path.write_text("", encoding="utf-8")


# --------------------------------------------------------------------------- snapshots

def response_snapshot(resp: LLMResponse) -> dict[str, Any]:
    return {"text": resp.text, "model": resp.model, "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens, "tokens": resp.input_tokens + resp.output_tokens,
            "latency_ms": float(resp.latency_ms), "cost_usd": float(resp.cost_usd)}


def trajectory_snapshot(t: Trajectory) -> dict[str, Any]:
    it = sum(s.input_tokens for s in t.steps)
    ot = sum(s.output_tokens for s in t.steps)
    return {"text": t.final_answer, "model": t.model, "input_tokens": it, "output_tokens": ot,
            "tokens": it + ot, "latency_ms": float(t.total_latency_ms), "cost_usd": float(t.total_cost),
            "success": t.success, "tool_names": t.tool_names, "trajectory": t.to_dict()}


def snapshot_to_trajectory(snap: dict[str, Any], task_id: str) -> Trajectory:
    """Rebuild a Trajectory from a snapshot (a plain completion becomes a one-step trajectory)."""
    if "trajectory" in snap:
        return Trajectory.from_dict(snap["trajectory"])
    step = Step(index=0, kind="final", content=snap.get("text", ""), model=snap.get("model", ""),
                input_tokens=int(snap.get("input_tokens", 0)), output_tokens=int(snap.get("output_tokens", 0)),
                latency_ms=float(snap.get("latency_ms", 0.0)), cost_usd=float(snap.get("cost_usd", 0.0)))
    return Trajectory(task_id=task_id, steps=[step], final_answer=snap.get("text", ""),
                      success=snap.get("success"), model=snap.get("model", ""))


# --------------------------------------------------------------------------- proxy

class _ShadowBase:
    """Shared machinery: sampling, background executor, bookkeeping, never-raise shadow runs."""

    def __init__(self, sample_rate: float, seed: int, store: Any, max_workers: int,
                 shadow_timeout_ms: float | None, record_unsampled: bool) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError("sample_rate must be in [0, 1]")
        self.sample_rate = sample_rate
        self.seed = seed
        self.store = store if store is not None else MemoryStore()
        self.shadow_timeout_ms = shadow_timeout_ms
        self.record_unsampled = record_unsampled
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="shadow")
        self._pending: set[Future] = set()
        self._lock = threading.Lock()
        self.stats: dict[str, int] = {"requests": 0, "shadowed": 0, "shadow_errors": 0,
                                      "shadow_timeouts": 0}

    def _bump(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1

    def _dispatch(self, record: dict[str, Any], shadow_call: Callable[[], dict[str, Any]]) -> None:
        """Run `shadow_call` in the background; whatever happens, a record lands in the store."""
        self._bump("requests")
        if not should_shadow(record["request_id"], self.sample_rate, self.seed):
            if self.record_unsampled:
                self._safe_append(record)
            return
        record["shadowed"] = True
        self._bump("shadowed")
        fut = self._executor.submit(self._run_shadow, record, shadow_call)
        with self._lock:
            self._pending.add(fut)
        fut.add_done_callback(self._forget)

    def _forget(self, fut: Future) -> None:
        with self._lock:
            self._pending.discard(fut)

    def _run_shadow(self, record: dict[str, Any], shadow_call: Callable[[], dict[str, Any]]) -> None:
        t0 = time.perf_counter()
        try:
            snap = shadow_call()
            elapsed = (time.perf_counter() - t0) * 1000
            record["shadow"] = snap
            if self.shadow_timeout_ms is not None and elapsed > self.shadow_timeout_ms:
                record["shadow_error"] = (f"timeout: shadow took {elapsed:.0f}ms "
                                          f"(limit {self.shadow_timeout_ms:.0f}ms)")
                self._bump("shadow_timeouts")
                log.warning("shadow timeout for %s: %s", record["request_id"], record["shadow_error"])
        except Exception as e:  # noqa: BLE001 - shadow failures must never propagate
            record["shadow_error"] = f"{type(e).__name__}: {e}"
            self._bump("shadow_errors")
            log.warning("shadow failure for %s: %s", record["request_id"], record["shadow_error"])
        finally:
            self._safe_append(record)

    def _safe_append(self, record: dict[str, Any]) -> None:
        try:
            self.store.append(record)
        except Exception:
            log.exception("could not persist shadow record %s", record.get("request_id"))

    def drain(self, timeout: float | None = None) -> int:
        """Wait for in-flight shadow calls. Returns how many are still pending (0 = all done)."""
        with self._lock:
            pending = set(self._pending)
        if pending:
            wait(pending, timeout=timeout)
        with self._lock:
            return len(self._pending)

    def close(self) -> None:
        self.drain()
        self._executor.shutdown(wait=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class ShadowProxy(_ShadowBase):
    """Drop-in LLM client that mirrors a sampled slice of calls to a shadow model.

    `complete()` returns the primary response synchronously. Primary errors propagate (that is
    production behaviour); shadow errors, timeouts and store failures are logged and recorded.
    """

    def __init__(self, primary: LLMClient, shadow: LLMClient, sample_rate: float = 0.05,
                 seed: int = 0, store: Any = None, max_workers: int = 4,
                 shadow_timeout_ms: float | None = None, record_unsampled: bool = True) -> None:
        super().__init__(sample_rate, seed, store, max_workers, shadow_timeout_ms, record_unsampled)
        self.primary = primary
        self.shadow = shadow
        self.model = getattr(primary, "model", "primary")

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0,
                 max_tokens: int = 1024, seed: int | None = None, request_id: str | None = None,
                 meta: dict[str, Any] | None = None) -> LLMResponse:
        rid = request_id or uuid.uuid4().hex
        meta = dict(meta or {})
        kw = {"system": system, "temperature": temperature, "max_tokens": max_tokens, "seed": seed}
        resp = self.primary.complete(prompt, **kw)
        record = {"request_id": rid, "kind": "completion", "prompt": prompt, "system": system,
                  "meta": meta, "ts": float(meta.get("ts", time.time())), "shadowed": False,
                  "primary": response_snapshot(resp), "shadow": None, "shadow_error": None}
        self._dispatch(record, lambda: response_snapshot(self.shadow.complete(prompt, **kw)))
        return resp


AgentFn = Callable[[str, Any], Trajectory]


class ShadowAgentRunner(_ShadowBase):
    """Same idea for whole agent runs: the primary agent answers, the shadow agent runs in the background."""

    def __init__(self, primary_agent: AgentFn, shadow_agent: AgentFn, sample_rate: float = 0.05,
                 seed: int = 0, store: Any = None, max_workers: int = 4,
                 shadow_timeout_ms: float | None = None, record_unsampled: bool = True) -> None:
        super().__init__(sample_rate, seed, store, max_workers, shadow_timeout_ms, record_unsampled)
        self.primary_agent = primary_agent
        self.shadow_agent = shadow_agent

    def run(self, task_id: str, task_input: Any, *, request_id: str | None = None,
            meta: dict[str, Any] | None = None) -> Trajectory:
        rid = request_id or task_id
        meta = dict(meta or {})
        traj = self.primary_agent(task_id, task_input)
        record = {"request_id": rid, "kind": "trajectory", "task_id": task_id,
                  "prompt": task_input if isinstance(task_input, str) else json.dumps(task_input, default=str),
                  "system": "", "meta": meta, "ts": float(meta.get("ts", time.time())),
                  "shadowed": False, "primary": trajectory_snapshot(traj), "shadow": None,
                  "shadow_error": None}
        self._dispatch(record, lambda: trajectory_snapshot(self.shadow_agent(task_id, task_input)))
        return traj


# --------------------------------------------------------------------------- diffing

_PUNCT = re.compile(r"[^\w\s]")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, drop a trailing period-ish noise."""
    return " ".join(_PUNCT.sub(" ", (text or "").lower()).split())


def normalized_match(prompt: str, a: str, b: str) -> bool:
    """Default judge: agreement iff the normalised answers are identical."""
    return normalize(a) == normalize(b)


def containment_match(prompt: str, a: str, b: str) -> bool:
    """Lenient judge: agreement iff one normalised answer contains the other (terse vs verbose)."""
    na, nb = normalize(a), normalize(b)
    return bool(na) and bool(nb) and (na in nb or nb in na)


JUDGES: dict[str, Judge] = {"normalized": normalized_match, "contains": containment_match}


def edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance between two tool-name sequences (insert/delete/substitute = 1)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


CLASSES = ("same", "shadow_better", "shadow_worse", "different_unknown", "shadow_error")
Judge = Callable[[str, str, str], "bool | str"]


@dataclass
class DiffResult:
    request_id: str
    classification: str
    exact_match: bool
    normalized_match: bool
    judge_agree: bool
    tool_edit_distance: int
    primary_tools: list[str]
    shadow_tools: list[str]
    token_delta: int
    cost_delta: float
    latency_delta_ms: float
    primary_success: bool | None
    shadow_success: bool | None
    shadow_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrajectoryDiff:
    """Compare a primary/shadow pair.

    `judge(prompt, primary_answer, shadow_answer)` may return a bool (do the answers agree?) or
    one of the class labels directly. Classification rules, in order:
      1. no shadow result or a recorded error -> ``shadow_error``
      2. judge returned a label               -> that label
      3. both have success labels, they differ -> ``shadow_better`` / ``shadow_worse``
      4. judge agrees (or both succeeded)      -> ``same``
      5. otherwise                             -> ``different_unknown``
    """

    def __init__(self, judge: Judge | None = None) -> None:
        self.judge: Judge = judge or normalized_match

    def compare(self, primary: Trajectory, shadow: Trajectory | None, *, prompt: str = "",
                shadow_error: str | None = None, request_id: str = "") -> DiffResult:
        p_text = primary.final_answer
        p_tools = primary.tool_names
        if shadow is None or shadow_error:
            return DiffResult(request_id=request_id, classification="shadow_error", exact_match=False,
                              normalized_match=False, judge_agree=False, tool_edit_distance=len(p_tools),
                              primary_tools=p_tools, shadow_tools=shadow.tool_names if shadow else [],
                              token_delta=(shadow.total_tokens - primary.total_tokens) if shadow else 0,
                              cost_delta=(shadow.total_cost - primary.total_cost) if shadow else 0.0,
                              latency_delta_ms=(shadow.total_latency_ms - primary.total_latency_ms) if shadow else 0.0,
                              primary_success=primary.success, shadow_success=shadow.success if shadow else None,
                              shadow_error=shadow_error or "no shadow result")
        s_text = shadow.final_answer
        exact = p_text == s_text
        norm = normalize(p_text) == normalize(s_text)
        verdict = self.judge(prompt, p_text, s_text)
        if isinstance(verdict, str):
            if verdict not in CLASSES:
                raise ValueError(f"judge returned unknown label {verdict!r}")
            label, agree = verdict, verdict == "same"
        else:
            agree = bool(verdict)
            label = self._classify(agree, primary.success, shadow.success)
        return DiffResult(request_id=request_id, classification=label, exact_match=exact,
                          normalized_match=norm, judge_agree=agree,
                          tool_edit_distance=edit_distance(p_tools, shadow.tool_names),
                          primary_tools=p_tools, shadow_tools=shadow.tool_names,
                          token_delta=shadow.total_tokens - primary.total_tokens,
                          cost_delta=shadow.total_cost - primary.total_cost,
                          latency_delta_ms=shadow.total_latency_ms - primary.total_latency_ms,
                          primary_success=primary.success, shadow_success=shadow.success)

    @staticmethod
    def _classify(agree: bool, p_ok: bool | None, s_ok: bool | None) -> str:
        if p_ok is not None and s_ok is not None and p_ok != s_ok:
            return "shadow_better" if s_ok else "shadow_worse"
        if agree or (p_ok is True and s_ok is True):
            return "same"
        return "different_unknown"

    def from_record(self, rec: dict[str, Any], expected_key: str = "expected") -> DiffResult:
        """Diff a stored record. For plain completions an `expected` answer in `meta` supplies
        success labels (normalised containment), so better/worse can be classified."""
        rid = rec["request_id"]
        primary = snapshot_to_trajectory(rec["primary"], rid)
        shadow = snapshot_to_trajectory(rec["shadow"], rid) if rec.get("shadow") else None
        expected = (rec.get("meta") or {}).get(expected_key)
        if expected is not None and rec.get("kind", "completion") == "completion":
            exp = normalize(str(expected))
            primary.success = exp in normalize(primary.final_answer)
            if shadow is not None:
                shadow.success = exp in normalize(shadow.final_answer)
        return self.compare(primary, shadow, prompt=rec.get("prompt", ""),
                            shadow_error=rec.get("shadow_error"), request_id=rid)
