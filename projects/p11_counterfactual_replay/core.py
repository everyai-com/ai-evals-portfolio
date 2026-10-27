"""Counterfactual Replay Debugger.

Record every LLM hop and tool call of an agent into an ``evalkit.trajectory.Trajectory`` (one
``Step`` per hop, each tagged with a ``node_id`` and a content hash of its inputs), then
re-execute the same agent with one node swapped and watch what changes downstream.

* ``Recorder``  — an ``ExecutionContext`` that runs hops live (through an ``LLMClient`` and a
  tool registry) and records them.
* ``Replayer``  — an ``ExecutionContext`` that serves every hop *before* the override point
  from the recording (zero model/tool calls), applies the override at that node, and executes
  everything *after* it live (optionally with a different LLM client or tool implementation).
  ``mode="memo"`` additionally reuses a recorded output downstream whenever the hop's input
  hash is unchanged, so only hops whose inputs actually changed are re-executed.
* ``diff``      — step-by-step comparison of two trajectories (first divergence, changed
  steps, final-answer divergence) rendered as markdown.
* ``bisect``    — tries, in order, overriding each node of a bad run with the golden run's
  output for that node; the earliest node whose override repairs the final answer is the
  node that introduced the error.

An *agent* is any callable ``agent(ctx) -> str`` that only talks to the world through
``ctx.llm(node_id, prompt, system="")`` and ``ctx.tool(node_id, name, **args)``.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from evalkit.io import read_jsonl, stable_hash, write_jsonl
from evalkit.llm import LLMClient
from evalkit.report import md_table
from evalkit.trajectory import Step, Trajectory

Tool = Callable[..., Any]
ToolRegistry = dict[str, Tool]


class ExecutionContext(Protocol):
    def llm(self, node_id: str, prompt: str, system: str = "") -> str: ...
    def tool(self, node_id: str, name: str, **args: Any) -> Any: ...


Agent = Callable[[ExecutionContext], str]


# ---------------------------------------------------------------------------
# helpers over the shared Trajectory schema
# ---------------------------------------------------------------------------

def input_hash(kind: str, inputs: dict[str, Any]) -> str:
    """Content hash of a hop's inputs (``evalkit.io.stable_hash`` over kind + inputs)."""
    return stable_hash({"kind": kind, "inputs": inputs})


def node_id_of(step: Step) -> str:
    return str(step.meta.get("node_id", step.index))


def node_output(step: Step) -> Any:
    """The output a hop produced: LLM text for ``llm`` steps, tool result for ``tool`` steps."""
    if step.kind == "tool" and step.tool_call is not None:
        return step.tool_call.result
    return step.content


def hops(traj: Trajectory) -> list[Step]:
    return [s for s in traj.steps if s.kind in ("llm", "tool")]


def load_run(path: str | Path) -> Trajectory:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{path}: empty run file")
    return Trajectory.from_dict(rows[0])


def save_run(traj: Trajectory, path: str | Path) -> Path:
    path = Path(path)
    write_jsonl(path, [traj.to_dict()])
    return path


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------

class Recorder:
    """Execution context that runs every hop live and records it."""

    def __init__(self, llm: LLMClient, tools: ToolRegistry, task_id: str = "run",
                 meta: dict[str, Any] | None = None):
        self.client = llm
        self.tools = tools
        self.traj = Trajectory(task_id=task_id, model=getattr(llm, "model", ""), meta=dict(meta or {}))

    def llm(self, node_id: str, prompt: str, system: str = "") -> str:
        inputs = {"prompt": prompt, "system": system}
        resp = self.client.complete(prompt, system=system)
        self.traj.add_llm(resp.text, model=resp.model, input_tokens=resp.input_tokens,
                          output_tokens=resp.output_tokens, latency_ms=resp.latency_ms,
                          cost_usd=resp.cost_usd,
                          meta={"node_id": node_id, "inputs": inputs,
                                "input_hash": input_hash("llm", inputs), "source": "live"})
        return resp.text

    def tool(self, node_id: str, name: str, **args: Any) -> Any:
        if name not in self.tools:
            raise KeyError(f"unknown tool {name!r}")
        inputs = {"name": name, "args": args}
        result = self.tools[name](**args)
        self.traj.add_tool(name, args, result=result,
                           meta={"node_id": node_id, "inputs": inputs,
                                 "input_hash": input_hash("tool", inputs), "source": "live"})
        return result

    def run(self, agent: Agent) -> Trajectory:
        answer = agent(self)
        self.traj.finish(answer)
        return self.traj

    def save(self, run_dir: str | Path) -> Path:
        return save_run(self.traj, Path(run_dir) / f"{self.traj.task_id}.jsonl")


# ---------------------------------------------------------------------------
# Replayer
# ---------------------------------------------------------------------------

class Replayer:
    """Execution context that replays a recording with one or more nodes swapped.

    Parameters
    ----------
    recorded      : the trajectory to replay.
    overrides     : ``{node_id: new_output}`` — the node's output is replaced verbatim.
    tool_overrides: ``{node_id: callable}`` — the node is re-executed with this implementation.
    llm, tools    : live backends for hops after the override point (defaults: none; a live
                    hop without a backend raises).
    mode          : ``"live"`` re-executes every hop after the override point; ``"memo"``
                    reuses the recorded output when the hop's input hash is unchanged.
    """

    def __init__(self, recorded: Trajectory, overrides: dict[str, Any] | None = None, *,
                 tool_overrides: dict[str, Tool] | None = None, llm: LLMClient | None = None,
                 tools: ToolRegistry | None = None, mode: str = "live"):
        if mode not in ("live", "memo"):
            raise ValueError("mode must be 'live' or 'memo'")
        self.recorded = recorded
        self.overrides = dict(overrides or {})
        self.tool_overrides = dict(tool_overrides or {})
        self.client = llm
        self.tools = tools or {}
        self.mode = mode
        self._hops = hops(recorded)
        self._by_node = {node_id_of(s): i for i, s in enumerate(self._hops)}
        unknown = (set(self.overrides) | set(self.tool_overrides)) - set(self._by_node)
        if unknown:
            raise KeyError(f"override targets not in recording: {sorted(unknown)}")
        idxs = [self._by_node[n] for n in (*self.overrides, *self.tool_overrides)]
        self.override_index = min(idxs) if idxs else len(self._hops)
        self._pos = 0
        self.live_calls = 0
        self.cache_hits = 0
        self.traj = Trajectory(task_id=f"{recorded.task_id}+replay",
                               model=getattr(llm, "model", recorded.model),
                               meta={"replay_of": recorded.task_id, "overrides": self.overrides,
                                     "tool_overrides": sorted(self.tool_overrides), "mode": mode})

    # ---- bookkeeping ----
    def _locate(self, node_id: str) -> tuple[int | None, Step | None]:
        """Find the recorded hop for ``node_id`` at or after the cursor."""
        for i in range(self._pos, len(self._hops)):
            if node_id_of(self._hops[i]) == node_id:
                self._pos = i + 1
                return i, self._hops[i]
        return None, None

    def _record(self, kind: str, node_id: str, inputs: dict[str, Any], output: Any, source: str,
                model: str = "", **kw: Any) -> None:
        meta = {"node_id": node_id, "inputs": inputs, "input_hash": input_hash(kind, inputs),
                "source": source}
        if kind == "llm":
            self.traj.add_llm(str(output), model=model, meta=meta, **kw)
        else:
            self.traj.add_tool(inputs["name"], inputs["args"], result=output, meta=meta)

    def _decide(self, node_id: str, kind: str, inputs: dict[str, Any]) -> tuple[str, Any]:
        """Return (source, output) where source is cache/override/None (None = execute live)."""
        i, step = self._locate(node_id)
        if node_id in self.overrides:
            return "override", self.overrides[node_id]
        if node_id in self.tool_overrides:
            return "tool_override", None
        if step is None:
            return "live", None
        if i < self.override_index:
            return "cache", node_output(step)
        if self.mode == "memo" and step.meta.get("input_hash") == input_hash(kind, inputs):
            return "cache", node_output(step)
        return "live", None

    # ---- ExecutionContext ----
    def llm(self, node_id: str, prompt: str, system: str = "") -> str:
        inputs = {"prompt": prompt, "system": system}
        source, out = self._decide(node_id, "llm", inputs)
        if source in ("cache", "override"):
            self.cache_hits += source == "cache"
            self._record("llm", node_id, inputs, out, source, model=self.recorded.model)
            return str(out)
        if self.client is None:
            raise RuntimeError(f"hop {node_id!r} needs a live LLM but none was provided")
        resp = self.client.complete(prompt, system=system)
        self.live_calls += 1
        self._record("llm", node_id, inputs, resp.text, "live", model=resp.model,
                     input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                     latency_ms=resp.latency_ms, cost_usd=resp.cost_usd)
        return resp.text

    def tool(self, node_id: str, name: str, **args: Any) -> Any:
        inputs = {"name": name, "args": args}
        source, out = self._decide(node_id, "tool", inputs)
        if source in ("cache", "override"):
            self.cache_hits += source == "cache"
            self._record("tool", node_id, inputs, out, source)
            return out
        fn = self.tool_overrides.get(node_id) or self.tools.get(name)
        if fn is None:
            raise RuntimeError(f"hop {node_id!r} needs a live tool {name!r} but none was provided")
        out = fn(**args)
        self.live_calls += 1
        self._record("tool", node_id, inputs, out, source if source == "tool_override" else "live")
        return out

    def run(self, agent: Agent) -> Trajectory:
        answer = agent(self)
        self.traj.finish(answer)
        self.traj.meta["cache_hits"] = self.cache_hits
        self.traj.meta["live_calls"] = self.live_calls
        return self.traj


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------

@dataclass
class DiffRow:
    index: int
    node_a: str | None
    node_b: str | None
    output_a: Any
    output_b: Any
    source_b: str = ""

    @property
    def changed(self) -> bool:
        return self.node_a != self.node_b or self.output_a != self.output_b


@dataclass
class DiffResult:
    rows: list[DiffRow]
    final_a: str
    final_b: str
    first_divergence: int | None = None

    @property
    def changed_indices(self) -> list[int]:
        return [r.index for r in self.rows if r.changed]

    @property
    def final_diverges(self) -> bool:
        return self.final_a != self.final_b

    def to_dict(self) -> dict[str, Any]:
        return {"first_divergence": self.first_divergence, "changed_steps": self.changed_indices,
                "final_diverges": self.final_diverges, "final_a": self.final_a, "final_b": self.final_b,
                "rows": [{"index": r.index, "node_a": r.node_a, "node_b": r.node_b,
                          "output_a": r.output_a, "output_b": r.output_b, "source_b": r.source_b,
                          "changed": r.changed} for r in self.rows]}

    def to_markdown(self, width: int = 72) -> str:
        def short(v: Any) -> str:
            s = v if isinstance(v, str) else json.dumps(v, sort_keys=True, default=str)
            s = s.replace("\n", " ").replace("|", "\\|")
            return s if len(s) <= width else s[: width - 1] + "…"
        rows = [[r.index, r.node_a or "-", r.node_b or "-", r.source_b or "-",
                 "**changed**" if r.changed else "same", short(r.output_a), short(r.output_b)]
                for r in self.rows]
        lines = ["## Trajectory diff", "",
                 md_table(["#", "node (A)", "node (B)", "B source", "status", "output A", "output B"], rows),
                 "",
                 f"- first divergence: {'none' if self.first_divergence is None else f'step {self.first_divergence} ({self.rows[self.first_divergence].node_b})'}",
                 f"- changed steps: {self.changed_indices or 'none'}",
                 f"- final answer A: `{self.final_a}`",
                 f"- final answer B: `{self.final_b}`",
                 f"- final answers {'DIVERGE' if self.final_diverges else 'agree'}"]
        return "\n".join(lines)


def diff(a: Trajectory, b: Trajectory) -> DiffResult:
    ha, hb = hops(a), hops(b)
    rows: list[DiffRow] = []
    first: int | None = None
    for i in range(max(len(ha), len(hb))):
        sa = ha[i] if i < len(ha) else None
        sb = hb[i] if i < len(hb) else None
        row = DiffRow(index=i, node_a=node_id_of(sa) if sa else None, node_b=node_id_of(sb) if sb else None,
                      output_a=node_output(sa) if sa else None, output_b=node_output(sb) if sb else None,
                      source_b=str(sb.meta.get("source", "")) if sb else "")
        if row.changed and first is None:
            first = i
        rows.append(row)
    return DiffResult(rows=rows, final_a=a.final_answer, final_b=b.final_answer, first_divergence=first)


# ---------------------------------------------------------------------------
# bisect
# ---------------------------------------------------------------------------

@dataclass
class BisectResult:
    culprit: str | None
    culprit_index: int | None
    expected: str
    tried: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"culprit": self.culprit, "culprit_index": self.culprit_index,
                "expected": self.expected, "tried": self.tried}

    def to_markdown(self) -> str:
        rows = [[t["index"], t["node"], t["final_answer"], "**fixed**" if t["fixed"] else "still wrong"]
                for t in self.tried]
        head = (f"## Bisect: culprit = `{self.culprit}` (hop {self.culprit_index})" if self.culprit
                else "## Bisect: no single-node override repairs the run")
        return "\n".join([head, "", md_table(["#", "override node -> golden", "final answer", "result"], rows)])


def bisect(bad: Trajectory, golden: Trajectory, agent: Agent, *, llm: LLMClient | None = None,
           tools: ToolRegistry | None = None, expected: str | None = None) -> BisectResult:
    """Find the earliest node whose override to the golden output fixes the final answer.

    Replays run in ``memo`` mode so that hops whose inputs are unchanged keep their recorded
    (possibly corrupted) outputs; only hops downstream of a real change are re-executed. This
    is what isolates the *source* of a bad value instead of the first node in the graph.
    """
    expected = golden.final_answer if expected is None else expected
    golden_out = {node_id_of(s): node_output(s) for s in hops(golden)}
    tried: list[dict[str, Any]] = []
    for i, step in enumerate(hops(bad)):
        nid = node_id_of(step)
        if nid not in golden_out:
            continue
        rp = Replayer(bad, {nid: golden_out[nid]}, llm=llm, tools=tools, mode="memo")
        out = rp.run(agent)
        fixed = out.final_answer == expected
        tried.append({"index": i, "node": nid, "final_answer": out.final_answer, "fixed": fixed,
                      "live_calls": rp.live_calls})
        if fixed:
            return BisectResult(culprit=nid, culprit_index=i, expected=expected, tried=tried)
    return BisectResult(culprit=None, culprit_index=None, expected=expected, tried=tried)
