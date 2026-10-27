"""Shared trajectory schema for agent evals (projects 1, 2, 8, 9, 11, 13)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Step:
    """One hop of an agent loop: an LLM turn that may emit tool calls."""
    index: int
    kind: str                      # "llm" | "tool" | "final"
    content: str = ""              # LLM text or final answer
    tool_call: ToolCall | None = None
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class Trajectory:
    task_id: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str = ""
    success: bool | None = None
    model: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- derived ----
    @property
    def tool_calls(self) -> list[ToolCall]:
        return [s.tool_call for s in self.steps if s.tool_call is not None]

    @property
    def tool_names(self) -> list[str]:
        return [tc.name for tc in self.tool_calls]

    @property
    def total_cost(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.steps)

    # ---- (de)serialisation ----
    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "steps": [s.to_dict() for s in self.steps],
                "final_answer": self.final_answer, "success": self.success,
                "model": self.model, "meta": self.meta}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Trajectory:
        steps = []
        for s in d.get("steps", []):
            tc = s.get("tool_call")
            steps.append(Step(index=s["index"], kind=s["kind"], content=s.get("content", ""),
                              tool_call=ToolCall(**tc) if tc else None, model=s.get("model", ""),
                              input_tokens=s.get("input_tokens", 0), output_tokens=s.get("output_tokens", 0),
                              latency_ms=s.get("latency_ms", 0.0), cost_usd=s.get("cost_usd", 0.0),
                              meta=s.get("meta", {})))
        return cls(task_id=d["task_id"], steps=steps, final_answer=d.get("final_answer", ""),
                   success=d.get("success"), model=d.get("model", ""), meta=d.get("meta", {}))

    @classmethod
    def from_json(cls, s: str) -> Trajectory:
        return cls.from_dict(json.loads(s))

    def add_tool(self, name: str, args: dict[str, Any], result: Any = None, error: str | None = None,
                 **kw: Any) -> Trajectory:
        self.steps.append(Step(index=len(self.steps), kind="tool",
                               tool_call=ToolCall(name=name, args=args, result=result, error=error), **kw))
        return self

    def add_llm(self, content: str, **kw: Any) -> Trajectory:
        self.steps.append(Step(index=len(self.steps), kind="llm", content=content, **kw))
        return self

    def finish(self, answer: str, success: bool | None = None) -> Trajectory:
        self.steps.append(Step(index=len(self.steps), kind="final", content=answer))
        self.final_answer = answer
        self.success = success
        return self
