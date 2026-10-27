"""evalkit: shared core for the AI Evals Engineering portfolio.

Modules
-------
io          JSONL read/write, stable hashing, dataset versioning helpers
metrics     precision/recall/F1, Cohen's kappa, Wilson + bootstrap CIs, IR metrics
stats       bootstrap and paired tests for model-vs-model deltas
llm         LLMClient protocol, deterministic MockLLM, optional AnthropicLLM
trajectory  ToolCall / Step / Trajectory dataclasses shared by agent projects
report      markdown table helpers
pytest_plugin  --eval-dataset JSONL loader + end-of-run metric report
"""
from evalkit.io import read_jsonl, stable_hash, write_jsonl
from evalkit.llm import LLMClient, LLMResponse, MockLLM, get_client
from evalkit.trajectory import Step, ToolCall, Trajectory

__all__ = [
    "LLMClient",
    "LLMResponse",
    "MockLLM",
    "Step",
    "ToolCall",
    "Trajectory",
    "get_client",
    "read_jsonl",
    "stable_hash",
    "write_jsonl",
]
