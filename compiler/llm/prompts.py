"""Prompts that constrain an LLM to the strategy allowlist."""

from __future__ import annotations

from compiler.graph.graph_summary import GraphSummary
from compiler.strategies import ALLOWED_STRATEGIES

SYSTEM_PROMPT = """You are a neural-compiler advisor for ONNX models targeting UAV edge inference.

You may ONLY recommend one strategy name from the allowlist. Do not invent passes,
custom operators, quantization schemes, or TensorRT tactics that are not listed.
Reply with a single strategy name on the first line, then a short rationale.
"""


def allowlist_block() -> str:
    lines = ["Allowlisted strategies:"]
    for name, strategy in ALLOWED_STRATEGIES.items():
        lines.append(f"- {name}: {strategy.description}")
    return "\n".join(lines)


def user_prompt(summary: GraphSummary) -> str:
    return (
        f"{allowlist_block()}\n\n"
        f"Graph summary JSON:\n{summary.to_dict()}\n\n"
        "Pick exactly one allowlisted strategy name."
    )
