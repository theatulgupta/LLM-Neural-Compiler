"""Prompts that constrain an LLM to the strategy allowlist."""

from __future__ import annotations

import json

from compiler.graph.graph_summary import GraphSummary
from compiler.schema_validate import LLM_PROPOSAL_SCHEMA_PATH, load_schema
from compiler.strategies import ALLOWED_STRATEGIES

SYSTEM_PROMPT = """You are a neural-compiler advisor for ONNX models targeting UAV edge inference.

You may ONLY pick one strategy from the allowlist. Do not invent passes, custom
operators, quantization schemes, TensorRT tactics, latency, FPS, IPS, or accuracy
numbers. Those metrics exist only after the compiler runs ONNX Runtime.

Reply with a single JSON object matching the provided schema and nothing else.
"""


def allowlist_block() -> str:
    lines = ["Allowlisted strategies:"]
    for name, strategy in ALLOWED_STRATEGIES.items():
        lines.append(f"- {name}: {strategy.description}")
    return "\n".join(lines)


def proposal_schema_json() -> str:
    return json.dumps(load_schema(LLM_PROPOSAL_SCHEMA_PATH), indent=2)


def user_prompt(summary: GraphSummary) -> str:
    payload = dict(summary.to_dict())
    # Keep the prompt factual; do not include prior latency numbers.
    return (
        f"{allowlist_block()}\n\n"
        f"Proposal JSON schema:\n{proposal_schema_json()}\n\n"
        f"Graph summary JSON:\n{json.dumps(payload, default=str)}\n\n"
        'Return JSON: {"strategy": "<allowlisted name>", "rationale": "<short, no metrics>", '
        '"source": "groq"}.'
    )
