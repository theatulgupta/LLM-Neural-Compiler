"""Prompts that constrain an LLM to the plan schema."""

from __future__ import annotations

import json

from compiler.graph.graph_summary import GraphSummary
from compiler.hardware.profile import probe_hardware
from compiler.llm.context_builder import build_context, render_user_prompt
from compiler.planner.atoms import PASS_ATOMS
from compiler.schema_validate import LLM_PLAN_SCHEMA_PATH, load_schema
from compiler.strategies import ALLOWED_STRATEGIES

SYSTEM_PROMPT = """You are a neural-compiler planner for ONNX computation graphs.

You propose an ordered list of allowlisted atoms (graph passes) plus ORT session
options. You do not edit the DAG yourself. Do not invent passes, custom operators,
TensorRT tactics, latency, FPS, IPS, or accuracy numbers. Those metrics exist only
after the compiler rewrites the DAG and runs ONNX Runtime.

Reply with a single JSON object matching the provided schema and nothing else.
Each step must be justified by a pattern in the graph summary.
"""


def allowlist_block() -> str:
    lines = ["Allowlisted presets (you may also compose atoms):"]
    for name, strategy in ALLOWED_STRATEGIES.items():
        lines.append(f"- {name}: {strategy.description}")
    lines.append("Allowlisted pass atoms: " + ", ".join(PASS_ATOMS))
    return "\n".join(lines)


def proposal_schema_json() -> str:
    return json.dumps(load_schema(LLM_PLAN_SCHEMA_PATH), indent=2)


def user_prompt(summary: GraphSummary) -> str:
    context = build_context(summary, probe_hardware(), {}, [])
    return (
        f"{allowlist_block()}\n\n"
        f"Plan JSON schema:\n{proposal_schema_json()}\n\n"
        f"{render_user_prompt(context)}"
    )
