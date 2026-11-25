"""Prompts that constrain an LLM to the multi-dimensional strategy allowlist."""

from __future__ import annotations

import json
from typing import Any

from compiler.graph.graph_summary import GraphSummary
from compiler.strategies import (
    ALLOWED_ENABLE_MEM_PATTERN,
    ALLOWED_EXECUTION_MODES,
    ALLOWED_GRAPH_OPT_LEVELS,
    ALLOWED_INTER_OP_THREADS,
    ALLOWED_INTRA_OP_THREADS,
    ALLOWED_PRECISIONS,
    ALLOWED_STRATEGIES,
    available_execution_providers,
)

SYSTEM_PROMPT = """You are a neural-compiler advisor for ONNX models targeting real-time UAV edge inference.

You MUST choose only from the allowlisted strategy dimensions and available execution providers listed
in the user prompt. Do NOT invent custom compiler flags, TensorRT tactics, or unavailable providers.
Return ONLY valid JSON matching:
{
  "strategy": "baseline" | "ort_basic" | "ort_extended" | "ort_all",
  "config": {
    "execution_provider": "<one of the available execution providers listed below>",
    "graph_optimization_level": "disable" | "basic" | "extended" | "all",
    "precision": "fp32" | "int8_dynamic" | "int8_static",
    "execution_mode": "sequential" | "parallel",
    "intra_op_num_threads": 1 | 2 | 4 | 8,
    "inter_op_num_threads": 1 | 2,
    "enable_mem_pattern": true | false
  },
  "rationale": "<concise technical justification>",
  "source": "<advisor name>"
}
"""


def allowlist_block() -> str:
    avail_eps = list(available_execution_providers())
    lines = ["Allowlisted preset strategies:"]
    for name, strategy in ALLOWED_STRATEGIES.items():
        lines.append(f"- {name}: {strategy.description}")
    lines.extend(
        [
            "",
            "Allowlisted multi-dimensional configuration space:",
            f"- execution_provider (installed on this host): {avail_eps}",
            f"- graph_optimization_level: {list(ALLOWED_GRAPH_OPT_LEVELS)}",
            f"- precision: {list(ALLOWED_PRECISIONS[:3])} (fp16 only if converter available)",
            f"- execution_mode: {list(ALLOWED_EXECUTION_MODES)}",
            f"- intra_op_num_threads: {list(ALLOWED_INTRA_OP_THREADS)}",
            f"- inter_op_num_threads: {list(ALLOWED_INTER_OP_THREADS)}",
            f"- enable_mem_pattern: {list(ALLOWED_ENABLE_MEM_PATTERN)}",
        ]
    )
    return "\n".join(lines)


def format_history_feedback(history_records: list[dict[str, Any]] | None, *, limit: int = 5) -> str:
    if not history_records:
        return "No prior benchmark runs recorded for this model/host."
    recent = history_records[-limit:]
    lines = ["Recent historical benchmark feedback (closed-loop context):"]
    for rec in recent:
        strat = rec.get("strategy", {})
        comp = rec.get("compile", {})
        bench = rec.get("benchmark") or {}
        lat = bench.get("latency_ms") or {}
        fid = rec.get("fidelity") or {}
        lines.append(
            f"- strategy={strat.get('strategy')} config={strat.get('config')} "
            f"compile_ok={comp.get('ok')} mean_ms={lat.get('mean')} p95_ms={lat.get('p95')} "
            f"cos_sim={fid.get('cosine_similarity')}"
        )
    return "\n".join(lines)


def user_prompt(
    summary: GraphSummary,
    *,
    hardware: str = "ARM64 Linux CPU (4-8 cores, shared with PX4/ROS 2 flight stack)",
    objective: str = "Minimize P95 inference latency under 80 ms while preserving accuracy (cosine_similarity >= 0.99)",
    history_feedback: list[dict[str, Any]] | None = None,
) -> str:
    return (
        f"{allowlist_block()}\n\n"
        f"Target Hardware: {hardware}\n"
        f"Objective: {objective}\n\n"
        f"Graph summary JSON:\n{json.dumps(summary.to_dict(), indent=2)}\n\n"
        f"{format_history_feedback(history_feedback)}\n\n"
        "Respond with a single JSON object conforming strictly to the schema."
    )
