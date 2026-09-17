"""Build the factual JSON the LLM is allowed to see."""

from __future__ import annotations

import json
from typing import Any

from compiler.graph.graph_summary import GraphSummary
from compiler.hardware.profile import HardwareProfile
from compiler.planner.atoms import ATOMS, PASS_ATOMS


def build_context(
    summary: GraphSummary,
    hardware: HardwareProfile,
    constraints: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    op_counts = dict(sorted(summary.op_counts.items(), key=lambda item: (-item[1], item[0]))[:15])
    atoms = []
    for name in PASS_ATOMS:
        atom = ATOMS[name]
        atoms.append(
            {
                "name": name,
                "requires": list(atom.requires),
                "hardware_requires": list(atom.hardware_requires),
                "numerics": atom.numerics,
            }
        )
    history_lines = []
    for row in (history or [])[:5]:
        history_lines.append(
            {
                "plan_id": row.get("plan_id") or row.get("strategy") or row.get("kind"),
                "p50_ms": row.get("p50_ms"),
                "passed": row.get("passed"),
            }
        )
    return {
        "graph": {
            "sha256": summary.sha256,
            "node_count": summary.node_count,
            "opset": summary.opset,
            "op_counts_top": op_counts,
            "flops_total": summary.flops_total,
            "patterns": summary.patterns,
            "memory": summary.memory,
            "inputs": summary.inputs,
            "outputs": summary.outputs,
            "notes": list(summary.notes),
            "static_shapes": summary.static_shapes,
        },
        "hardware": hardware.to_dict(),
        "constraints": dict(constraints or {}),
        "allowlisted_atoms": atoms,
        "history": history_lines,
    }


def render_user_prompt(context: dict[str, Any]) -> str:
    return (
        "You are planning allowlisted ONNX graph passes for a CPU UAV companion.\n"
        "Do not invent latency, FPS, IPS, or accuracy numbers.\n"
        "Justify each step with a graph pattern from the context.\n\n"
        f"{json.dumps(context, default=str, indent=2)}\n"
    )
