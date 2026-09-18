"""Build the LLM chat messages from graph/hardware facts.

The system string is fixed. The user string changes with this graph, this
board, and the last compile rows. The HTTP client only sends the messages;
it does not write them.
"""

from __future__ import annotations

import json
from typing import Any

from compiler.graph.graph_summary import GraphSummary
from compiler.hardware.profile import HardwareProfile, probe_hardware
from compiler.llm.context_builder import build_context
from compiler.planner.atoms import ATOMS, PASS_ATOMS
from compiler.planner.plan import ALLOWED_STRATEGIES
from compiler.schema import LLM_PLAN_SCHEMA_PATH, load_schema

SYSTEM_PROMPT = """You are a neural-compiler planner.

Propose allowlisted graph-pass atoms plus session options. Do not edit the DAG.
Do not invent passes, latency, FPS, IPS, or accuracy. Reply with one JSON object.
"""


def _applicable(summary: GraphSummary, context: dict[str, Any]) -> tuple[list[str], list[str]]:
    patterns = summary.patterns or {}
    hardware = context.get("hardware") or {}
    features = set(hardware.get("features") or [])
    fp16 = bool(hardware.get("fp16_execution"))
    yes: list[str] = []
    no: list[str] = []
    for name in PASS_ATOMS:
        atom = ATOMS[name]
        missing = [item for item in atom.requires if int(patterns.get(item, 0) or 0) == 0]
        for req in atom.hardware_requires:
            if req == "fp16_execution" and not fp16:
                missing.append(req)
            elif req != "fp16_execution" and req not in features:
                missing.append(req)
        if missing:
            no.append(f"{name} ({', '.join(missing)})")
        else:
            yes.append(name)
    return yes, no


def _format_feedback(feedback: list[dict[str, Any]]) -> str:
    lines = ["Previous attempts"]
    for index, item in enumerate(feedback, 1):
        n = item.get("n", index)
        outcome = item.get("outcome", "")
        detail = item.get("detail", "")
        lines.append(f"- attempt {n} {outcome}: {detail}")
    lines.append("Reply again with a corrected plan. Only allowlisted atoms. Do not repeat rejected atoms.")
    return "\n".join(lines)


def build_messages(
    summary: GraphSummary,
    context: dict[str, Any] | None = None,
    hardware: HardwareProfile | None = None,
    constraints: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    feedback: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    ctx = (
        context
        if context is not None
        else build_context(summary, hardware or probe_hardware(), constraints, history)
    )
    yes, no = _applicable(summary, ctx)
    history_rows = ctx.get("history") or []
    if history_rows:
        hist = "\n".join(
            f"- {row.get('plan_id')} p50_ms={row.get('p50_ms')} passed={row.get('passed')}"
            for row in history_rows[:5]
        )
    else:
        hist = "none"
    presets = "\n".join(f"- {name}: {s.description}" for name, s in ALLOWED_STRATEGIES.items())
    user = (
        "Plan allowlisted passes for a UAV companion. No invented metrics.\n\n"
        f"IR origin={summary.origin_format} ir={summary.ir}\n\n"
        f"Presets:\n{presets}\n"
        f"Atoms: {', '.join(PASS_ATOMS)}\n"
        f"Applicable: {', '.join(yes) or 'none'}\n"
        f"Skip unless the pattern exists: {'; '.join(no) or 'none'}\n\n"
        f"Constraints: {json.dumps(ctx.get('constraints') or {}, default=str)}\n"
        f"History:\n{hist}\n\n"
        f"Schema:\n{json.dumps(load_schema(LLM_PLAN_SCHEMA_PATH), indent=2)}\n\n"
        f"{json.dumps(ctx, default=str, indent=2)}\n"
    )
    if feedback:
        user = f"{user}\n{_format_feedback(feedback)}\n"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def user_prompt(summary: GraphSummary) -> str:
    return build_messages(summary)[1]["content"]
