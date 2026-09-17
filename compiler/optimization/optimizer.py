"""Apply an allowlisted plan's graph passes."""

from __future__ import annotations

import time
from typing import Any

import onnx

from compiler.optimization.passes import apply_pass, graph_ir_snapshot
from compiler.planner.plan import Plan, Strategy


def apply_strategy(model: onnx.ModelProto, strategy: Strategy) -> onnx.ModelProto:
    updated = onnx.ModelProto()
    updated.CopyFrom(model)
    for name in strategy.passes:
        updated = apply_pass(name, updated)
    return updated


def apply_plan(model: onnx.ModelProto, plan: Plan) -> tuple[onnx.ModelProto, list[dict[str, Any]]]:
    updated = onnx.ModelProto()
    updated.CopyFrom(model)
    applied: list[dict[str, Any]] = []
    for step in plan.steps:
        atom = str(step["atom"])
        params = dict(step.get("params") or {})
        before = graph_ir_snapshot(updated)
        t0 = time.perf_counter()
        updated = apply_pass(atom, updated, params)
        ms = (time.perf_counter() - t0) * 1000.0
        after = graph_ir_snapshot(updated)
        before_ops = before["op_counts"] if isinstance(before["op_counts"], dict) else {}
        after_ops = after["op_counts"] if isinstance(after["op_counts"], dict) else {}
        applied.append(
            {
                "atom": atom,
                "params": params,
                "ms": ms,
                "nodes_before": int(before["node_count"]),
                "nodes_after": int(after["node_count"]),
                "op_delta": {
                    key: int(after_ops.get(key, 0)) - int(before_ops.get(key, 0))
                    for key in sorted(set(before_ops) | set(after_ops))
                    if int(after_ops.get(key, 0)) != int(before_ops.get(key, 0))
                },
            }
        )
    return updated, applied


def prepare_for_ort(model: onnx.ModelProto) -> onnx.ModelProto:
    """Identity: FusedConv is a real ORT CPU op. Kept so old imports do not break."""

    return model
