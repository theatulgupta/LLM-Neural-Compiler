"""Apply a Plan to GraphIR. ONNX is the only engine wired today."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import onnx

from compiler.errors import CompilerError
from compiler.graph.graph_loader import LoadedGraph, from_onnx_bytes
from compiler.optimization.optimizer import apply_plan, apply_strategy
from compiler.planner.plan import Plan, Strategy

OnnxPassFn = Callable[[onnx.ModelProto, Plan], tuple[onnx.ModelProto, list[dict[str, Any]]]]

_IR_PLAN_ENGINES: dict[str, OnnxPassFn] = {
    "onnx": apply_plan,
}


def register_ir_engine(ir: str, apply: OnnxPassFn) -> None:
    _IR_PLAN_ENGINES[ir] = apply


def apply_plan_on_graph(loaded: LoadedGraph, plan: Plan) -> tuple[LoadedGraph, list[dict[str, Any]]]:
    engine = _IR_PLAN_ENGINES.get(loaded.ir)
    if engine is None:
        raise CompilerError(f"no pass engine for IR {loaded.ir!r}; known={sorted(_IR_PLAN_ENGINES)}")
    model, steps = engine(loaded.model, plan)
    data = model.SerializeToString()
    updated = from_onnx_bytes(
        model,
        data,
        source=loaded.source,
        origin_format=loaded.origin_format,
        sha256=loaded.sha256,
    )
    return updated, steps


def apply_strategy_on_graph(loaded: LoadedGraph, strategy: Strategy) -> LoadedGraph:
    if loaded.ir != "onnx":
        raise CompilerError(f"no strategy engine for IR {loaded.ir!r}")
    model = apply_strategy(loaded.model, strategy)
    data = model.SerializeToString()
    return from_onnx_bytes(
        model,
        data,
        source=loaded.source,
        origin_format=loaded.origin_format,
        sha256=loaded.sha256,
    )
