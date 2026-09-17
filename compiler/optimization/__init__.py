"""IR pass-engine registry plus ONNX pass implementations."""

from __future__ import annotations

from compiler.optimization.engine import apply_plan_on_graph, apply_strategy_on_graph, register_ir_engine
from compiler.optimization.optimizer import apply_plan, apply_strategy, prepare_for_ort
from compiler.optimization.passes import apply_pass

__all__ = [
    "apply_pass",
    "apply_plan",
    "apply_plan_on_graph",
    "apply_strategy",
    "apply_strategy_on_graph",
    "prepare_for_ort",
    "register_ir_engine",
]
