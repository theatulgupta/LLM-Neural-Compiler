"""Apply an allowlisted strategy's passes to an ONNX model."""

from __future__ import annotations

import onnx

from compiler.optimization.passes import apply_pass
from compiler.strategies import Strategy


def apply_strategy(model: onnx.ModelProto, strategy: Strategy) -> onnx.ModelProto:
    updated = onnx.ModelProto()
    updated.CopyFrom(model)
    for name in strategy.passes:
        updated = apply_pass(name, updated)
    return updated
