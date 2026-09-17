"""Apply an allowlisted strategy's graph passes, then prepare a backend-legal ONNX."""

from __future__ import annotations

import onnx

from compiler.optimization.passes import apply_pass, expand_fused_conv
from compiler.strategies import Strategy


def apply_strategy(model: onnx.ModelProto, strategy: Strategy) -> onnx.ModelProto:
    updated = onnx.ModelProto()
    updated.CopyFrom(model)
    for name in strategy.passes:
        updated = apply_pass(name, updated)
    return updated


def prepare_for_ort(model: onnx.ModelProto) -> onnx.ModelProto:
    """Expand compiler-only ops (FusedConv) so ORT CPU can run the graph."""

    return expand_fused_conv(model)
