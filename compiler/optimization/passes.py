"""Allowlisted ONNX graph passes applied before backend compilation."""

from __future__ import annotations

from collections.abc import Callable

import onnx
from onnx import shape_inference

from compiler.graph.contracts import verify_dimensional_contracts
from compiler.strategies import Strategy

PassFn = Callable[[onnx.ModelProto], onnx.ModelProto]


def onnx_shape_infer(model: onnx.ModelProto) -> onnx.ModelProto:
    return shape_inference.infer_shapes(model)


ALLOWED_PASSES: dict[str, PassFn] = {
    "onnx_shape_infer": onnx_shape_infer,
}


def apply_pass(name: str, model: onnx.ModelProto) -> onnx.ModelProto:
    if name not in ALLOWED_PASSES:
        raise ValueError(f"pass {name!r} is not allowlisted; allowed={sorted(ALLOWED_PASSES)}")
    return ALLOWED_PASSES[name](model)


def apply_strategy(
    model: onnx.ModelProto,
    strategy: Strategy,
    *,
    verify_contracts: bool = True,
) -> onnx.ModelProto:
    updated = onnx.ModelProto()
    updated.CopyFrom(model)
    for name in strategy.passes:
        updated = apply_pass(name, updated)
    # Catch shape/dimension mismatches before handing the graph to ORT.
    if verify_contracts:
        verify_dimensional_contracts(updated)
    return updated
