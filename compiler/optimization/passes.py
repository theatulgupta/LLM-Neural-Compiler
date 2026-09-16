"""Allowlisted graph passes applied before a backend compile."""

from __future__ import annotations

from collections.abc import Callable

import onnx
from onnx import shape_inference

PassFn = Callable[[onnx.ModelProto], onnx.ModelProto]

ALLOWED_PASSES: dict[str, PassFn]


def onnx_shape_infer(model: onnx.ModelProto) -> onnx.ModelProto:
    return shape_inference.infer_shapes(model)


ALLOWED_PASSES = {
    "onnx_shape_infer": onnx_shape_infer,
}


def apply_pass(name: str, model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        fn = ALLOWED_PASSES[name]
    except KeyError as exc:
        raise ValueError(f"pass {name!r} is not allowlisted; allowed={sorted(ALLOWED_PASSES)}") from exc
    return fn(model)
