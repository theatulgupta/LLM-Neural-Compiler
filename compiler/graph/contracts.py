"""Deterministic dimensional and shape-contract verification for ONNX graphs."""

from __future__ import annotations

from typing import Any

import onnx
from onnx import numpy_helper, shape_inference

from compiler.errors import DimensionalContractError


def _attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def _collect_shapes(model: onnx.ModelProto) -> dict[str, list[int | None]]:
    shapes: dict[str, list[int | None]] = {}
    graph = model.graph

    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        shapes[init.name] = [int(d) for d in arr.shape]

    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if not vi.type.HasField("tensor_type"):
            continue
        dims: list[int | None] = []
        for dim in vi.type.tensor_type.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            else:
                dims.append(None)
        shapes[vi.name] = dims
    return shapes


def verify_dimensional_contracts(model: onnx.ModelProto) -> dict[str, Any]:
    """Run strict ONNX shape inference and verify Conv/Flatten/Gemm contracts."""

    try:
        inferred = shape_inference.infer_shapes(model, check_type=True, strict_mode=True)
    except Exception as exc:
        raise DimensionalContractError(f"strict shape inference failed: {exc}") from exc

    shapes = _collect_shapes(inferred)
    checked_contracts: list[str] = []

    for node in inferred.graph.node:
        if node.op_type == "Conv" and len(node.input) >= 2:
            x_shape = shapes.get(node.input[0])
            w_shape = shapes.get(node.input[1])
            group = _attr_int(node, "group", 1)
            if x_shape and w_shape and len(x_shape) >= 2 and len(w_shape) >= 2:
                cin = x_shape[1]
                w_cin = w_shape[1]
                if cin is not None and w_cin is not None and cin != w_cin * group:
                    raise DimensionalContractError(
                        f"Conv node {node.name or '<unnamed>'} channel mismatch: "
                        f"input C={cin} != weight_Cin*group ({w_cin}*{group})"
                    )
                checked_contracts.append(f"Conv:{node.name or node.output[0]}")

        elif node.op_type == "Gemm" and len(node.input) >= 2:
            a_shape = shapes.get(node.input[0])
            b_shape = shapes.get(node.input[1])
            trans_a = _attr_int(node, "transA", 0)
            trans_b = _attr_int(node, "transB", 0)
            if a_shape and b_shape and len(a_shape) == 2 and len(b_shape) == 2:
                k_a = a_shape[0] if trans_a else a_shape[1]
                k_b = b_shape[1] if trans_b else b_shape[0]
                if k_a is not None and k_b is not None and k_a != k_b:
                    raise DimensionalContractError(
                        f"Gemm node {node.name or '<unnamed>'} inner dimension mismatch: "
                        f"A_K={k_a} != B_K={k_b}"
                    )
                checked_contracts.append(f"Gemm:{node.name or node.output[0]}")

        elif node.op_type == "MatMul" and len(node.input) >= 2:
            a_shape = shapes.get(node.input[0])
            b_shape = shapes.get(node.input[1])
            if a_shape and b_shape and len(a_shape) >= 2 and len(b_shape) >= 2:
                k_a = a_shape[-1]
                k_b = b_shape[-2]
                if k_a is not None and k_b is not None and k_a != k_b:
                    raise DimensionalContractError(
                        f"MatMul node {node.name or '<unnamed>'} inner dimension mismatch: "
                        f"A_K={k_a} != B_K={k_b}"
                    )
                checked_contracts.append(f"MatMul:{node.name or node.output[0]}")

    return {
        "passed": True,
        "checked_count": len(checked_contracts),
        "contracts": checked_contracts,
    }
