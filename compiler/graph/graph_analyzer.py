"""Graph statistics: op counts, parameter volume, and shape-aware FLOPs/MACs."""

from __future__ import annotations

from collections import Counter
import math

import onnx
from onnx import numpy_helper, shape_inference

from compiler.graph.graph_loader import LoadedGraph


def _dims(tensor: onnx.ValueInfoProto) -> list[int | None]:
    out: list[int | None] = []
    for dim in tensor.type.tensor_type.shape.dim:
        if dim.HasField("dim_value") and dim.dim_value > 0:
            out.append(int(dim.dim_value))
        else:
            out.append(None)
    return out


def _dtype_name(tensor: onnx.ValueInfoProto) -> str:
    elem = tensor.type.tensor_type.elem_type
    return onnx.TensorProto.DataType.Name(elem).lower()


def _attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def _attr_ints(node: onnx.NodeProto, name: str) -> list[int]:
    for attr in node.attribute:
        if attr.name == name:
            return [int(v) for v in attr.ints]
    return []


def _resolved_shape(
    name: str, shapes: dict[str, list[int | None]], *, batch_fallback: int = 1
) -> list[int] | None:
    raw = shapes.get(name)
    if not raw:
        return None
    resolved: list[int] = []
    for idx, d in enumerate(raw):
        if d is not None and d > 0:
            resolved.append(d)
        elif idx == 0:
            resolved.append(batch_fallback)
        else:
            return None
    return resolved


def analyze_graph(loaded: LoadedGraph) -> dict[str, object]:
    model = shape_inference.infer_shapes(loaded.model)
    graph = model.graph
    op_counts = Counter(node.op_type for node in graph.node)

    param_count = 0
    initializer_bytes = 0
    inits: dict[str, object] = {}
    for initializer in graph.initializer:
        array = numpy_helper.to_array(initializer)
        inits[initializer.name] = array
        param_count += int(array.size)
        initializer_bytes += int(array.nbytes)

    estimated_flops = _estimate_flops(graph, op_counts, inits)
    estimated_macs = estimated_flops // 2 if estimated_flops is not None else None
    estimated_gflops = (estimated_flops / 1e9) if estimated_flops is not None else None

    return {
        "op_counts": dict(sorted(op_counts.items())),
        "node_count": len(graph.node),
        "initializer_count": len(graph.initializer),
        "param_count": param_count,
        "initializer_bytes": initializer_bytes,
        "estimated_flops": estimated_flops,
        "estimated_macs": estimated_macs,
        "estimated_gflops": estimated_gflops,
        "inputs": [
            {"name": item.name, "shape": _dims(item), "dtype": _dtype_name(item)}
            for item in graph.input
            if item.name not in {init.name for init in graph.initializer}
        ],
        "outputs": [
            {"name": item.name, "shape": _dims(item), "dtype": _dtype_name(item)}
            for item in graph.output
        ],
        "opset": loaded.opset,
        "ir_version": loaded.ir_version,
        "producer": loaded.model.producer_name,
    }


def _estimate_flops(
    graph: onnx.GraphProto,
    op_counts: Counter[str],
    inits: dict[str, object] | None = None,
) -> int | None:
    """Shape-aware static FLOPs estimate across standard CNN, MLP, and Transformer ops."""

    if inits is None:
        inits = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    shapes: dict[str, list[int | None]] = {}
    for name, arr in inits.items():
        shapes[name] = [int(x) for x in arr.shape]  # type: ignore[attr-defined]
    for vi in list(graph.input) + list(graph.value_info) + list(graph.output):
        if vi.type.HasField("tensor_type"):
            shapes[vi.name] = _dims(vi)

    total = 0
    known = False

    for node in graph.node:
        op = node.op_type
        out_shape = _resolved_shape(node.output[0], shapes) if node.output else None

        if op == "Conv" and len(node.input) >= 2 and node.input[1] in inits:
            weight = inits[node.input[1]]
            w_shape = weight.shape  # type: ignore[attr-defined]
            cout, cin_per_group = int(w_shape[0]), int(w_shape[1])
            kernel_spatial = math.prod(int(x) for x in w_shape[2:]) if len(w_shape) > 2 else 1
            if out_shape and len(out_shape) >= 3:
                batch = out_shape[0]
                out_spatial = math.prod(out_shape[2:])
            else:
                batch = 1
                out_spatial = 1
            bias_flops = (batch * cout * out_spatial) if len(node.input) >= 3 else 0
            total += 2 * batch * cout * out_spatial * cin_per_group * kernel_spatial + bias_flops
            known = True

        elif op == "Gemm" and len(node.input) >= 2:
            a_shape = _resolved_shape(node.input[0], shapes)
            b_shape = _resolved_shape(node.input[1], shapes)
            trans_a = _attr_int(node, "transA", 0)
            trans_b = _attr_int(node, "transB", 0)
            if a_shape and b_shape and len(a_shape) == 2 and len(b_shape) == 2:
                m = a_shape[1] if trans_a else a_shape[0]
                k = a_shape[0] if trans_a else a_shape[1]
                n = b_shape[0] if trans_b else b_shape[1]
                bias_flops = (m * n) if len(node.input) >= 3 else 0
                total += 2 * m * n * k + bias_flops
                known = True
            elif node.input[1] in inits:
                w_shape = inits[node.input[1]].shape  # type: ignore[attr-defined]
                n, k = (int(w_shape[0]), int(w_shape[1])) if trans_b else (int(w_shape[1]), int(w_shape[0]))
                total += 2 * n * k
                known = True

        elif op == "MatMul" and len(node.input) >= 2:
            a_shape = _resolved_shape(node.input[0], shapes)
            b_shape = _resolved_shape(node.input[1], shapes)
            if a_shape and b_shape and len(a_shape) >= 2 and len(b_shape) >= 2:
                batch_dims = math.prod(out_shape[:-2]) if out_shape and len(out_shape) > 2 else 1
                m = a_shape[-2]
                k = a_shape[-1]
                n = b_shape[-1]
                total += 2 * batch_dims * m * k * n
                known = True

        elif op == "BatchNormalization" and out_shape:
            total += 4 * math.prod(out_shape)
            known = True

        elif op in {"MaxPool", "AveragePool"} and out_shape:
            k_shape = _attr_ints(node, "kernel_shape")
            k_area = math.prod(k_shape) if k_shape else 1
            total += math.prod(out_shape) * k_area
            known = True

        elif op == "GlobalAveragePool" and node.input:
            in_shape = _resolved_shape(node.input[0], shapes)
            if in_shape:
                total += math.prod(in_shape)
                known = True

        elif op in {"Relu", "LeakyRelu", "Sigmoid", "Tanh", "Silu", "Clip", "Add", "Sub", "Mul", "Div"} and out_shape:
            total += math.prod(out_shape)
            known = True

        elif op == "Softmax" and out_shape:
            total += 5 * math.prod(out_shape)
            known = True

    if not known and op_counts:
        return None
    return total
