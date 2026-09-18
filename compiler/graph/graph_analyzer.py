"""Graph statistics: op counts, parameter volume, shape-inferred FLOPs."""

from __future__ import annotations

from collections import Counter

import onnx
from onnx import numpy_helper, shape_inference

from compiler.graph.graph_loader import LoadedGraph
from compiler.graph.memory import activation_footprint
from compiler.graph.patterns import detect_patterns


def _dims(tensor: onnx.ValueInfoProto) -> list[int | None]:
    out: list[int | None] = []
    for dim in tensor.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            out.append(int(dim.dim_value))
        else:
            out.append(None)
    return out


def _dtype_name(tensor: onnx.ValueInfoProto) -> str:
    elem = tensor.type.tensor_type.elem_type
    return onnx.TensorProto.DataType.Name(elem).lower()


def _collect_shapes(graph: onnx.GraphProto) -> tuple[dict[str, tuple[int | None, ...]], dict[str, int], bool]:
    shapes: dict[str, tuple[int | None, ...]] = {}
    dtypes: dict[str, int] = {}
    static = True

    def _ingest(tensor: onnx.ValueInfoProto) -> None:
        nonlocal static
        dims = tuple(_dims(tensor))
        shapes[tensor.name] = dims
        dtypes[tensor.name] = int(tensor.type.tensor_type.elem_type)
        if any(d is None or d <= 0 for d in dims):
            static = False

    for item in list(graph.input) + list(graph.output) + list(graph.value_info):
        _ingest(item)
    for item in graph.initializer:
        shapes[item.name] = tuple(int(d) for d in item.dims)
        dtypes[item.name] = int(item.data_type)
    return shapes, dtypes, static


def _attr_ints(node: onnx.NodeProto, name: str) -> list[int] | None:
    for attr in node.attribute:
        if attr.name == name:
            if attr.ints:
                return [int(v) for v in attr.ints]
            if attr.HasField("i"):
                return [int(attr.i)]
    return None


def _attr_int(node: onnx.NodeProto, name: str, default: int = 0) -> int:
    for attr in node.attribute:
        if attr.name == name and attr.HasField("i"):
            return int(attr.i)
    return default


def _node_flops(
    node: onnx.NodeProto,
    shapes: dict[str, tuple[int | None, ...]],
    inits: dict[str, object],
) -> tuple[int, bool]:
    """Return (flops, known). 2x MACs for conv/gemm/matmul; numel for elementwise."""

    out_shape = shapes.get(node.output[0]) if node.output else None

    if node.op_type in {"Conv", "ConvTranspose"} and len(node.input) >= 2:
        weight_shape = shapes.get(node.input[1])
        if weight_shape is None or len(weight_shape) < 4 or out_shape is None or len(out_shape) < 4:
            return 0, False
        if any(d is None for d in weight_shape[:4]) or any(d is None for d in out_shape[:4]):
            return 0, False
        n, cout, hout, wout = (
            int(out_shape[0] or 1),
            int(out_shape[1]),
            int(out_shape[2]),
            int(out_shape[3]),
        )
        groups = max(1, _attr_int(node, "group", 1))
        if node.op_type == "Conv":
            cin = int(weight_shape[1])
            kh, kw = int(weight_shape[2]), int(weight_shape[3])
            return 2 * n * cout * cin * kh * kw * hout * wout, True
        # ConvTranspose weight is [Cin, Cout/groups, kh, kw]
        cin = int(weight_shape[0])
        kh, kw = int(weight_shape[2]), int(weight_shape[3])
        cout_w = int(weight_shape[1])
        return 2 * n * cin * cout_w * groups * kh * kw * hout * wout, True

    if node.op_type in {"Gemm", "MatMul"} and len(node.input) >= 2:
        a = shapes.get(node.input[0])
        b = shapes.get(node.input[1])
        if a is None or b is None or len(a) < 2 or len(b) < 2:
            return 0, False
        if any(d is None for d in a[-2:] + b[-2:]):
            return 0, False
        trans_a = bool(_attr_int(node, "transA", 0)) if node.op_type == "Gemm" else False
        trans_b = bool(_attr_int(node, "transB", 0)) if node.op_type == "Gemm" else False
        m = int(a[-1] if trans_a else a[-2])
        ka = int(a[-2] if trans_a else a[-1])
        kb = int(b[-1] if trans_b else b[-2])
        n = int(b[-2] if trans_b else b[-1])
        if ka != kb:
            k = ka if ka > 0 else kb
        else:
            k = ka
        batch = 1
        for dim in a[:-2]:
            if dim is None:
                return 0, False
            batch *= int(dim)
        return 2 * batch * m * n * k, True

    if node.op_type in {
        "Add",
        "Mul",
        "Div",
        "Sub",
        "Relu",
        "Sigmoid",
        "Tanh",
        "Clip",
        "LeakyRelu",
        "MaxPool",
        "AveragePool",
        "GlobalAveragePool",
        "Flatten",
        "Reshape",
        "Transpose",
        "Concat",
        "Resize",
        "Identity",
    }:
        if out_shape is None:
            return 0, False
        total = 1
        for dim in out_shape:
            if dim is None or dim <= 0:
                return 0, False
            total *= int(dim)
        return total, True

    return 0, False


def analyze_graph(loaded: LoadedGraph) -> dict[str, object]:
    try:
        model = shape_inference.infer_shapes(loaded.model, strict_mode=False)
    except Exception:  # noqa: BLE001 — fused / custom ops may not infer
        model = loaded.model
    graph = model.graph
    op_counts = Counter(node.op_type for node in graph.node)
    shapes, dtypes, static_shapes = _collect_shapes(graph)
    inits = {init.name: numpy_helper.to_array(init) for init in graph.initializer}

    param_count = 0
    initializer_bytes = 0
    for initializer in graph.initializer:
        array = numpy_helper.to_array(initializer)
        param_count += int(array.size)
        initializer_bytes += int(array.nbytes)

    flops_total = 0
    flops_by_op: dict[str, int] = {}
    unknown: list[str] = []
    per_node: list[dict[str, object]] = []
    for node in graph.node:
        flops, known = _node_flops(node, shapes, inits)
        if not known:
            unknown.append(f"{node.name or node.output[0] if node.output else node.op_type}:{node.op_type}")
        flops_total += flops
        flops_by_op[node.op_type] = flops_by_op.get(node.op_type, 0) + flops
        out_shape = list(shapes.get(node.output[0], ())) if node.output else []
        per_node.append(
            {
                "name": node.name or (node.output[0] if node.output else node.op_type),
                "op_type": node.op_type,
                "flops": flops,
                "out_shape": out_shape,
            }
        )
    per_node.sort(key=lambda row: int(row["flops"]), reverse=True)
    top_nodes = per_node[:10]
    patterns = detect_patterns(model)
    memory = activation_footprint(model, shapes, dtypes)
    graph_inputs = {item.name for item in graph.input}
    init_names = {init.name for init in graph.initializer}

    return {
        "op_counts": dict(sorted(op_counts.items())),
        "node_count": len(graph.node),
        "initializer_count": len(graph.initializer),
        "param_count": param_count,
        "initializer_bytes": initializer_bytes,
        "estimated_flops": flops_total,
        "flops_total": flops_total,
        "flops_by_op": dict(sorted(flops_by_op.items())),
        "flops_unknown_ops": unknown,
        "top_nodes": top_nodes,
        "patterns": patterns,
        "memory": memory,
        "static_shapes": static_shapes,
        "shapes": {
            name: list(dims)
            for name, dims in shapes.items()
            if name in graph_inputs or name not in init_names
        },
        "inputs": [
            {"name": item.name, "shape": _dims(item), "dtype": _dtype_name(item)}
            for item in graph.input
            if item.name not in init_names
        ],
        "outputs": [
            {"name": item.name, "shape": _dims(item), "dtype": _dtype_name(item)} for item in graph.output
        ],
        "opset": loaded.opset,
        "ir_version": loaded.ir_version,
        "producer": loaded.model.producer_name,
    }
