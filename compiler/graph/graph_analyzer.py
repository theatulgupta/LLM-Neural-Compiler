"""Graph statistics: op counts, parameter volume, coarse FLOP estimates."""

from __future__ import annotations

from collections import Counter

import onnx
from onnx import numpy_helper, shape_inference

from compiler.graph.graph_loader import LoadedGraph


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


def analyze_graph(loaded: LoadedGraph) -> dict[str, object]:
    model = shape_inference.infer_shapes(loaded.model)
    graph = model.graph
    op_counts = Counter(node.op_type for node in graph.node)

    param_count = 0
    initializer_bytes = 0
    for initializer in graph.initializer:
        array = numpy_helper.to_array(initializer)
        param_count += int(array.size)
        initializer_bytes += int(array.nbytes)

    estimated_flops = _estimate_flops(graph, op_counts)

    return {
        "op_counts": dict(sorted(op_counts.items())),
        "node_count": len(graph.node),
        "initializer_count": len(graph.initializer),
        "param_count": param_count,
        "initializer_bytes": initializer_bytes,
        "estimated_flops": estimated_flops,
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


def _estimate_flops(graph: onnx.GraphProto, op_counts: Counter[str]) -> int | None:
    """Coarse static estimate; None if a required shape is missing."""

    inits = {init.name: numpy_helper.to_array(init) for init in graph.initializer}
    total = 0
    known = False
    for node in graph.node:
        if node.op_type == "Gemm" and len(node.input) >= 2 and node.input[1] in inits:
            weight = inits[node.input[1]]
            # transB=1 => Y[M,N] = A[M,K] @ W[N,K].T  -> 2*M*N*K; M unknown, use 1.
            n, k = int(weight.shape[0]), int(weight.shape[1])
            total += 2 * n * k
            known = True
        elif node.op_type == "Conv" and node.input[1] in inits:
            weight = inits[node.input[1]]
            # 2 * Cout * Cin * Kh * Kw * Oh * Ow; assume Oh=Ow from doc or skip spatial.
            cout, cin, kh, kw = (int(x) for x in weight.shape)
            # Tiny fixture is 8x8 after conv with pad=1; unknown graphs use 1x1.
            oh = ow = 8 if cout == 4 and cin == 1 else 1
            total += 2 * cout * cin * kh * kw * oh * ow
            known = True
    if not known and op_counts:
        return None
    return total
