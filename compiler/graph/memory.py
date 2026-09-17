"""Static activation / weight footprint from inferred tensor shapes."""

from __future__ import annotations

from collections import defaultdict

import onnx

_ELEM_BYTES = {
    onnx.TensorProto.FLOAT: 4,
    onnx.TensorProto.FLOAT16: 2,
    onnx.TensorProto.BFLOAT16: 2,
    onnx.TensorProto.DOUBLE: 8,
    onnx.TensorProto.INT8: 1,
    onnx.TensorProto.UINT8: 1,
    onnx.TensorProto.INT32: 4,
    onnx.TensorProto.INT64: 8,
    onnx.TensorProto.BOOL: 1,
}


def _numel(shape: tuple[int | None, ...]) -> int | None:
    total = 1
    for dim in shape:
        if dim is None or dim <= 0:
            return None
        total *= int(dim)
    return total


def _bytes(shape: tuple[int | None, ...], dtype: int) -> int | None:
    n = _numel(shape)
    if n is None:
        return None
    return n * int(_ELEM_BYTES.get(dtype, 4))


def activation_footprint(
    model: onnx.ModelProto,
    shapes: dict[str, tuple[int | None, ...]],
    dtypes: dict[str, int],
) -> dict[str, object]:
    graph = model.graph
    inits = {item.name for item in graph.initializer}
    weights_bytes = 0
    for item in graph.initializer:
        array_shape = tuple(int(d) for d in item.dims)
        nbytes = _bytes(array_shape, item.data_type)
        if nbytes is not None:
            weights_bytes += nbytes

    last_use: dict[str, int] = {}
    produced_at: dict[str, int] = {}
    for index, node in enumerate(graph.node):
        for name in node.input:
            if name and name not in inits:
                last_use[name] = index
        for name in node.output:
            if name:
                produced_at[name] = index
                last_use.setdefault(name, index)

    live: dict[int, list[str]] = defaultdict(list)
    for name, start in produced_at.items():
        end = last_use.get(name, start)
        for tick in range(start, end + 1):
            live[tick].append(name)

    activations_total = 0
    largest_name = ""
    largest_bytes = 0
    largest_shape: tuple[int | None, ...] = ()
    seen: set[str] = set()
    for name, shape in shapes.items():
        if name in inits or name in seen:
            continue
        seen.add(name)
        nbytes = _bytes(shape, dtypes.get(name, onnx.TensorProto.FLOAT))
        if nbytes is None:
            continue
        activations_total += nbytes
        if nbytes > largest_bytes:
            largest_bytes = nbytes
            largest_name = name
            largest_shape = shape

    peak = 0
    for names in live.values():
        tick_bytes = 0
        for name in names:
            nbytes = _bytes(shapes.get(name, ()), dtypes.get(name, onnx.TensorProto.FLOAT))
            if nbytes:
                tick_bytes += nbytes
        if tick_bytes > peak:
            peak = tick_bytes

    return {
        "weights_bytes": weights_bytes,
        "activations_total_bytes": activations_total,
        "peak_live_bytes": peak,
        "largest_tensor": {"name": largest_name, "bytes": largest_bytes, "shape": list(largest_shape)},
    }
