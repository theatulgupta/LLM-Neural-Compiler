"""Allowlisted graph passes that copy-and-mutate an ONNX ModelProto.

Unknown names are rejected. Passes must not invent latency. Shape inference
fills shapes; the other passes rewrite the DAG (fusion, folding, DCE).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable

import numpy as np
import onnx
from onnx import helper, numpy_helper, shape_inference

from compiler.utils.hashing import sha256_bytes

PassFn = Callable[[onnx.ModelProto], onnx.ModelProto]

ALLOWED_PASSES: dict[str, PassFn]


def _copy_model(model: onnx.ModelProto) -> onnx.ModelProto:
    updated = onnx.ModelProto()
    updated.CopyFrom(model)
    return updated


def _attr_map(node: onnx.NodeProto) -> dict[str, object]:
    values: dict[str, object] = {}
    for attr in node.attribute:
        if attr.type == onnx.AttributeProto.INTS:
            values[attr.name] = list(attr.ints)
        elif attr.type == onnx.AttributeProto.INT:
            values[attr.name] = int(attr.i)
        elif attr.type == onnx.AttributeProto.FLOATS:
            values[attr.name] = list(attr.floats)
        elif attr.type == onnx.AttributeProto.FLOAT:
            values[attr.name] = float(attr.f)
        elif attr.type == onnx.AttributeProto.STRING:
            raw = attr.s
            values[attr.name] = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
    return values


def _replace_uses(nodes: Iterable[onnx.NodeProto], old: str, new: str) -> None:
    for node in nodes:
        for index, name in enumerate(node.input):
            if name == old:
                node.input[index] = new
        for index, name in enumerate(node.output):
            if name == old:
                node.output[index] = new


def _graph_io_rename(graph: onnx.GraphProto, old: str, new: str) -> None:
    for item in list(graph.input) + list(graph.output) + list(graph.value_info):
        if item.name == old:
            item.name = new


def onnx_shape_infer(model: onnx.ModelProto) -> onnx.ModelProto:
    updated = _copy_model(model)
    try:
        return shape_inference.infer_shapes(updated)
    except Exception:  # noqa: BLE001 — fused compiler ops may not infer; keep the DAG
        return updated


def eliminate_identity(model: onnx.ModelProto) -> onnx.ModelProto:
    """Dead-node-lite: drop Identity / nop Dropout and rewire the DAG."""

    updated = _copy_model(model)
    graph = updated.graph
    kept: list[onnx.NodeProto] = []
    for node in list(graph.node):
        if node.op_type == "Identity" and len(node.input) == 1 and len(node.output) == 1:
            src, dst = node.input[0], node.output[0]
            _replace_uses(graph.node, dst, src)
            _graph_io_rename(graph, dst, src)
            continue
        if node.op_type == "Dropout":
            ratio = float(_attr_map(node).get("ratio", 0.5) or 0.0)
            if ratio == 0.0 and len(node.input) >= 1 and len(node.output) >= 1:
                src, dst = node.input[0], node.output[0]
                _replace_uses(graph.node, dst, src)
                _graph_io_rename(graph, dst, src)
                continue
        kept.append(node)
    del graph.node[:]
    graph.node.extend(kept)
    return updated


def constant_folding(model: onnx.ModelProto) -> onnx.ModelProto:
    """Fold Relu/Identity on initializers into new initializers; drop the node."""

    updated = _copy_model(model)
    graph = updated.graph
    init = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    kept: list[onnx.NodeProto] = []
    extra_inits: list[onnx.TensorProto] = []
    for node in list(graph.node):
        if node.op_type == "Relu" and node.input and node.input[0] in init:
            folded = np.maximum(init[node.input[0]], 0).astype(init[node.input[0]].dtype, copy=False)
            extra_inits.append(numpy_helper.from_array(folded, name=node.output[0]))
            init[node.output[0]] = folded
            continue
        if node.op_type in {"Add", "Mul"} and len(node.input) == 2 and node.input[0] in init and node.input[1] in init:
            left, right = init[node.input[0]], init[node.input[1]]
            folded = left + right if node.op_type == "Add" else left * right
            extra_inits.append(numpy_helper.from_array(np.asarray(folded), name=node.output[0]))
            init[node.output[0]] = np.asarray(folded)
            continue
        kept.append(node)
    del graph.node[:]
    graph.node.extend(kept)
    graph.initializer.extend(extra_inits)
    return updated


def fuse_bn_into_conv(model: onnx.ModelProto) -> onnx.ModelProto:
    """Conv followed by BatchNormalization becomes one Conv (standard BN fold)."""

    updated = _copy_model(model)
    graph = updated.graph
    init = {item.name: item for item in graph.initializer}
    nodes = list(graph.node)
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in nodes:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    skip: set[int] = set()
    kept: list[onnx.NodeProto] = []
    new_inits: list[onnx.TensorProto] = []
    removed_init: set[str] = set()

    def _arr(name: str) -> np.ndarray:
        return numpy_helper.to_array(init[name])

    for index, node in enumerate(nodes):
        if index in skip:
            continue
        if node.op_type != "Conv" or len(node.output) != 1:
            kept.append(node)
            continue
        conv_out = node.output[0]
        following = [item for item in consumers.get(conv_out, []) if item is not node]
        if len(following) != 1 or following[0].op_type != "BatchNormalization":
            kept.append(node)
            continue
        bn = following[0]
        if len(bn.input) < 5:
            kept.append(node)
            continue
        weight_name = node.input[1]
        if weight_name not in init:
            kept.append(node)
            continue
        weight = _arr(weight_name)
        bias = _arr(node.input[2]) if len(node.input) > 2 and node.input[2] in init else np.zeros(weight.shape[0], dtype=weight.dtype)
        scale = _arr(bn.input[1])
        shift = _arr(bn.input[2])
        mean = _arr(bn.input[3])
        var = _arr(bn.input[4])
        eps = float(_attr_map(bn).get("epsilon", 1e-5) or 1e-5)
        std = np.sqrt(var + eps)
        factor = (scale / std).astype(weight.dtype, copy=False)
        new_w = weight * factor.reshape((-1,) + (1,) * (weight.ndim - 1))
        new_b = (bias - mean) * factor + shift
        w_name = weight_name + "_bn"
        b_name = (node.input[2] if len(node.input) > 2 else weight_name + "_bias") + "_bn"
        new_inits.append(numpy_helper.from_array(new_w, name=w_name))
        new_inits.append(numpy_helper.from_array(new_b.astype(weight.dtype, copy=False), name=b_name))
        fused = helper.make_node(
            "Conv",
            [node.input[0], w_name, b_name],
            [bn.output[0]],
            name=(node.name or "conv") + "_bn",
            **{key: value for key, value in _attr_map(node).items() if key in {"kernel_shape", "pads", "strides", "dilations", "group", "auto_pad"}},
        )
        kept.append(fused)
        skip.add(nodes.index(bn))
        removed_init.update([weight_name, bn.input[1], bn.input[2], bn.input[3], bn.input[4]])
        if len(node.input) > 2:
            removed_init.add(node.input[2])

    del graph.node[:]
    graph.node.extend(kept)
    remaining = [item for item in graph.initializer if item.name not in removed_init]
    del graph.initializer[:]
    graph.initializer.extend(remaining)
    graph.initializer.extend(new_inits)
    return updated


def fuse_conv_relu(model: onnx.ModelProto) -> onnx.ModelProto:
    """Conv whose only consumer is Relu becomes one FusedConv (compiler IR).

    ORT CPU may not run ``com.microsoft.FusedConv``. ``prepare_for_ort`` expands
    it back to Conv+Relu before the session so numerics stay the same.
    """

    updated = _copy_model(model)
    graph = updated.graph
    nodes = list(graph.node)
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in nodes:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    skip: set[int] = set()
    kept: list[onnx.NodeProto] = []
    fused_any = False
    for index, node in enumerate(nodes):
        if index in skip:
            continue
        if node.op_type != "Conv" or len(node.output) != 1:
            kept.append(node)
            continue
        following = [item for item in consumers.get(node.output[0], []) if item is not node]
        if len(following) != 1 or following[0].op_type != "Relu" or len(following[0].input) != 1:
            kept.append(node)
            continue
        relu = following[0]
        conv_attrs = {
            key: value
            for key, value in _attr_map(node).items()
            if key in {"kernel_shape", "pads", "strides", "dilations", "group", "auto_pad"}
        }
        fused = helper.make_node(
            "FusedConv",
            list(node.input),
            list(relu.output),
            name=(relu.output[0] if relu.output else node.output[0]) + "__fused",
            domain="com.microsoft",
            activation="Relu",
            **conv_attrs,
        )
        kept.append(fused)
        skip.add(nodes.index(relu))
        fused_any = True

    if fused_any:
        domains = {item.domain for item in updated.opset_import}
        if "com.microsoft" not in domains:
            updated.opset_import.extend([helper.make_opsetid("com.microsoft", 1)])

    del graph.node[:]
    graph.node.extend(kept)
    return updated


def expand_fused_conv(model: onnx.ModelProto) -> onnx.ModelProto:
    """Turn compiler FusedConv nodes back into Conv+Relu for ORT CPU."""

    updated = _copy_model(model)
    graph = updated.graph
    kept: list[onnx.NodeProto] = []
    for node in list(graph.node):
        if node.op_type != "FusedConv":
            kept.append(node)
            continue
        attrs = _attr_map(node)
        activation = str(attrs.get("activation", "Relu"))
        conv_attrs = {
            key: value
            for key, value in attrs.items()
            if key in {"kernel_shape", "pads", "strides", "dilations", "group", "auto_pad"}
        }
        conv_out = node.output[0] + "__preact"
        conv = helper.make_node(
            "Conv",
            list(node.input),
            [conv_out],
            name=(node.name or node.output[0]) + "__conv",
            **conv_attrs,
        )
        act = helper.make_node(
            activation if activation in {"Relu", "Sigmoid", "Tanh"} else "Relu",
            [conv_out],
            list(node.output),
            name=(node.name or node.output[0]) + "__act",
        )
        kept.extend([conv, act])
    del graph.node[:]
    graph.node.extend(kept)
    # Microsoft opset is unused after expansion.
    remaining = [item for item in updated.opset_import if item.domain != "com.microsoft"]
    del updated.opset_import[:]
    updated.opset_import.extend(remaining)
    return updated


ALLOWED_PASSES = {
    "onnx_shape_infer": onnx_shape_infer,
    "eliminate_identity": eliminate_identity,
    "constant_folding": constant_folding,
    "fuse_bn_into_conv": fuse_bn_into_conv,
    "fuse_conv_relu": fuse_conv_relu,
}


def graph_ir_snapshot(model: onnx.ModelProto) -> dict[str, object]:
    """DAG stats without requiring shape inference (FusedConv may not infer)."""

    op_counts = Counter(node.op_type for node in model.graph.node)
    data = model.SerializeToString()
    return {
        "node_count": len(model.graph.node),
        "op_counts": dict(sorted(op_counts.items())),
        "sha256": sha256_bytes(data),
    }


def apply_pass(name: str, model: onnx.ModelProto) -> onnx.ModelProto:
    try:
        fn = ALLOWED_PASSES[name]
    except KeyError as exc:
        raise ValueError(f"pass {name!r} is not allowlisted; allowed={sorted(ALLOWED_PASSES)}") from exc
    return fn(model)
