"""Detect fusable / foldable subgraphs. Counts only; no graph mutation."""

from __future__ import annotations

from collections import defaultdict

import onnx


def detect_patterns(model: onnx.ModelProto) -> dict[str, int]:
    graph = model.graph
    consumers: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    producers: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for name in node.input:
            if name:
                consumers[name].append(node)
        for name in node.output:
            if name:
                producers[name] = node

    inits = {item.name for item in graph.initializer}
    constants = set(inits)
    for node in graph.node:
        if node.op_type == "Constant" and node.output:
            constants.add(node.output[0])

    counts = {
        "conv_bn": 0,
        "conv_relu": 0,
        "conv_silu": 0,
        "conv_add_residual": 0,
        "matmul_add": 0,
        "identity_nodes": 0,
        "dropout_nodes": 0,
        "bn_standalone": 0,
        "concat": 0,
        "resize": 0,
        "reshape_transpose_chains": 0,
        "foldable_constant_nodes": 0,
    }

    bn_used: set[int] = set()
    for node in graph.node:
        if node.op_type == "Identity":
            counts["identity_nodes"] += 1
        elif node.op_type == "Dropout":
            counts["dropout_nodes"] += 1
        elif node.op_type == "Concat":
            counts["concat"] += 1
        elif node.op_type in {"Resize", "Upsample"}:
            counts["resize"] += 1
        if (
            node.op_type in {"Relu", "Add", "Mul", "Identity"}
            and node.input
            and all(name in constants for name in node.input if name)
        ):
            counts["foldable_constant_nodes"] += 1

    for node in graph.node:
        if node.op_type != "Conv" or not node.output:
            continue
        following = [item for item in consumers.get(node.output[0], []) if item is not node]
        if len(following) == 1 and following[0].op_type == "BatchNormalization":
            counts["conv_bn"] += 1
            bn_used.add(id(following[0]))
        if len(following) == 1 and following[0].op_type == "Relu":
            counts["conv_relu"] += 1
        if any(item.op_type == "Add" for item in following):
            counts["conv_add_residual"] += 1
        # SiLU: Conv -> Sigmoid(conv) and Mul(conv, sigmoid)
        sigmoids = [item for item in following if item.op_type == "Sigmoid" and item.output]
        for sig in sigmoids:
            muls = [
                item
                for item in consumers.get(sig.output[0], [])
                if item.op_type == "Mul" and node.output[0] in item.input
            ]
            if muls:
                counts["conv_silu"] += 1
                break

    for node in graph.node:
        if node.op_type == "BatchNormalization" and id(node) not in bn_used:
            counts["bn_standalone"] += 1
        if node.op_type == "MatMul" and node.output:
            adds = [item for item in consumers.get(node.output[0], []) if item.op_type == "Add"]
            if adds:
                counts["matmul_add"] += 1

    for node in graph.node:
        if node.op_type != "Reshape" or not node.output:
            continue
        following = consumers.get(node.output[0], [])
        if any(item.op_type == "Transpose" for item in following):
            counts["reshape_transpose_chains"] += 1

    return counts
