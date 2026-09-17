"""Tiny depth-like CNN used in tests. Not a real UAV weight file.

Input 1x3x8x8 -> a few Conv/Relu layers -> 1x1x8x8. Node count is above the
heuristic tiny-graph cutoff so recommend() can pick graph_fuse without
downloading MiDaS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

OPSET = 13
IR_VERSION = 8
SEED = 1
INPUT_SHAPE = (1, 3, 8, 8)
INPUT_NAME = "images"
OUTPUT_NAME = "depth"


def write_tiny_depth(path: Path, *, seed: int = SEED) -> Path:
    rng = np.random.default_rng(seed)
    channels = [3, 4, 4, 4, 4, 1]
    initializers = []
    nodes = []
    current = INPUT_NAME
    for index in range(len(channels) - 1):
        cin = channels[index]
        cout = channels[index + 1]
        w_name = f"conv{index}_w"
        b_name = f"conv{index}_b"
        out_name = OUTPUT_NAME if index == len(channels) - 2 else f"conv{index}"
        k = 1 if index == len(channels) - 2 else 3
        pad = 0 if k == 1 else 1
        initializers.append(
            numpy_helper.from_array(
                rng.standard_normal((cout, cin, k, k), dtype=np.float32) * 0.05,
                name=w_name,
            )
        )
        initializers.append(numpy_helper.from_array(np.zeros((cout,), dtype=np.float32), name=b_name))
        nodes.append(
            helper.make_node(
                "Conv",
                [current, w_name, b_name],
                [out_name],
                kernel_shape=[k, k],
                pads=[pad, pad, pad, pad],
                strides=[1, 1],
            )
        )
        if out_name != OUTPUT_NAME:
            relu_name = f"relu{index}"
            nodes.append(helper.make_node("Relu", [out_name], [relu_name]))
            current = relu_name
        else:
            current = out_name

    graph = helper.make_graph(
        nodes,
        "tiny_depth",
        [helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, list(INPUT_SHAPE))],
        [helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, [1, 1, 8, 8])],
        initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="llm-neural-compiler",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = "tiny depth-like fixture; RGB 8x8 to 1-channel depth"
    import onnx

    onnx.checker.check_model(model, full_check=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path
