"""Deterministic tiny CNN fixture used by tests and the ORT CPU baseline.

Layout (NCHW, opset 13):

    input [1, 1, 8, 8]
      -> Conv 1x4, k=3, pad=1     [1, 4, 8, 8]
      -> Relu
      -> MaxPool 2x2, stride 2    [1, 4, 4, 4]
      -> Flatten axis=1           [1, 64]
      -> Gemm transB=1            [1, 8]

The first compile failure on this machine was Gemm K=16 against Flatten 64.
Gemm B is stored with shape [N, K] (transB=1). K must equal 4*4*4 = 64, not
the channel count or a 4x4 spatial guess of 16.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from compiler.errors import FixtureError

OPSET = 13
IR_VERSION = 8
SEED = 0
INPUT_SHAPE = (1, 1, 8, 8)
CONV_OUT_CHANNELS = 4
POOL_SPATIAL = 4
GEMM_OUT = 8
INPUT_NAME = "input"
OUTPUT_NAME = "logits"


def flatten_features(
    conv_out_channels: int = CONV_OUT_CHANNELS,
    pool_spatial: int = POOL_SPATIAL,
) -> int:
    return int(conv_out_channels * pool_spatial * pool_spatial)


def write_tiny_cnn(
    path: Path,
    *,
    gemm_k: int | None = None,
    seed: int = SEED,
    validate: bool = True,
) -> Path:
    """Write the tiny CNN. ``gemm_k`` defaults to the Flatten width (64)."""

    expected_k = flatten_features()
    k = expected_k if gemm_k is None else int(gemm_k)
    rng = np.random.default_rng(seed)

    conv_w = rng.standard_normal((CONV_OUT_CHANNELS, INPUT_SHAPE[1], 3, 3), dtype=np.float32)
    conv_b = np.zeros((CONV_OUT_CHANNELS,), dtype=np.float32)
    gemm_w = rng.standard_normal((GEMM_OUT, k), dtype=np.float32)
    gemm_b = np.zeros((GEMM_OUT,), dtype=np.float32)

    input_tensor = helper.make_tensor_value_info(INPUT_NAME, TensorProto.FLOAT, list(INPUT_SHAPE))
    output_tensor = helper.make_tensor_value_info(OUTPUT_NAME, TensorProto.FLOAT, [INPUT_SHAPE[0], GEMM_OUT])
    flatten_info = helper.make_tensor_value_info("flat", TensorProto.FLOAT, [INPUT_SHAPE[0], expected_k])

    nodes = [
        helper.make_node(
            "Conv",
            [INPUT_NAME, "conv_w", "conv_b"],
            ["conv"],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Relu", ["conv"], ["relu"]),
        helper.make_node(
            "MaxPool",
            ["relu"],
            ["pool"],
            kernel_shape=[2, 2],
            strides=[2, 2],
        ),
        helper.make_node("Flatten", ["pool"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "gemm_w", "gemm_b"], [OUTPUT_NAME], transB=1, alpha=1.0, beta=1.0),
    ]

    graph = helper.make_graph(
        nodes,
        "tiny_cnn",
        [input_tensor],
        [output_tensor],
        [
            numpy_helper.from_array(conv_w, name="conv_w"),
            numpy_helper.from_array(conv_b, name="conv_b"),
            numpy_helper.from_array(gemm_w, name="gemm_w"),
            numpy_helper.from_array(gemm_b, name="gemm_b"),
        ],
        value_info=[flatten_info],
    )
    model = helper.make_model(
        graph,
        producer_name="llm-neural-compiler",
        ir_version=IR_VERSION,
        opset_imports=[helper.make_opsetid("", OPSET)],
    )
    model.doc_string = (
        f"tiny CNN fixture; Flatten K={expected_k}; Gemm K={k} (transB=1, W shape [{GEMM_OUT},{k}])"
    )

    if k != expected_k and validate:
        raise FixtureError(
            f"Gemm K={k} does not match Flatten width {expected_k}; "
            "pass validate=False only when constructing the known-broken regression case"
        )

    if validate:
        onnx.checker.check_model(model, full_check=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path
