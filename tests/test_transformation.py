from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from compiler.graph.graph_loader import load_graph
from compiler.optimization.optimizer import apply_strategy, prepare_for_ort
from compiler.optimization.passes import (
    apply_pass,
    eliminate_identity,
    fuse_bn_into_conv,
    fuse_conv_relu,
    fuse_matmul_add_gemm,
    graph_ir_snapshot,
)
from compiler.pipeline import compile_and_benchmark
from compiler.planner import get_strategy
from nnc.backends.ort_cpu import OrtCpuBackend


def _conv_relu_model() -> onnx.ModelProto:
    conv_w = np.ones((2, 1, 1, 1), dtype=np.float32)
    conv_b = np.zeros((2,), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv", ["x", "w", "b"], ["c"], kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]
            ),
            helper.make_node("Relu", ["c"], ["y"]),
        ],
        "conv_relu",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 2])],
        [numpy_helper.from_array(conv_w, "w"), numpy_helper.from_array(conv_b, "b")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)


def _conv_identity_relu() -> onnx.ModelProto:
    conv_w = np.ones((1, 1, 1, 1), dtype=np.float32)
    conv_b = np.zeros((1,), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv", ["x", "w", "b"], ["c"], kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]
            ),
            helper.make_node("Identity", ["c"], ["i"]),
            helper.make_node("Relu", ["i"], ["y"]),
        ],
        "id",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])],
        [numpy_helper.from_array(conv_w, "w"), numpy_helper.from_array(conv_b, "b")],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)


def _conv_bn() -> onnx.ModelProto:
    conv_w = np.ones((2, 1, 1, 1), dtype=np.float32)
    conv_b = np.zeros((2,), dtype=np.float32)
    scale = np.array([2.0, 2.0], dtype=np.float32)
    shift = np.zeros((2,), dtype=np.float32)
    mean = np.zeros((2,), dtype=np.float32)
    var = np.ones((2,), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv", ["x", "w", "b"], ["c"], kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1]
            ),
            helper.make_node("BatchNormalization", ["c", "s", "t", "m", "v"], ["y"]),
        ],
        "bn",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 2])],
        [
            numpy_helper.from_array(conv_w, "w"),
            numpy_helper.from_array(conv_b, "b"),
            numpy_helper.from_array(scale, "s"),
            numpy_helper.from_array(shift, "t"),
            numpy_helper.from_array(mean, "m"),
            numpy_helper.from_array(var, "v"),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)


def test_unknown_pass_rejected() -> None:
    try:
        apply_pass("quantize_int4_magic", _conv_relu_model())
    except ValueError as exc:
        assert "quantize_int4_magic" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_eliminate_identity_drops_node() -> None:
    before = graph_ir_snapshot(_conv_identity_relu())
    after_model = eliminate_identity(_conv_identity_relu())
    after = graph_ir_snapshot(after_model)
    assert after["op_counts"].get("Identity", 0) == 0  # type: ignore[union-attr]
    assert int(after["node_count"]) == int(before["node_count"]) - 1


def test_fuse_conv_relu_changes_tiny_cnn(tiny_path) -> None:
    loaded = load_graph(tiny_path)
    before = graph_ir_snapshot(loaded.model)
    fused = fuse_conv_relu(loaded.model)
    after = graph_ir_snapshot(fused)
    assert after["op_counts"].get("Relu", 0) == 0  # type: ignore[union-attr]
    assert after["op_counts"].get("FusedConv", 0) == 1  # type: ignore[union-attr]
    assert int(after["node_count"]) == int(before["node_count"]) - 1
    backend = OrtCpuBackend()
    session = backend.compile(fused.SerializeToString(), graph_opt="disable")
    x = np.ones((1, 1, 8, 8), dtype=np.float32)
    native = backend.compile(loaded.model.SerializeToString(), graph_opt="disable")
    a = backend.infer(native, {native.input_names[0]: x})[0]
    b = backend.infer(session, {session.input_names[0]: x})[0]
    assert np.max(np.abs(a - b)) < 1e-5


def test_fuse_bn_into_conv_drops_bn() -> None:
    fused = fuse_bn_into_conv(_conv_bn())
    ops = graph_ir_snapshot(fused)["op_counts"]
    assert ops.get("BatchNormalization", 0) == 0  # type: ignore[union-attr]
    assert ops.get("Conv", 0) == 1  # type: ignore[union-attr]


def test_graph_fuse_strategy_on_tiny_cnn_compiles(tiny_path, tmp_path) -> None:
    record = compile_and_benchmark(
        tiny_path,
        backend_name="ort_cpu",
        strategy_name="graph_fuse",
        warmup=1,
        iters=2,
        results_dir=tmp_path / "results",
        model_kind="fixture",
    )
    assert record["compile"]["ok"] is True
    assert record["graph_changed"] is True
    assert record["graph_after"]["op_counts"].get("FusedConv", 0) == 1
    assert record["graph_runtime"]["op_counts"].get("FusedConv", 0) == 1
    assert record["fps_claimed"] is False
    assert "fuse_conv_relu" in record["passes_applied"]


def test_prepare_for_ort_keeps_fusedconv() -> None:
    fused = apply_strategy(_conv_relu_model(), get_strategy("graph_fuse"))
    runtime = prepare_for_ort(fused)
    assert graph_ir_snapshot(runtime)["op_counts"].get("FusedConv", 0) == 1  # type: ignore[union-attr]


def test_fuse_matmul_add_gemm() -> None:
    w = np.ones((4, 3), dtype=np.float32)
    b = np.zeros((3,), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "w"], ["m"]),
            helper.make_node("Add", ["m", "b"], ["y"]),
        ],
        "mm",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        [numpy_helper.from_array(w, "w"), numpy_helper.from_array(b, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
    fused = fuse_matmul_add_gemm(model)
    ops = graph_ir_snapshot(fused)["op_counts"]
    assert ops.get("Gemm", 0) == 1
    assert ops.get("MatMul", 0) == 0
