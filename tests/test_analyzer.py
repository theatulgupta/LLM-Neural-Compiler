from __future__ import annotations

from pathlib import Path

from compiler.graph.graph_analyzer import analyze_graph
from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph
from compiler.graph.patterns import detect_patterns
from compiler.optimization.passes import fuse_conv_relu


def test_analyzer_sees_flatten_and_gemm(tiny_path) -> None:
    stats = analyze_graph(load_graph(tiny_path))
    assert stats["op_counts"]["Flatten"] == 1
    assert stats["op_counts"]["Gemm"] == 1
    assert stats["op_counts"]["Conv"] == 1
    inputs = stats["inputs"]
    assert inputs[0]["shape"] == [1, 1, 8, 8]


def test_summary_notes_gemm_contract(tiny_path) -> None:
    summary = summarize_graph(load_graph(tiny_path))
    assert any("Gemm K" in note for note in summary.notes)
    assert summary.param_count > 0


def test_analyzer_v2_tiny_flops_and_patterns(tiny_path) -> None:
    loaded = load_graph(tiny_path)
    stats = analyze_graph(loaded)
    assert int(stats["flops_total"]) >= 4608 + 1024
    # Conv 1->4, 3x3, 8x8 out = 2*1*4*1*3*3*8*8 = 4608
    assert int(stats["flops_by_op"]["Conv"]) == 4608
    assert int(stats["flops_by_op"]["Gemm"]) == 1024
    patterns = stats["patterns"]
    assert patterns["conv_relu"] == 1
    assert int(stats["memory"]["peak_live_bytes"]) > 0
    summary = summarize_graph(loaded)
    assert summary.static_shapes is True
    assert summary.patterns["conv_relu"] == 1
    fused = fuse_conv_relu(loaded.model)
    assert detect_patterns(fused)["conv_relu"] == 0


def test_analyzer_v2_yolov8n_if_present() -> None:
    onnx_path = Path("experiments/models/yolov8n.onnx")
    if not onnx_path.is_file():
        return
    stats = analyze_graph(load_graph(onnx_path))
    assert 8e9 <= float(stats["flops_total"]) <= 9.5e9
    assert int(stats["patterns"]["conv_silu"]) >= 50
    assert int(stats["patterns"]["conv_bn"]) == 0
