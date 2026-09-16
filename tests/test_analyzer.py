from __future__ import annotations

from compiler.graph.graph_analyzer import analyze_graph
from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph


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
