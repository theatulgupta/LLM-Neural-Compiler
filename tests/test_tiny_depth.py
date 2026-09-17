from __future__ import annotations

from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph
from compiler.llm.recommendation_engine import recommend_strategy
from compiler.pipeline import compile_and_benchmark


def test_tiny_depth_input_is_rgb(tiny_depth_path) -> None:
    loaded = load_graph(tiny_depth_path)
    summary = summarize_graph(loaded)
    assert summary.inputs[0]["shape"] == [1, 3, 8, 8]
    assert summary.outputs[0]["shape"] == [1, 1, 8, 8]
    assert summary.op_counts.get("Conv", 0) >= 4
    assert summary.node_count > 8


def test_tiny_depth_recommend_is_not_yolo_hardcoded(tiny_depth_path) -> None:
    rec = recommend_strategy(summarize_graph(load_graph(tiny_depth_path)))
    assert rec.strategy.name == "ort_extended"
    assert rec.source == "heuristic"


def test_tiny_depth_compiles_on_ort(tiny_depth_path, tmp_path) -> None:
    record = compile_and_benchmark(
        tiny_depth_path,
        backend_name="ort_cpu",
        strategy_name="baseline",
        warmup=1,
        iters=3,
        results_dir=tmp_path / "results",
        model_kind="tiny_depth",
    )
    assert record["compile"]["ok"] is True
    assert record["fps_claimed"] is False
    assert record["host"]["uname_m"]
    assert record["benchmark"]["latency_ms"]["mean"] > 0
