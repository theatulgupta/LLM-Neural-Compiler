from __future__ import annotations

from pathlib import Path

from compiler.graph.graph_loader import load_graph
from compiler.optimization.passes import graph_ir_snapshot
from compiler.pipeline import compile_verify_profile
from compiler.planner.plan import get_plan


def test_pipeline_v2_graph_fuse_tiny(tiny_path, tmp_path) -> None:
    record = compile_verify_profile(
        tiny_path,
        plan=get_plan("graph_fuse"),
        kind="fixture",
        task="classify",
        results_dir=tmp_path,
        warmup=1,
        iters=2,
    )
    assert record["compile"]["ok"] is True
    assert record["graph_runtime"]["op_counts"].get("FusedConv", 0) == 1
    assert record["verification"]["passed"] is True
    assert Path(record["artifact"]["path"]).is_file()
    assert record["schema_version"] == 2


def test_pipeline_v2_int8_does_not_raise(tiny_path, tmp_path) -> None:
    record = compile_verify_profile(
        tiny_path,
        plan=get_plan("int8_dynamic"),
        kind="fixture",
        task="classify",
        results_dir=tmp_path,
        warmup=0,
        iters=1,
    )
    assert "compile" in record
    assert record.get("fps_claimed") is False
