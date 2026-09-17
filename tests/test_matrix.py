from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from compiler.cli import main
from compiler.exporters import write_skip
from compiler.matrix import measure_path
from compiler.schema_validate import validate_run_result


def test_write_skip_is_schema_valid(tmp_path) -> None:
    path = write_skip(
        kind="midas_small",
        path="experiments/models/midas_small.onnx",
        reason="unit test: pretended export failed",
        results_dir=tmp_path,
        stem="skip_midas_small",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_run_result(payload)
    assert payload["compile"]["ok"] is False
    assert payload["skip"]["reason"]
    assert payload["fps_claimed"] is False
    assert "gsk_" not in path.read_text(encoding="utf-8")


def test_matrix_native_vs_optimized_on_tiny_depth(tiny_depth_path, tmp_path) -> None:
    pair = measure_path(
        tiny_depth_path,
        kind="tiny_depth",
        task="depth",
        native_strategy="baseline",
        warmup=1,
        iters=3,
        results_dir=tmp_path,
    )
    assert pair["native"]["strategy"] == "baseline"
    assert pair["optimized"]["strategy"] == "graph_fuse"
    assert pair["native"]["compile_ok"] is True
    assert pair["optimized"]["compile_ok"] is True
    assert pair["native"]["p50_ms"] > 0
    assert pair["optimized"]["p50_ms"] > 0
    assert pair["fps_claimed"] is False
    assert pair["sha256"]


def test_cli_zoo_and_depth_fixture(tmp_path) -> None:
    assert main(["zoo"]) == 0
    depth = tmp_path / "tiny_depth.onnx"
    assert main(["emit-fixture", "--kind", "depth", "--out", str(depth)]) == 0
    assert depth.is_file()
    assert main(["analyze", str(depth)]) == 0
    assert main(["recommend", str(depth)]) == 0


def test_cli_matrix_fixture_only(tiny_depth_path, tmp_path) -> None:
    results = tmp_path / "results"
    code = main(
        [
            "matrix",
            "--kind",
            "not-in-zoo-so-empty",
            "--fixture",
            str(tiny_depth_path),
            "--results",
            str(results),
            "--warmup",
            "1",
            "--iters",
            "2",
        ]
    )
    # zoo kind missing -> skip; fixture still measured so exit 0
    assert code == 0
    paper = results / "paper_matrix.json"
    assert paper.is_file()
    payload = json.loads(paper.read_text(encoding="utf-8"))
    kinds = [row["kind"] for row in payload["models"]]
    assert "fixture" in kinds
    assert payload["fps_claimed"] is False


@pytest.mark.integration
def test_optional_zoo_onnx_skipped_by_default() -> None:
    if os.environ.get("NNC_RUN_ZOO") != "1":
        pytest.skip("large ONNX integration; set NNC_RUN_ZOO=1")
    onnx = Path("experiments/models/yolov8n.onnx")
    if not onnx.is_file():
        pytest.skip("yolov8n.onnx not exported on this machine")
    assert main(["analyze", str(onnx)]) == 0
