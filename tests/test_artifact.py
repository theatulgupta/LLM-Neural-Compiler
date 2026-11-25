from __future__ import annotations

from pathlib import Path

import pytest

from nnc.artifact import benchmark_artifact, load_ort_artifact


def test_load_tiny_fixture_and_infer(tiny_path) -> None:
    artifact = load_ort_artifact(tiny_path, graph_opt="disable")
    assert artifact.input_shape == (1, 1, 8, 8)
    assert "CPUExecutionProvider" in artifact.providers
    record = benchmark_artifact(artifact, warmup=1, iters=3)
    assert record["backend"] == "ort_cpu"
    assert record["latency_ms"]["mean"] > 0
    assert record["source"] == "synthetic"
    assert record["throughput_ips"] > 0


def test_missing_artifact_is_file_not_found(tmp_path) -> None:
    missing = tmp_path / "nope.onnx"
    with pytest.raises(FileNotFoundError):
        load_ort_artifact(missing)


@pytest.mark.skipif(
    not Path("experiments/models/yolov8n.onnx").is_file(),
    reason="YOLOv8n ONNX is gitignored and not present",
)
def test_load_yolov8n_ort_artifact() -> None:
    path = Path("experiments/models/yolov8n.onnx")
    artifact = load_ort_artifact(path, graph_opt="extended")
    assert artifact.input_shape[1:] == (3, 640, 640)
    record = benchmark_artifact(artifact, warmup=0, iters=1)
    assert record["latency_ms"]["mean"] > 0
    assert record["model"]["sha256"] == artifact.sha256
