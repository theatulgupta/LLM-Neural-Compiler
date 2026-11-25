from __future__ import annotations

from pathlib import Path

import pytest

from compiler.parsers.tiny_cnn import flatten_features, write_tiny_cnn


@pytest.fixture
def tiny_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_cnn.onnx"
    write_tiny_cnn(path)
    return path


@pytest.fixture
def broken_k16_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_cnn_k16.onnx"
    write_tiny_cnn(path, gemm_k=16, validate=False)
    return path
