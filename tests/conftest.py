from __future__ import annotations

import os
from pathlib import Path

import pytest

from compiler.parsers.tiny_cnn import write_tiny_cnn
from compiler.parsers.tiny_depth import write_tiny_depth


@pytest.fixture(autouse=True)
def _pin_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests off the network even when a Groq key exists on the host."""

    monkeypatch.setenv("NNC_LLM", os.environ.get("NNC_LLM", "heuristic"))


@pytest.fixture
def tiny_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_cnn.onnx"
    write_tiny_cnn(path)
    return path


@pytest.fixture
def tiny_depth_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_depth.onnx"
    write_tiny_depth(path)
    return path


@pytest.fixture
def broken_k16_path(tmp_path: Path) -> Path:
    path = tmp_path / "tiny_cnn_k16.onnx"
    write_tiny_cnn(path, gemm_k=16, validate=False)
    return path
