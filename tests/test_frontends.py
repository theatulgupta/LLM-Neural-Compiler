from __future__ import annotations

from pathlib import Path

import pytest

from compiler.errors import FrontendSkip, UnknownFrontendError
from compiler.graph.graph_loader import load_graph


def test_onnx_ingest(tiny_path) -> None:
    loaded = load_graph(tiny_path)
    assert loaded.origin_format == "onnx"
    assert loaded.ir == "onnx"
    assert loaded.opset > 0


def test_torch_weights_skip(tmp_path: Path) -> None:
    path = tmp_path / "weights.pt"
    path.write_bytes(b"not-a-graph")
    with pytest.raises(FrontendSkip) as exc:
        load_graph(path)
    assert exc.value.frontend == "torch"


def test_unknown_suffix(tmp_path: Path) -> None:
    path = tmp_path / "model.xyz"
    path.write_bytes(b"x")
    with pytest.raises(UnknownFrontendError):
        load_graph(path)
