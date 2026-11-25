from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper

from compiler.errors import FixtureError
from compiler.parsers.tiny_cnn import flatten_features, write_tiny_cnn


def test_flatten_width_is_64() -> None:
    assert flatten_features() == 64


def test_default_gemm_k_matches_flatten(tiny_path) -> None:
    model = onnx.load(str(tiny_path))
    weights = {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}
    gemm_w = weights["gemm_w"]
    assert gemm_w.shape == (8, 64)
    assert gemm_w.dtype == np.float32


def test_refuses_to_emit_k16_without_broken_flag(tmp_path) -> None:
    path = tmp_path / "bad.onnx"
    try:
        write_tiny_cnn(path, gemm_k=16, validate=True)
    except FixtureError as exc:
        assert "64" in str(exc)
        assert "16" in str(exc)
    else:
        raise AssertionError("expected FixtureError for Gemm K=16 vs Flatten 64")
