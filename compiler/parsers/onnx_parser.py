"""Load and validate ONNX protobufs."""

from __future__ import annotations

from pathlib import Path

import onnx


def parse_onnx_bytes(data: bytes, *, check: bool = True) -> onnx.ModelProto:
    model = onnx.load_from_string(data)
    if check:
        onnx.checker.check_model(model, full_check=True)
    return model


def parse_onnx_path(path: Path, *, check: bool = True) -> onnx.ModelProto:
    model = onnx.load(str(path))
    if check:
        onnx.checker.check_model(model, full_check=True)
    return model
