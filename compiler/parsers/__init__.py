from __future__ import annotations

from compiler.parsers.onnx_parser import parse_onnx_bytes, parse_onnx_path
from compiler.parsers.tiny_cnn import flatten_features, write_tiny_cnn

__all__ = ["parse_onnx_bytes", "parse_onnx_path", "flatten_features", "write_tiny_cnn"]
