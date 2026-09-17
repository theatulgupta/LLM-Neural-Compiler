"""ONNX → GraphIR."""

from __future__ import annotations

from pathlib import Path

from compiler.frontends.base import Frontend
from compiler.frontends.registry import register_frontend
from compiler.graph.graph_loader import LoadedGraph, load_onnx_path


@register_frontend
class OnnxFrontend(Frontend):
    name = "onnx"
    suffixes = (".onnx",)

    def ingest(self, path: Path, *, check: bool = True) -> LoadedGraph:
        return load_onnx_path(path, check=check)
