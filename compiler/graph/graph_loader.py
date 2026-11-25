"""Load and validate ONNX models for static analysis and compilation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import checker

from compiler.utils import sha256_bytes, sha256_file


@dataclass(frozen=True, slots=True)
class LoadedGraph:
    model: onnx.ModelProto
    source: str
    sha256: str
    bytes: bytes

    @property
    def opset(self) -> int:
        imports = list(self.model.opset_import)
        return int(imports[0].version) if imports else 0

    @property
    def ir_version(self) -> int:
        return int(self.model.ir_version)


def load_graph(path: Path, *, check: bool = True) -> LoadedGraph:
    data = path.read_bytes()
    model = onnx.load(str(path))
    if check:
        checker.check_model(model)
    return LoadedGraph(model=model, source=str(path), sha256=sha256_file(path), bytes=data)


def load_graph_bytes(data: bytes, *, source: str = "<bytes>", check: bool = True) -> LoadedGraph:
    model = onnx.load_from_string(data)
    if check:
        checker.check_model(model)
    return LoadedGraph(model=model, source=source, sha256=sha256_bytes(data), bytes=data)
