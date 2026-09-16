"""Load an ONNX graph into a thin wrapper used by analysis and compile."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnx

from compiler.parsers.onnx_parser import parse_onnx_bytes, parse_onnx_path
from compiler.utils.hashing import sha256_bytes, sha256_file


@dataclass(frozen=True, slots=True)
class LoadedGraph:
    model: onnx.ModelProto
    source: str
    sha256: str
    bytes: bytes

    @property
    def opset(self) -> int:
        imports = list(self.model.opset_import)
        if not imports:
            return 0
        return int(imports[0].version)

    @property
    def ir_version(self) -> int:
        return int(self.model.ir_version)


def load_graph(path: Path, *, check: bool = True) -> LoadedGraph:
    data = path.read_bytes()
    model = parse_onnx_path(path, check=check)
    return LoadedGraph(model=model, source=str(path), sha256=sha256_file(path), bytes=data)


def load_graph_bytes(data: bytes, *, source: str = "<bytes>", check: bool = True) -> LoadedGraph:
    model = parse_onnx_bytes(data, check=check)
    return LoadedGraph(model=model, source=source, sha256=sha256_bytes(data), bytes=data)
