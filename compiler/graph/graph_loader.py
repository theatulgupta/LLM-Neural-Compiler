"""Load a model into GraphIR (ONNX today)."""

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
    origin_format: str = "onnx"
    ir: str = "onnx"

    @property
    def opset(self) -> int:
        imports = list(self.model.opset_import)
        if not imports:
            return 0
        return int(imports[0].version)

    @property
    def ir_version(self) -> int:
        return int(self.model.ir_version)


def from_onnx_bytes(
    model: onnx.ModelProto,
    data: bytes,
    *,
    source: str,
    origin_format: str = "onnx",
    sha256: str | None = None,
) -> LoadedGraph:
    return LoadedGraph(
        model=model,
        source=source,
        sha256=sha256 or sha256_bytes(data),
        bytes=data,
        origin_format=origin_format,
        ir="onnx",
    )


def load_graph(path: Path, *, check: bool = True) -> LoadedGraph:
    """Ingest any registered source format into GraphIR."""

    from compiler.frontends import ingest

    return ingest(path, check=check)


def load_graph_bytes(data: bytes, *, source: str = "<bytes>", check: bool = True) -> LoadedGraph:
    model = parse_onnx_bytes(data, check=check)
    return from_onnx_bytes(model, data, source=source, origin_format="onnx")


def load_onnx_path(path: Path, *, check: bool = True) -> LoadedGraph:
    """Direct ONNX load used by the onnx frontend (avoids ingest recursion)."""

    data = path.read_bytes()
    model = parse_onnx_path(path, check=check)
    return from_onnx_bytes(model, data, source=str(path), origin_format="onnx", sha256=sha256_file(path))
