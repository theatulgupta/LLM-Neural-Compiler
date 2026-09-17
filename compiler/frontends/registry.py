"""Load a source file into GraphIR. ONNX is implemented; other suffixes skip with a reason."""

from __future__ import annotations

from pathlib import Path

from compiler.errors import FrontendSkip, UnknownFrontendError
from compiler.frontends.base import Frontend
from compiler.graph.graph_loader import LoadedGraph, load_onnx_path

_FRONTENDS: dict[str, type[Frontend]] = {}

# Not GraphIR yet. Recognise the suffix so the CLI can skip instead of crashing.
_SKIP = {
    ".pt": ("torch", "PyTorch weights. Export to ONNX, then compile."),
    ".pth": ("torch", "PyTorch weights. Export to ONNX, then compile."),
    ".tflite": ("tflite", "TFLite is a runtime format. Convert to ONNX first."),
    ".engine": ("tensorrt", ".engine is a compiled TensorRT output, not a source graph."),
    ".pb": ("tensorflow", "Convert GraphDef/SavedModel to ONNX first."),
    ".xml": ("openvino", "OpenVINO IR is a backend target. Import ONNX instead."),
}


def register_frontend(cls: type[Frontend]) -> type[Frontend]:
    name = getattr(cls, "name", "") or ""
    if not name:
        raise ValueError(f"{cls.__name__} needs a .name")
    _FRONTENDS[name] = cls
    return cls


def unregister_frontend(name: str) -> None:
    _FRONTENDS.pop(name, None)


def known_frontends() -> tuple[str, ...]:
    return tuple(sorted({*_FRONTENDS, *(item[0] for item in _SKIP.values())}))


def known_suffixes() -> tuple[str, ...]:
    found = list(_SKIP)
    for cls in _FRONTENDS.values():
        found.extend(str(s).lower() for s in getattr(cls, "suffixes", ()) or ())
    return tuple(sorted(set(found)))


def ingest(path: Path, *, check: bool = True) -> LoadedGraph:
    suffix = path.suffix.lower()
    if suffix in _SKIP:
        name, reason = _SKIP[suffix]
        raise FrontendSkip(name, reason)
    for cls in _FRONTENDS.values():
        suffixes = tuple(str(s).lower() for s in getattr(cls, "suffixes", ()) or ())
        if suffix in suffixes:
            return cls().ingest(path, check=check)
    raise UnknownFrontendError(suffix, known_suffixes())


def frontend_catalog() -> list[dict[str, object]]:
    rows = [
        {"name": name, "suffixes": list(cls.suffixes), "available": True, "reason": None}
        for name, cls in sorted(_FRONTENDS.items())
    ]
    seen = {name for name, _reason in _SKIP.values()}
    for name in sorted(seen):
        suffixes = [sfx for sfx, (n, _r) in _SKIP.items() if n == name]
        _n, reason = next(pair for sfx, pair in _SKIP.items() if pair[0] == name)
        rows.append({"name": name, "suffixes": suffixes, "available": False, "reason": reason})
    return rows
