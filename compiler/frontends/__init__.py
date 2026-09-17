"""Source-format loaders. New importers register in this package."""

from __future__ import annotations

from compiler.frontends.base import Frontend
from compiler.frontends.onnx_frontend import OnnxFrontend
from compiler.frontends.registry import (
    frontend_catalog,
    ingest,
    known_frontends,
    known_suffixes,
    register_frontend,
    unregister_frontend,
)

__all__ = [
    "Frontend",
    "OnnxFrontend",
    "frontend_catalog",
    "ingest",
    "known_frontends",
    "known_suffixes",
    "register_frontend",
    "unregister_frontend",
]
