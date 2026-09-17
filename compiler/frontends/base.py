"""One importer. The pipeline does not switch on file suffixes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from compiler.graph.graph_loader import LoadedGraph


class Frontend(ABC):
    name: str
    suffixes: tuple[str, ...] = ()

    def available(self) -> tuple[bool, str | None]:
        return True, None

    @abstractmethod
    def ingest(self, path: Path, *, check: bool = True) -> LoadedGraph:
        ...
