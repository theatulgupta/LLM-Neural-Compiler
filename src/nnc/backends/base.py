from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BackendSkip:
    backend: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"backend": self.backend, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """Backend-neutral input metadata. Compilers must not poke session objects."""

    name: str
    shape: tuple[int | None, ...]
    dtype: str = "float32"

    def numpy_shape(self, fill: int = 1) -> tuple[int, ...]:
        return tuple(dim if isinstance(dim, int) and dim > 0 else fill for dim in self.shape)


@dataclass(frozen=True, slots=True)
class CompiledModel:
    backend: str
    session: Any
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    providers: tuple[str, ...]
    inputs: tuple[TensorSpec, ...] = ()


class Backend(ABC):
    name: str

    @abstractmethod
    def available(self) -> tuple[bool, str | None]:
        """Return (True, None) or (False, skip reason)."""

    @abstractmethod
    def compile(self, model_bytes: bytes, *, graph_opt: str) -> CompiledModel:
        """Build an executable. Must raise if available() is False."""

    @abstractmethod
    def infer(self, compiled: CompiledModel, feeds: dict[str, Any]) -> list[Any]:
        """Run one inference."""
