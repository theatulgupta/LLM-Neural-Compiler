"""Typed errors for the compiler pipeline."""

from __future__ import annotations


class CompilerError(Exception):
    """Base error for compile-time failures."""


class FixtureError(CompilerError):
    """ONNX fixture is invalid or internally inconsistent."""


class UnknownStrategyError(CompilerError):
    """LLM or caller requested a strategy outside the allowlist."""

    def __init__(self, name: str, allowed: tuple[str, ...]) -> None:
        self.name = name
        self.allowed = allowed
        super().__init__(
            f"strategy {name!r} is not allowlisted; allowed={list(allowed)}"
        )


class BackendSkipped(CompilerError):
    """A backend cannot run on this host; callers must record the reason."""

    def __init__(self, backend: str, reason: str) -> None:
        self.backend = backend
        self.reason = reason
        super().__init__(f"{backend} skipped: {reason}")
