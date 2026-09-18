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
        super().__init__(f"strategy {name!r} is not allowlisted; allowed={list(allowed)}")


class CalibrationUnavailable(CompilerError):
    """Static quantization needs frames that are not on disk."""


class FrontendSkip(CompilerError):
    """Source format is recognized but not imported into GraphIR on this host."""

    def __init__(self, frontend: str, reason: str) -> None:
        self.frontend = frontend
        self.reason = reason
        super().__init__(f"{frontend} skipped: {reason}")


class UnknownFrontendError(CompilerError):
    """No registered frontend claims this file suffix."""

    def __init__(self, suffix: str, known: tuple[str, ...]) -> None:
        self.suffix = suffix
        self.known = known
        super().__init__(f"no frontend for suffix {suffix!r}; known={list(known)}")
