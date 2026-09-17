"""CLI package. ``python -m compiler`` and the ``nnc`` console script enter here."""

from compiler.cli.app import build_parser, main

__all__ = ["build_parser", "main"]
