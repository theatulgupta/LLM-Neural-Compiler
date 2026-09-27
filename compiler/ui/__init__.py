"""Local operator console. Reads measured JSON and starts the existing pipeline."""

from compiler.ui.jobs import JobRunner
from compiler.ui.server import serve
from compiler.ui.view import build_dashboard

__all__ = ["JobRunner", "build_dashboard", "serve"]
