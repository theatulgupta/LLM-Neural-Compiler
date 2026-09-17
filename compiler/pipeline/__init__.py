"""Compile / optimize / matrix entry points. Callers import this package, not file names."""

from compiler.pipeline.compile import analyze_only, compile_and_benchmark, compile_verify_profile
from compiler.pipeline.matrix import measure_path, measure_spec, run_zoo_matrix
from compiler.pipeline.optimize import optimize_model
from nnc.backends import get_backend

__all__ = [
    "analyze_only",
    "compile_and_benchmark",
    "compile_verify_profile",
    "get_backend",
    "measure_path",
    "measure_spec",
    "optimize_model",
    "run_zoo_matrix",
]
