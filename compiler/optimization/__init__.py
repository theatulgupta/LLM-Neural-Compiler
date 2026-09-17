from __future__ import annotations

from compiler.optimization.optimizer import apply_plan, apply_strategy, prepare_for_ort
from compiler.optimization.passes import apply_pass

__all__ = ["apply_pass", "apply_plan", "apply_strategy", "prepare_for_ort"]
