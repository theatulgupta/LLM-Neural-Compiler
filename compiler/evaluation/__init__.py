"""Evaluation utilities: numerical fidelity, baselines, and oracle regret."""

from __future__ import annotations

from compiler.evaluation.fidelity import compare_outputs, evaluate_top1_agreement

__all__ = ["compare_outputs", "evaluate_top1_agreement"]
