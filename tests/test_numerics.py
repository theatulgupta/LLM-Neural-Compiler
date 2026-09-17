from __future__ import annotations

import numpy as np

from compiler.verification.numerics import compare_outputs
from compiler.verification.gates import evaluate_gates


def test_identical_outputs_zero_abs() -> None:
    a = [np.ones((2, 2), dtype=np.float32)]
    cmp_ = compare_outputs(a, a)
    assert cmp_["max_abs"] == 0.0
    gate = evaluate_gates(numerics_kind="exact", compare=cmp_, agreement={}, task="fixture", ref_scale=1.0)
    assert gate["passed"] is True


def test_perturbed_fails_exact() -> None:
    a = [np.ones((2, 2), dtype=np.float32)]
    b = [np.ones((2, 2), dtype=np.float32) + 1.0]
    cmp_ = compare_outputs(a, b)
    gate = evaluate_gates(numerics_kind="exact", compare=cmp_, agreement={}, task="fixture", ref_scale=1.0)
    assert gate["passed"] is False
