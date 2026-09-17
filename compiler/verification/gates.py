"""Pass/fail gates for exact vs approximate plans."""

from __future__ import annotations

from typing import Any


def evaluate_gates(
    *,
    numerics_kind: str,
    compare: dict[str, Any],
    agreement: dict[str, Any],
    task: str,
    ref_scale: float,
) -> dict[str, Any]:
    if numerics_kind == "exact":
        limit = 1e-4 * max(1.0, abs(ref_scale))
        passed = float(compare.get("max_abs") or 0) <= limit
        return {"passed": passed, "gate_used": "exact_max_abs", "limit": limit}
    cosine_ok = float(compare.get("cosine_min") or 0) >= 0.99
    if task in {"detect", "detect-lite", "pose", "segment"}:
        agr_ok = float(agreement.get("matched_ratio") or 0) >= 0.90
        passed = cosine_ok and agr_ok
        return {"passed": passed, "gate_used": "approx_detect", "cosine_ok": cosine_ok, "agreement_ok": agr_ok}
    if task == "classify":
        agr_ok = float(agreement.get("top1_agreement") or 0) >= 0.95
        passed = cosine_ok and agr_ok
        return {"passed": passed, "gate_used": "approx_classify", "cosine_ok": cosine_ok, "agreement_ok": agr_ok}
    passed = cosine_ok
    return {"passed": passed, "gate_used": "approx_cosine", "cosine_ok": cosine_ok}
