"""Numerical equivalence and task-accuracy verification against the FP32 baseline."""

from __future__ import annotations

import numpy as np


def compare_outputs(
    baseline_outputs: list[np.ndarray],
    candidate_outputs: list[np.ndarray],
) -> dict[str, float]:
    """Compute MAE, Max Absolute Error, and Cosine Similarity across output tensors."""

    if len(baseline_outputs) != len(candidate_outputs):
        raise ValueError(
            f"output tensor count mismatch: baseline={len(baseline_outputs)} candidate={len(candidate_outputs)}"
        )

    base_flat = np.concatenate([np.asarray(arr, dtype=np.float64).ravel() for arr in baseline_outputs])
    cand_flat = np.concatenate([np.asarray(arr, dtype=np.float64).ravel() for arr in candidate_outputs])

    if base_flat.shape != cand_flat.shape:
        raise ValueError(f"output shape mismatch: baseline={base_flat.shape} candidate={cand_flat.shape}")

    diff = np.abs(base_flat - cand_flat)
    mae = float(np.mean(diff)) if diff.size else 0.0
    max_abs = float(np.max(diff)) if diff.size else 0.0

    denom = float(np.linalg.norm(base_flat) * np.linalg.norm(cand_flat))
    if denom == 0.0:
        cosine_sim = 1.0 if np.allclose(base_flat, cand_flat) else 0.0
    else:
        cosine_sim = float(np.dot(base_flat, cand_flat) / denom)

    return {
        "mae": mae,
        "max_abs_error": max_abs,
        "cosine_similarity": cosine_sim,
    }


def evaluate_top1_agreement(
    baseline_logits: list[np.ndarray],
    candidate_logits: list[np.ndarray],
) -> float:
    """Fraction of samples whose argmax class prediction matches the FP32 baseline."""

    if not baseline_logits or len(baseline_logits) != len(candidate_logits):
        raise ValueError("baseline_logits and candidate_logits must be non-empty and equal length")
    matches = sum(
        int(np.argmax(b) == np.argmax(c)) for b, c in zip(baseline_logits, candidate_logits, strict=True)
    )
    return float(matches / len(baseline_logits))
