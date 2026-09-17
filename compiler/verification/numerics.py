"""Compare native vs optimized outputs. Never invent accuracy."""

from __future__ import annotations

from typing import Any

import numpy as np

from nnc.postprocess import decode


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    n = min(x.size, y.size)
    if n == 0:
        return 1.0
    x, y = x[:n], y[:n]
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0:
        return 1.0 if np.allclose(x, y) else 0.0
    return float(np.dot(x, y) / denom)


def compare_outputs(ref: list[np.ndarray], cand: list[np.ndarray]) -> dict[str, Any]:
    per = []
    max_abs = 0.0
    cosine_min = 1.0
    for index, (a, b) in enumerate(zip(ref, cand)):
        aa, bb = np.asarray(a), np.asarray(b)
        if aa.shape != bb.shape:
            flat_a = aa.astype(np.float64).ravel()
            flat_b = bb.astype(np.float64).ravel()
            n = min(flat_a.size, flat_b.size)
            diff = np.abs(flat_a[:n] - flat_b[:n]) if n else np.array([np.inf])
        else:
            diff = np.abs(aa.astype(np.float64) - bb.astype(np.float64))
        row_max = float(diff.max()) if diff.size else 0.0
        row_mean = float(diff.mean()) if diff.size else 0.0
        cos = _cosine(aa, bb)
        per.append({"name": str(index), "max_abs": row_max, "mean_abs": row_mean, "cosine": cos})
        max_abs = max(max_abs, row_max)
        cosine_min = min(cosine_min, cos)
    if not per:
        max_abs = float("inf")
        cosine_min = 0.0
    return {"per_output": per, "max_abs": max_abs, "cosine_min": cosine_min}


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def task_agreement(ref_raw: list[np.ndarray], cand_raw: list[np.ndarray], task: str) -> dict[str, Any]:
    if task in {"detect", "detect-lite", "pose", "segment"}:
        ref = decode(task, ref_raw)
        cand = decode(task, cand_raw)
        boxes_r, boxes_c = ref["boxes"], cand["boxes"]
        if len(boxes_r) == 0 and len(boxes_c) == 0:
            return {"matched_ratio": 1.0, "mean_iou": 1.0, "n_ref": 0, "n_cand": 0}
        matched = 0
        ious: list[float] = []
        used = set()
        for box in boxes_r:
            best_i, best = -1, 0.0
            for index, other in enumerate(boxes_c):
                if index in used:
                    continue
                val = _iou(box, other)
                if val > best:
                    best, best_i = val, index
            if best >= 0.5 and best_i >= 0:
                matched += 1
                used.add(best_i)
                ious.append(best)
        denom = max(len(boxes_r), 1)
        return {
            "matched_ratio": matched / denom,
            "mean_iou": float(sum(ious) / len(ious)) if ious else 0.0,
            "n_ref": int(len(boxes_r)),
            "n_cand": int(len(boxes_c)),
        }
    if task == "classify":
        a = decode("classify", ref_raw)["class_id"]
        b = decode("classify", cand_raw)["class_id"]
        return {"top1_agreement": 1.0 if a == b else 0.0, "ref": a, "cand": b}
    if task == "depth":
        r = np.asarray(ref_raw[0], dtype=np.float64)
        c = np.asarray(cand_raw[0], dtype=np.float64)
        n = min(r.size, c.size)
        rel = np.abs(r.ravel()[:n] - c.ravel()[:n]) / np.maximum(np.abs(r.ravel()[:n]), 1e-6)
        return {"relative_error_mean": float(rel.mean()) if n else 1.0}
    cmp_ = compare_outputs(ref_raw, cand_raw)
    return {"max_abs": cmp_["max_abs"], "cosine_min": cmp_["cosine_min"]}
