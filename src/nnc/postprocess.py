"""Task-specific decode. Numpy only; no claimed FPS."""

from __future__ import annotations

from typing import Any

import numpy as np


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[order[1:]] - inter
        iou = np.divide(inter, np.maximum(union, 1e-9))
        order = order[1:][iou <= iou_thr]
    return keep


def yolo_decode_nms(raw: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> dict[str, Any]:
    arr = np.asarray(raw)
    if arr.ndim == 3:
        arr = arr[0]
    # Ultralytics export is often [84, N] = xywh + classes
    if arr.shape[0] < arr.shape[1]:
        arr = arr
    else:
        arr = arr.T if arr.shape[0] > 4 and arr.shape[1] <= 32 else arr
    if arr.shape[0] >= 4 and arr.shape[1] > arr.shape[0]:
        # [C, N]
        xywh = arr[:4]
        cls = arr[4:]
        scores = cls.max(axis=0)
        classes = cls.argmax(axis=0)
        keep = scores >= conf
        xywh, scores, classes = xywh[:, keep], scores[keep], classes[keep]
        cx, cy, w, h = xywh
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    else:
        return {"boxes": np.zeros((0, 4), dtype=np.float32), "scores": np.zeros((0,), dtype=np.float32), "classes": np.zeros((0,), dtype=np.int32)}
    keep_idx = _nms(boxes, scores, iou)
    return {
        "boxes": boxes[keep_idx].astype(np.float32),
        "scores": scores[keep_idx].astype(np.float32),
        "classes": classes[keep_idx].astype(np.int32),
    }


def classify_top1(raw: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(raw).reshape(-1)
    idx = int(arr.argmax()) if arr.size else -1
    return {"class_id": idx, "score": float(arr[idx]) if idx >= 0 else 0.0}


def passthrough(raw: np.ndarray) -> dict[str, Any]:
    return {"tensor": np.asarray(raw)}


def decode(task: str, outputs: list[np.ndarray], *, conf: float = 0.25, iou: float = 0.45) -> dict[str, Any]:
    first = outputs[0] if outputs else np.zeros((0,))
    if task in {"detect", "detect-lite", "pose", "segment"}:
        return yolo_decode_nms(first, conf=conf, iou=iou)
    if task == "classify":
        return classify_top1(first)
    return passthrough(first)
