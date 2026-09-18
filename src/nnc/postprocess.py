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


def yolo_decode_nms(
    raw: np.ndarray,
    *,
    num_classes: int | None = None,
    conf: float = 0.25,
    iou: float = 0.45,
    keypoints_dim: int = 0,
) -> dict[str, Any]:
    """Ultralytics export layout is [C, N]: xywh, then class scores, then extras.

    `num_classes` is the class-score block. Pose uses 1 class then 17*3 keypoints.
    Segment uses 80 classes; mask coefficients after that are ignored (boxes only).
    """
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.size == 0:
        empty = {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "scores": np.zeros((0,), dtype=np.float32),
            "classes": np.zeros((0,), dtype=np.int32),
        }
        if keypoints_dim:
            empty["keypoints"] = np.zeros((0, keypoints_dim // 3, 3), dtype=np.float32)
        return empty
    # Prefer [C, N] (C << N). A [N, C] layout is transposed.
    if arr.shape[0] > arr.shape[1]:
        arr = arr.T
    rows = int(arr.shape[0])
    if num_classes is None:
        num_classes = max(1, rows - 4 - keypoints_dim)
    cls_end = 4 + int(num_classes)
    xywh = arr[:4]
    cls = arr[4:cls_end]
    if cls.size == 0:
        scores = np.zeros((arr.shape[1],), dtype=np.float32)
        classes = np.zeros((arr.shape[1],), dtype=np.int32)
    else:
        scores = cls.max(axis=0)
        classes = cls.argmax(axis=0)
    keep = scores >= conf
    xywh, scores, classes = xywh[:, keep], scores[keep], classes[keep]
    kpts = None
    if keypoints_dim and rows >= cls_end + keypoints_dim:
        kpts = arr[cls_end : cls_end + keypoints_dim][:, keep]
    cx, cy, w, h = xywh
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    keep_idx = _nms(boxes, scores, iou)
    out: dict[str, Any] = {
        "boxes": boxes[keep_idx].astype(np.float32),
        "scores": scores[keep_idx].astype(np.float32),
        "classes": classes[keep_idx].astype(np.int32),
    }
    if kpts is not None:
        n = len(keep_idx)
        out["keypoints"] = kpts[:, keep_idx].T.reshape(n, keypoints_dim // 3, 3).astype(np.float32)
    return out


def ssdlite_default_boxes(
    imgsz: int = 320,
    feature_maps: tuple[int, ...] = (20, 10, 5, 3, 2, 1),
    min_ratio: float = 0.2,
    max_ratio: float = 0.95,
    aspect_ratios: tuple[float, ...] = (2.0, 3.0),
    steps: tuple[int, ...] = (16, 32, 64, 107, 160, 320),
) -> np.ndarray:
    """SSDLite320 DefaultBoxGenerator boxes in xyxy on the letterbox canvas.

    Feature maps 20,10,5,3,2,1 with 6 anchors per cell = 3234. Steps follow
    torchvision SSDLite320 (320/3 rounded to 107).
    """
    n_maps = len(feature_maps)
    scales = [min_ratio + (max_ratio - min_ratio) * k / (n_maps - 1) for k in range(n_maps)]
    scales.append(1.0)
    boxes: list[list[float]] = []
    for k, f in enumerate(feature_maps):
        step = float(steps[k])
        s_k = scales[k]
        s_prime = float(np.sqrt(s_k * scales[k + 1]))
        sizes = [(s_k, s_k), (s_prime, s_prime)]
        for ar in aspect_ratios:
            sizes.append((s_k * np.sqrt(ar), s_k / np.sqrt(ar)))
            sizes.append((s_k / np.sqrt(ar), s_k * np.sqrt(ar)))
        for i in range(f):
            for j in range(f):
                cx = (j + 0.5) * step / imgsz
                cy = (i + 0.5) * step / imgsz
                for ws, hs in sizes:
                    boxes.append([cx - ws / 2, cy - hs / 2, cx + ws / 2, cy + hs / 2])
    arr = np.asarray(boxes, dtype=np.float32)
    return np.clip(arr, 0.0, 1.0)


_SSDLITE_BOXES: np.ndarray | None = None


def _ssdlite_priors() -> np.ndarray:
    global _SSDLITE_BOXES
    if _SSDLITE_BOXES is None:
        _SSDLITE_BOXES = ssdlite_default_boxes()
    return _SSDLITE_BOXES


def ssdlite_decode(
    bbox_regression: np.ndarray,
    cls_logits: np.ndarray,
    *,
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 320,
) -> dict[str, Any]:
    """Decode SSDLite heads. BoxCoder weights (10, 10, 5, 5). Class-agnostic NMS.

    torchvision uses per-class NMS; we document the simpler class-agnostic pass.
    Boxes are returned in letterbox pixels (imgsz x imgsz).
    """
    priors = _ssdlite_priors()
    deltas = np.asarray(bbox_regression, dtype=np.float32)
    logits = np.asarray(cls_logits, dtype=np.float32)
    if deltas.ndim == 3:
        deltas = deltas[0]
    if logits.ndim == 3:
        logits = logits[0]
    n = min(len(priors), len(deltas), len(logits))
    priors = priors[:n]
    deltas = deltas[:n]
    logits = logits[:n]
    pw = priors[:, 2] - priors[:, 0]
    ph = priors[:, 3] - priors[:, 1]
    pcx = (priors[:, 0] + priors[:, 2]) * 0.5
    pcy = (priors[:, 1] + priors[:, 3]) * 0.5
    dx, dy, dw, dh = deltas[:, 0] / 10.0, deltas[:, 1] / 10.0, deltas[:, 2] / 5.0, deltas[:, 3] / 5.0
    cx = dx * pw + pcx
    cy = dy * ph + pcy
    w = np.exp(np.clip(dw, -4.0, 4.0)) * pw
    h = np.exp(np.clip(dh, -4.0, 4.0)) * ph
    boxes01 = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes = np.clip(boxes01, 0.0, 1.0) * float(imgsz)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-9, None)
    scores = probs[:, 1:].max(axis=1)
    classes = probs[:, 1:].argmax(axis=1) + 1  # drop background class 0
    keep = scores >= conf
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
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
    tensor = np.asarray(raw)
    payload: dict[str, Any] = {"tensor": tensor}
    if tensor.size:
        payload["min"] = float(np.min(tensor))
        payload["max"] = float(np.max(tensor))
        payload["mean"] = float(np.mean(tensor))
    return payload


def boxes_to_original(boxes: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    """Map letterbox-pixel xyxy back to the original image."""
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.size == 0:
        return arr.reshape(0, 4)
    pad = meta.get("pad", (0, 0))
    pad_x, pad_y = float(pad[0]), float(pad[1])
    scale = float(meta.get("scale") or 1.0)
    out = arr.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_x) / max(scale, 1e-6)
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_y) / max(scale, 1e-6)
    return out


def decode(task: str, outputs: list[np.ndarray], *, conf: float = 0.25, iou: float = 0.45) -> dict[str, Any]:
    first = outputs[0] if outputs else np.zeros((0,))
    if task == "detect":
        return yolo_decode_nms(first, num_classes=None, conf=conf, iou=iou)
    if task == "pose":
        return yolo_decode_nms(first, num_classes=1, conf=conf, iou=iou, keypoints_dim=51)
    if task == "segment":
        # 80 COCO classes; rows 84:116 are mask coefficients and stay unused.
        det = yolo_decode_nms(first, num_classes=80, conf=conf, iou=iou)
        det["mask_decoded"] = False
        return det
    if task == "detect-lite":
        bbox = outputs[0] if outputs else np.zeros((0, 4))
        logits = outputs[1] if len(outputs) > 1 else np.zeros((0, 91))
        return ssdlite_decode(bbox, logits, conf=conf, iou=iou)
    if task == "classify":
        return classify_top1(first)
    return passthrough(first)
