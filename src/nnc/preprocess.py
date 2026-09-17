"""Shared image letterbox (numpy only). Used by the compiler and the ROS node."""

from __future__ import annotations

from typing import Any

import numpy as np


def letterbox(
    img_hwc_uint8: np.ndarray,
    imgsz: int,
    *,
    rgb: bool = True,
    pad_value: int = 114,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resize with stride-2 when divisible, else nearest-index; pad to square; NCHW /255."""

    img = np.asarray(img_hwc_uint8)
    if img.ndim != 3:
        raise ValueError(f"expected HWC image, got shape {img.shape}")
    if not rgb:
        img = img[:, :, ::-1]
    height, width = int(img.shape[0]), int(img.shape[1])
    target = int(imgsz)
    scale = min(target / max(1, height), target / max(1, width))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    if new_w == width // 2 and width % 2 == 0 and new_h == height // 2 and height % 2 == 0:
        resized = img[::2, ::2]
        new_h, new_w = resized.shape[0], resized.shape[1]
    elif (new_h, new_w) == (height, width):
        resized = img
    else:
        row_idx = np.clip((np.arange(new_h) * height / new_h).astype(np.int32), 0, height - 1)
        col_idx = np.clip((np.arange(new_w) * width / new_w).astype(np.int32), 0, width - 1)
        resized = img[row_idx][:, col_idx]
    canvas = np.full((target, target, 3), pad_value, dtype=np.uint8)
    pad_y = (target - new_h) // 2
    pad_x = (target - new_w) // 2
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    nchw = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None]
    meta = {"scale": float(scale), "pad": (int(pad_x), int(pad_y)), "resized": (new_w, new_h), "imgsz": target}
    return nchw, meta
