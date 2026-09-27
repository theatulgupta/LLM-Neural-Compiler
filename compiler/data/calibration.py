"""Calibration images for static quantization and numeric checks."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from compiler.catalog import REPO_ROOT
from compiler.errors import CalibrationUnavailable

CALIB_NPZ = REPO_ROOT / "experiments" / "calib" / "gz_frames.npz"


def calib_npz_path() -> Path:
    """NNC_CALIB_NPZ wins; otherwise experiments/calib/gz_frames.npz."""

    override = os.environ.get("NNC_CALIB_NPZ", "").strip()
    if override:
        return Path(override)
    return CALIB_NPZ


def ultralytics_asset_dirs() -> list[Path]:
    """Installed package assets, plus NNC_ULTRALYTICS_ASSETS when set."""

    found: list[Path] = []
    override = os.environ.get("NNC_ULTRALYTICS_ASSETS", "").strip()
    if override:
        found.append(Path(override))
    try:
        import ultralytics
    except ImportError:
        return found
    found.append(Path(ultralytics.__file__).resolve().parent / "assets")
    return found


def _letterbox_nchw(rgb: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    from nnc.preprocess import letterbox

    imgsz = int(shape[-1] if len(shape) >= 4 else 640)
    nchw, _meta = letterbox(rgb, imgsz, rgb=True)
    # Match channel count.
    if len(shape) == 4 and shape[1] == 1 and nchw.shape[1] == 3:
        nchw = nchw.mean(axis=1, keepdims=True)
    if len(shape) == 4 and tuple(nchw.shape) != tuple(
        d if d > 0 else nchw.shape[i] for i, d in enumerate(shape)
    ):
        # Resize channels/spatial with nearest if the model is not square 3xHxW.
        target = tuple(int(d) if d and d > 0 else nchw.shape[i] for i, d in enumerate(shape))
        if nchw.shape != target:
            tiled = np.zeros(target, dtype=np.float32)
            c = min(nchw.shape[1], target[1])
            h = min(nchw.shape[2], target[2])
            w = min(nchw.shape[3], target[3])
            tiled[:, :c, :h, :w] = nchw[:, :c, :h, :w]
            nchw = tiled
    return nchw.astype(np.float32, copy=False)


def _load_asset_rgbs(limit: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for folder in ultralytics_asset_dirs():
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png")):
            try:
                from PIL import Image
            except ImportError:
                break
            img = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
            frames.append(img)
            if len(frames) >= limit:
                return frames
    return frames


def load_calibration_rgb(*, source: str = "gz_frames", limit: int = 16) -> tuple[list[np.ndarray], str]:
    npz = calib_npz_path()
    if source == "gz_frames" and npz.is_file():
        data = np.load(npz)
        stacked = data["frames"]
        frames = [np.asarray(stacked[i]) for i in range(min(limit, stacked.shape[0]))]
        if frames:
            return frames, "gz_frames"
    assets = _load_asset_rgbs(limit)
    if assets:
        return assets, "assets"
    raise CalibrationUnavailable(
        "no calibration frames (gz_frames.npz missing and ultralytics assets missing)"
    )


def load_calibration_nchw(*, source: str, input_shape: tuple[int, ...], limit: int = 16) -> list[np.ndarray]:
    frames, _used = load_calibration_rgb(source=source, limit=limit)
    return [_letterbox_nchw(frame, input_shape) for frame in frames]
