"""PIL + NumPy letterbox calibration data reader for ONNX Runtime static INT8 quantization."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from onnxruntime.quantization import CalibrationDataReader
from PIL import Image


def letterbox_pil(
    image: Image.Image,
    target_hw: tuple[int, int] = (640, 640),
    fill_value: int = 114,
) -> np.ndarray:
    """Resize and pad a PIL RGB image to target_hw (H, W) preserving aspect ratio."""

    th, tw = target_hw
    iw, ih = image.size
    scale = min(tw / iw, th / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    resized = image.convert("RGB").resize((nw, nh), Image.Resampling.BILINEAR)

    canvas = Image.new("RGB", (tw, th), (fill_value, fill_value, fill_value))
    pad_w = (tw - nw) // 2
    pad_h = (th - nh) // 2
    canvas.paste(resized, (pad_w, pad_h))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(chw, dtype=np.float32)


class ImageCalibrationDataReader(CalibrationDataReader):
    """CalibrationDataReader yielding 640x640 letterboxed float32 NCHW tensors."""

    def __init__(
        self,
        image_paths: Iterable[Path | str],
        input_name: str,
        target_hw: tuple[int, int] = (640, 640),
    ) -> None:
        self._paths = [Path(p) for p in image_paths]
        self._input_name = input_name
        self._target_hw = target_hw
        self._iter = self._make_iter()

    def _make_iter(self):
        for path in self._paths:
            with Image.open(path) as img:
                yield {self._input_name: letterbox_pil(img, self._target_hw)}

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._iter, None)

    def rewind(self) -> None:
        self._iter = self._make_iter()
