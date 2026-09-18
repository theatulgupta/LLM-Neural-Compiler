"""ORT quantization wrappers. Failures raise; callers record JSON."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx

from compiler.errors import CalibrationUnavailable


def quantize_dynamic_int8(model: onnx.ModelProto, params: dict) -> onnx.ModelProto:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    weight = str(params.get("weight_type") or "qint8")
    weight_type = QuantType.QInt8 if weight == "qint8" else QuantType.QUInt8
    per_channel = bool(params.get("per_channel", True))
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.onnx"
        dst = Path(tmp) / "out.onnx"
        onnx.save(model, str(src))
        quantize_dynamic(str(src), str(dst), weight_type=weight_type, per_channel=per_channel)
        return onnx.load(str(dst))


class _Reader:
    def __init__(self, session_inputs: list[str], frames: list[np.ndarray]) -> None:
        self._names = session_inputs
        self._frames = frames
        self._i = 0

    def get_next(self):
        if self._i >= len(self._frames):
            return None
        item = {self._names[0]: self._frames[self._i]}
        self._i += 1
        return item

    def rewind(self) -> None:
        self._i = 0


def _calibration_tensors(params: dict, input_shape: tuple[int, ...]) -> list[np.ndarray]:
    from compiler.data.calibration import load_calibration_nchw

    source = str(params.get("calibration") or "gz_frames")
    try:
        return load_calibration_nchw(source=source, input_shape=input_shape, limit=16)
    except CalibrationUnavailable:
        if source == "gz_frames":
            return load_calibration_nchw(source="assets", input_shape=input_shape, limit=16)
        raise


def quantize_static_int8(model: onnx.ModelProto, params: dict) -> onnx.ModelProto:
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

    graph_input = next(
        item for item in model.graph.input if item.name not in {i.name for i in model.graph.initializer}
    )
    dims = []
    for dim in graph_input.type.tensor_type.shape.dim:
        dims.append(int(dim.dim_value) if dim.HasField("dim_value") and dim.dim_value > 0 else 1)
    frames = _calibration_tensors(params, tuple(dims))

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._inner = _Reader([graph_input.name], frames)

        def get_next(self):
            return self._inner.get_next()

        def rewind(self) -> None:
            self._inner.rewind()

    fmt = QuantFormat.QDQ if str(params.get("format") or "qdq") == "qdq" else QuantFormat.QOperator
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.onnx"
        dst = Path(tmp) / "out.onnx"
        onnx.save(model, str(src))
        quantize_static(
            str(src),
            str(dst),
            Reader(),
            quant_format=fmt,
            per_channel=bool(params.get("per_channel", True)),
            weight_type=QuantType.QInt8,
        )
        return onnx.load(str(dst))
