"""CPU-compatible ONNX model quantization helpers (FP32, INT8 dynamic/static, optional FP16)."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import onnx


class _ArrayCalibrationReader:
    """Minimal CalibrationDataReader compatible with onnxruntime.quantization."""

    def __init__(self, input_name: str, samples: Iterable[np.ndarray]) -> None:
        self._input_name = input_name
        self._iter = iter(samples)

    def get_next(self) -> dict[str, np.ndarray] | None:
        sample = next(self._iter, None)
        if sample is None:
            return None
        return {self._input_name: sample}


def quantize_model_bytes(
    model_bytes: bytes,
    precision: str,
    *,
    calibration_samples: list[np.ndarray] | None = None,
) -> bytes:
    """Transform serialized ONNX bytes according to an allowlisted precision mode."""

    if precision == "fp32":
        return model_bytes

    if precision == "fp16":
        try:
            from onnxconverter_common import float16
        except ImportError as exc:
            raise RuntimeError(
                "fp16 precision requires 'onnxconverter-common' to be installed"
            ) from exc
        model = onnx.load_from_string(model_bytes)
        fp16_model = float16.convert_float_to_fp16(model, keep_io_types=True)
        return fp16_model.SerializeToString()

    if precision in {"int8_dynamic", "int8_static"}:
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic, quantize_static

        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = Path(tmpdir) / "in.onnx"
            out_path = Path(tmpdir) / "out.onnx"
            in_path.write_bytes(model_bytes)

            if precision == "int8_dynamic":
                quantize_dynamic(
                    model_input=str(in_path),
                    model_output=str(out_path),
                    weight_type=QuantType.QInt8,
                    per_channel=False,
                )
            else:
                model = onnx.load_from_string(model_bytes)
                inits = {init.name for init in model.graph.initializer}
                inputs = [inp for inp in model.graph.input if inp.name not in inits]
                if not inputs:
                    raise RuntimeError("cannot run int8_static without a non-initializer graph input")
                inp = inputs[0]
                shape = [
                    int(d.dim_value) if d.HasField("dim_value") and d.dim_value > 0 else 1
                    for d in inp.type.tensor_type.shape.dim
                ]
                samples = calibration_samples
                if not samples:
                    rng = np.random.default_rng(0)
                    samples = [rng.random(shape, dtype=np.float32) for _ in range(8)]
                reader = _ArrayCalibrationReader(inp.name, samples)
                quantize_static(
                    model_input=str(in_path),
                    model_output=str(out_path),
                    calibration_data_reader=reader,
                    quant_format=QuantFormat.QDQ,
                    weight_type=QuantType.QInt8,
                    activation_type=QuantType.QUInt8,
                )
            return out_path.read_bytes()

    raise ValueError(f"unsupported precision mode: {precision!r}")
