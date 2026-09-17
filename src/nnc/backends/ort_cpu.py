"""ONNX Runtime CPU backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import onnxruntime as ort

from nnc.backends.base import Backend, CompiledModel, TensorSpec
from nnc.backends.registry import register_backend

_OPT_LEVELS = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def _tensor_spec(item: Any) -> TensorSpec:
    shape: list[int | None] = []
    for dim in item.shape:
        if isinstance(dim, int) and dim > 0:
            shape.append(dim)
        else:
            shape.append(None)
    dtype = "float32" if item.type in {"tensor(float)", "tensor(float32)"} else str(item.type)
    return TensorSpec(name=item.name, shape=tuple(shape), dtype=dtype)


@register_backend
class OrtCpuBackend(Backend):
    name = "ort_cpu"

    def available(self) -> tuple[bool, str | None]:
        if "CPUExecutionProvider" not in ort.get_all_providers():
            return False, "onnxruntime was built without CPUExecutionProvider"
        return True, None

    def compile(self, model_bytes: bytes, *, graph_opt: str) -> CompiledModel:
        ok, reason = self.available()
        if not ok:
            raise RuntimeError(reason)
        if graph_opt not in _OPT_LEVELS:
            raise ValueError(f"unknown ORT graph_opt {graph_opt!r}; allowed={sorted(_OPT_LEVELS)}")

        options = ort.SessionOptions()
        options.graph_optimization_level = _OPT_LEVELS[graph_opt]
        session = ort.InferenceSession(
            model_bytes,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        specs = tuple(_tensor_spec(item) for item in session.get_inputs())
        return CompiledModel(
            backend=self.name,
            session=session,
            input_names=tuple(spec.name for spec in specs),
            output_names=tuple(item.name for item in session.get_outputs()),
            providers=tuple(session.get_providers()),
            inputs=specs,
        )

    def infer(self, compiled: CompiledModel, feeds: dict[str, Any]) -> list[np.ndarray]:
        return compiled.session.run(None, feeds)
