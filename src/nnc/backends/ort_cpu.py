"""ONNX Runtime CPU backend."""

from __future__ import annotations

from typing import Any

import numpy as np
import onnxruntime as ort

from nnc.backends.base import Backend, CompiledModel

_OPT_LEVELS = {
    "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


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
        return CompiledModel(
            backend=self.name,
            session=session,
            input_names=tuple(item.name for item in session.get_inputs()),
            output_names=tuple(item.name for item in session.get_outputs()),
            providers=tuple(session.get_providers()),
        )

    def infer(self, compiled: CompiledModel, feeds: dict[str, Any]) -> list[np.ndarray]:
        return compiled.session.run(None, feeds)
