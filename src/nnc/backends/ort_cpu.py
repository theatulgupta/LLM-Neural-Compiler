"""ONNX Runtime CPU backend supporting multi-dimensional execution strategies."""

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

_EXEC_MODES = {
    "sequential": ort.ExecutionMode.ORT_SEQUENTIAL,
    "parallel": ort.ExecutionMode.ORT_PARALLEL,
}


class OrtCpuBackend(Backend):
    name = "ort_cpu"

    def available(self) -> tuple[bool, str | None]:
        if "CPUExecutionProvider" not in ort.get_all_providers():
            return False, "onnxruntime was built without CPUExecutionProvider"
        return True, None

    def compile(
        self,
        model_bytes: bytes,
        *,
        graph_opt: str,
        execution_provider: str = "CPUExecutionProvider",
        execution_mode: str = "sequential",
        intra_op_num_threads: int = 0,
        inter_op_num_threads: int = 0,
        enable_mem_pattern: bool = True,
    ) -> CompiledModel:
        ok, reason = self.available()
        if not ok:
            raise RuntimeError(reason)
        if graph_opt not in _OPT_LEVELS:
            raise ValueError(f"unknown ORT graph_opt {graph_opt!r}; allowed={sorted(_OPT_LEVELS)}")
        if execution_mode not in _EXEC_MODES:
            raise ValueError(f"unknown execution_mode {execution_mode!r}; allowed={sorted(_EXEC_MODES)}")

        available_eps = set(ort.get_all_providers())
        if execution_provider == "XNNPACKExecutionProvider":
            if "XNNPACKExecutionProvider" not in available_eps:
                raise RuntimeError("onnxruntime was built without XNNPACKExecutionProvider")
            providers = ["XNNPACKExecutionProvider", "CPUExecutionProvider"]
        elif execution_provider == "CPUExecutionProvider":
            providers = ["CPUExecutionProvider"]
        else:
            raise ValueError(f"OrtCpuBackend does not support execution_provider={execution_provider!r}")

        options = ort.SessionOptions()
        options.graph_optimization_level = _OPT_LEVELS[graph_opt]
        options.execution_mode = _EXEC_MODES[execution_mode]
        options.enable_mem_pattern = bool(enable_mem_pattern)
        if intra_op_num_threads > 0:
            options.intra_op_num_threads = int(intra_op_num_threads)
        if inter_op_num_threads > 0:
            options.inter_op_num_threads = int(inter_op_num_threads)

        session = ort.InferenceSession(
            model_bytes,
            sess_options=options,
            providers=providers,
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
