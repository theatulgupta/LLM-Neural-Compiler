from __future__ import annotations

from nnc.backends.base import BackendOptions
from nnc.backends.ort_cpu import OrtCpuBackend
import numpy as np
import os


def test_backend_thread_options(tiny_path) -> None:
    backend = OrtCpuBackend()
    data = tiny_path.read_bytes()
    one = backend.compile(data, options=BackendOptions(graph_opt="disable", intra_op_threads=1))
    many = backend.compile(
        data,
        options=BackendOptions(graph_opt="disable", intra_op_threads=os.cpu_count() or 2),
    )
    x = np.zeros((1, 1, 8, 8), dtype=np.float32)
    a = backend.infer(one, {one.input_names[0]: x})[0]
    b = backend.infer(many, {many.input_names[0]: x})[0]
    assert a.shape == b.shape
