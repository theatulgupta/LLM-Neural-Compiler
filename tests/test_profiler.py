from __future__ import annotations

import numpy as np

from compiler.profiling.profiler import profile_session
from nnc.backends.ort_cpu import OrtCpuBackend


def test_profiler_rss(tiny_path) -> None:
    backend = OrtCpuBackend()
    compiled = backend.compile(tiny_path.read_bytes(), graph_opt="disable")
    feeds = {compiled.input_names[0]: np.zeros((1, 1, 8, 8), dtype=np.float32)}
    result = profile_session(backend, compiled, feeds, warmup=1, iters=3)
    assert result.rss_mb["peak"] >= result.rss_mb["before"]
    assert result.latency_ms["mean"] > 0
