from __future__ import annotations

from compiler.pipeline import compile_and_benchmark
from nnc.backends.ort_cpu import OrtCpuBackend
from nnc.backends.tensorrt import TensorRtBackend, nvidia_probe


def test_ort_cpu_compiles_and_runs_tiny(tiny_path, tmp_path) -> None:
    record = compile_and_benchmark(
        tiny_path,
        backend_name="ort_cpu",
        strategy_name="baseline",
        warmup=2,
        iters=5,
        results_dir=tmp_path / "results",
        model_kind="fixture",
    )
    assert record["compile"]["ok"] is True
    assert record["skip"] is None
    assert record["benchmark"]["iters"] == 5
    assert record["benchmark"]["latency_ms"]["mean"] > 0
    history = (tmp_path / "results" / "history.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(history) == 1


def test_ort_cpu_rejects_gemm_k16(broken_k16_path, tmp_path) -> None:
    """Regression: Flatten 64 vs Gemm K=16 must not be reported as a successful compile."""

    record = compile_and_benchmark(
        broken_k16_path,
        backend_name="ort_cpu",
        strategy_name="baseline",
        warmup=0,
        iters=1,
        results_dir=tmp_path / "results",
        model_kind="fixture-broken-k16",
    )
    assert record["compile"]["ok"] is False
    assert record["compile"]["error"]
    err = record["compile"]["error"].lower()
    assert "16" in err or "64" in err or "gemm" in err or "shape" in err or "dimension" in err


def test_ort_session_fails_on_k16_bytes(broken_k16_path) -> None:
    import numpy as np

    backend = OrtCpuBackend()
    try:
        compiled = backend.compile(broken_k16_path.read_bytes(), graph_opt="disable")
        name = compiled.input_names[0]
        backend.infer(compiled, {name: np.zeros((1, 1, 8, 8), dtype=np.float32)})
    except Exception as exc:
        message = str(exc).lower()
        assert message
        return
    raise AssertionError("ORT compiled and executed Gemm K=16 against Flatten 64")


def test_tensorrt_skip_without_nvidia() -> None:
    present, reason = nvidia_probe()
    backend = TensorRtBackend()
    ok, skip = backend.available()
    if present:
        assert ok is True
        return
    assert ok is False
    assert skip is not None
    assert "nvidia" in skip.lower() or "tensorrt" in skip.lower()


def test_ort_cpu_backend_reports_available() -> None:
    ok, reason = OrtCpuBackend().available()
    assert ok is True
    assert reason is None
