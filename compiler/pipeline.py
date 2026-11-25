"""Compile + benchmark an ONNX model through an allowlisted strategy and backend."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnx

from compiler.evaluation.fidelity import compare_outputs
from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph
from compiler.history import append_history, new_run_record, write_run_json
from compiler.llm.recommendation_engine import load_recent_history, recommend_strategy
from compiler.optimization.passes import apply_strategy
from compiler.optimization.quantizer import quantize_model_bytes
from compiler.strategies import Strategy
from compiler.utils import utc_now_iso
from nnc.backends.base import Backend, BackendSkip, CompiledModel
from nnc.backends.ort_cpu import OrtCpuBackend
from nnc.backends.tensorrt import TensorRtBackend
from nnc.probe import probe_host

BackendName = Literal["ort_cpu", "tensorrt"]


def get_backend(name: BackendName) -> Backend:
    if name == "ort_cpu":
        return OrtCpuBackend()
    if name == "tensorrt":
        return TensorRtBackend()
    raise ValueError(f"unknown backend {name!r}")


def _input_feeds(compiled: CompiledModel, model: onnx.ModelProto, rng: np.random.Generator) -> dict[str, np.ndarray]:
    inits = {item.name for item in model.graph.initializer}
    feeds: dict[str, np.ndarray] = {}
    session_inputs = compiled.session.get_inputs()
    for item in session_inputs:
        if item.name in inits:
            continue
        shape = []
        for dim in item.shape:
            if isinstance(dim, int) and dim > 0:
                shape.append(dim)
            else:
                shape.append(1)
        dtype = np.float32 if item.type == "tensor(float)" else np.float32
        feeds[item.name] = rng.standard_normal(shape, dtype=dtype)
    return feeds


def _latency_stats(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    if not ordered:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
        return float(ordered[idx])

    mean = float(sum(ordered) / len(ordered))
    return {
        "mean": mean,
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _compile_with_strategy(backend: Backend, model_bytes: bytes, strategy: Strategy) -> CompiledModel:
    if isinstance(backend, OrtCpuBackend):
        return backend.compile(
            model_bytes,
            graph_opt=strategy.ort_graph_opt,
            execution_provider=strategy.execution_provider,
            execution_mode=strategy.execution_mode,
            intra_op_num_threads=strategy.intra_op_num_threads,
            inter_op_num_threads=strategy.inter_op_num_threads,
            enable_mem_pattern=strategy.enable_mem_pattern,
        )
    return backend.compile(model_bytes, graph_opt=strategy.ort_graph_opt)


def compile_and_benchmark(
    model_path: Path,
    *,
    backend_name: BackendName = "ort_cpu",
    strategy_name: str | None = None,
    strategy_obj: Strategy | None = None,
    client: Any = None,
    fallback_on_error: bool = False,
    check_fidelity: bool = True,
    warmup: int = 10,
    iters: int = 50,
    seed: int = 0,
    results_dir: Path | None = None,
    model_kind: str = "onnx",
) -> dict[str, Any]:
    try:
        loaded = load_graph(model_path, check=True)
    except Exception as exc:  # noqa: BLE001 — invalid graphs are compile failures
        loaded = load_graph(model_path, check=False)
        summary = summarize_graph(loaded)
        record = new_run_record(
            run_id=str(uuid.uuid4()),
            model={
                "path": str(model_path),
                "sha256": loaded.sha256,
                "kind": model_kind,
                "opset": loaded.opset,
            },
            graph=summary.to_dict(),
            backend=backend_name,
            strategy={"strategy": strategy_name or "baseline", "rationale": "load failed before recommend", "source": "pipeline"},
            compile={"ok": False, "ms": None, "error": f"{type(exc).__name__}: {exc}"},
            benchmark=None,
            skip=None,
        )
        _persist(record, results_dir)
        return record

    summary = summarize_graph(loaded)
    history_feedback = load_recent_history(results_dir, model_sha256=loaded.sha256)
    recommendation = recommend_strategy(
        summary,
        client=client,
        override=strategy_name,
        override_strategy=strategy_obj,
        history_feedback=history_feedback,
        fallback_on_error=fallback_on_error,
    )
    strategy = recommendation.strategy

    run_id = str(uuid.uuid4())
    host = probe_host()
    backend = get_backend(backend_name)
    available, skip_reason = backend.available()

    record: dict[str, Any] = new_run_record(
        run_id=run_id,
        model={
            "path": str(model_path),
            "sha256": loaded.sha256,
            "kind": model_kind,
            "opset": loaded.opset,
        },
        graph=summary.to_dict(),
        backend=backend_name,
        strategy=recommendation.to_dict(),
        host={
            "machine": host["platform"]["machine"],
            "python": host["platform"]["python"],
            "nvidia": host["nvidia"],
        },
    )

    if not available:
        skip = BackendSkip(backend=backend_name, reason=skip_reason or "unavailable")
        record["compile"] = {"ok": False, "ms": None, "error": None}
        record["benchmark"] = None
        record["skip"] = skip.to_dict()
        _persist(record, results_dir)
        return record

    compile_error: str | None = None
    compiled: CompiledModel | None = None
    optimized = loaded.model
    t0 = time.perf_counter()
    try:
        optimized = apply_strategy(loaded.model, strategy, verify_contracts=True)
        optimized_bytes = optimized.SerializeToString()
        prepared_bytes = quantize_model_bytes(optimized_bytes, strategy.precision)
        compiled = _compile_with_strategy(backend, prepared_bytes, strategy)
        trial_rng = np.random.default_rng(seed)
        trial_feeds = _input_feeds(compiled, optimized, trial_rng)
        trial_outputs = backend.infer(compiled, trial_feeds)
    except Exception as exc:  # noqa: BLE001 — persist the failure, do not fake success
        compile_error = f"{type(exc).__name__}: {exc}"
        compiled = None
        trial_outputs = []
        trial_feeds = {}
    compile_ms = (time.perf_counter() - t0) * 1000.0

    if compiled is None:
        record["compile"] = {"ok": False, "ms": compile_ms, "error": compile_error}
        record["benchmark"] = None
        record["skip"] = None
        _persist(record, results_dir)
        return record

    fidelity: dict[str, float] | None = None
    if check_fidelity and isinstance(backend, OrtCpuBackend):
        try:
            ref = backend.compile(loaded.bytes, graph_opt="disable")
            ref_outputs = backend.infer(ref, trial_feeds)
            fidelity = compare_outputs(ref_outputs, trial_outputs)
        except Exception:  # noqa: BLE001
            fidelity = None

    rng = np.random.default_rng(seed)
    feeds = _input_feeds(compiled, optimized, rng)
    for _ in range(max(0, warmup)):
        backend.infer(compiled, feeds)

    samples: list[float] = []
    for _ in range(max(1, iters)):
        t1 = time.perf_counter()
        backend.infer(compiled, feeds)
        samples.append((time.perf_counter() - t1) * 1000.0)

    latency = _latency_stats(samples)
    mean = latency["mean"]
    throughput = 1000.0 / mean if mean > 0 else 0.0
    record["compile"] = {
        "ok": True,
        "ms": compile_ms,
        "error": None,
        "providers": list(compiled.providers),
        "graph_opt": strategy.ort_graph_opt,
        "precision": strategy.precision,
    }
    record["benchmark"] = {
        "warmup": warmup,
        "iters": iters,
        "latency_ms": latency,
        "throughput_ips": throughput,
        "measured_at": utc_now_iso(),
    }
    if fidelity is not None:
        record["fidelity"] = fidelity
    record["skip"] = None
    _persist(record, results_dir)
    return record


def _persist(record: dict[str, Any], results_dir: Path | None) -> None:
    if results_dir is None:
        return
    run_path = results_dir / "runs" / f"{record['run_id']}.json"
    write_run_json(run_path, record)
    append_history(results_dir / "history.jsonl", record)


def analyze_only(model_path: Path) -> tuple[Any, Any]:
    loaded = load_graph(model_path)
    return loaded, summarize_graph(loaded)
