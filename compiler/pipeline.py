"""Compile + benchmark an ONNX model through an allowlisted plan and backend."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from compiler.catalog import REPO_ROOT
from compiler.graph.graph_loader import LoadedGraph, load_graph
from compiler.graph.graph_summary import GraphSummary, summarize_graph
from compiler.hardware.profile import probe_hardware
from compiler.history import append_history, new_run_record, write_run_json
from compiler.llm.recommendation_engine import recommend_strategy
from compiler.optimization.optimizer import apply_plan, apply_strategy
from compiler.optimization.passes import graph_ir_snapshot
from compiler.planner.plan import Plan, get_plan
from compiler.planner.verifier import verify_plan
from compiler.profiling.profiler import profile_session
from compiler.utils.hashing import sha256_bytes
from compiler.utils.timeutil import utc_now_iso
from compiler.verification.gates import evaluate_gates
from compiler.verification.numerics import compare_outputs, task_agreement
from nnc.backends import BackendSkip, CompiledModel, get_backend
from nnc.backends.base import BackendOptions
from nnc.probe import probe_host, probe_machine_id

ARTIFACT_ROOT = REPO_ROOT / "experiments" / "artifacts"


def _numpy_dtype(name: str) -> np.dtype:
    if name in {"float16", "tensor(float16)"}:
        return np.float16
    return np.float32


def _input_feeds(compiled: CompiledModel, model: onnx.ModelProto, rng: np.random.Generator) -> dict[str, np.ndarray]:
    inits = {item.name for item in model.graph.initializer}
    feeds: dict[str, np.ndarray] = {}
    specs = compiled.inputs or tuple()
    if not specs:
        raise RuntimeError(f"{compiled.backend} compiled model is missing TensorSpec inputs")
    for spec in specs:
        if spec.name in inits:
            continue
        feeds[spec.name] = rng.standard_normal(spec.numpy_shape(), dtype=_numpy_dtype(spec.dtype))
    return feeds


def _host_block() -> dict[str, Any]:
    host = probe_host()
    hw = probe_hardware()
    return {
        "machine": host["platform"]["machine"],
        "uname_m": host.get("uname_m") or host["platform"]["machine"],
        "machine_id": host.get("machine_id") or probe_machine_id(),
        "python": host["platform"]["python"],
        "system": host["platform"]["system"],
        "nvidia": host["nvidia"],
        "hardware": hw.to_dict(),
    }


def _latency_stats(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    if not ordered:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
        return float(ordered[idx])

    mean = float(sum(ordered) / len(ordered))
    return {
        "mean": mean,
        "p50": pct(50),
        "p95": pct(95),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
    }


def _options_from_plan(plan: Plan) -> BackendOptions:
    return BackendOptions(
        graph_opt=plan.ort_graph_opt,
        intra_op_threads=plan.intra_op_threads,
        execution_mode=plan.execution_mode,
    )


def _save_artifact(kind: str, plan_id: str, model: onnx.ModelProto, plan: Plan) -> dict[str, Any]:
    folder = ARTIFACT_ROOT / kind
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{plan_id}.onnx"
    data = model.SerializeToString()
    path.write_bytes(data)
    sidecar = Path(str(path) + ".json")
    payload = {
        "plan": plan.to_dict(),
        "options": _options_from_plan(plan).to_dict(),
        "sha256": sha256_bytes(data),
        "path": str(path),
    }
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": str(path), "sha256": payload["sha256"], "sidecar": str(sidecar), "bytes": len(data)}


def _verification_feeds(
    compiled: CompiledModel,
    model: onnx.ModelProto,
    *,
    task: str,
    seed: int,
) -> tuple[list[dict[str, np.ndarray]], str]:
    specs = compiled.inputs
    if not specs:
        return [], "none"
    shape = specs[0].numpy_shape()
    name = specs[0].name
    try:
        from compiler.data.calibration import load_calibration_nchw

        tensors = load_calibration_nchw(source="gz_frames", input_shape=shape, limit=8)
        return [{name: t} for t in tensors], "calib"
    except Exception:  # noqa: BLE001
        rng = np.random.default_rng(seed)
        return [_input_feeds(compiled, model, rng)], "random"


def compile_verify_profile(
    model_path: Path,
    *,
    plan: Plan,
    backend_name: str = "ort_cpu",
    kind: str = "onnx",
    task: str = "detect",
    results_dir: Path | None = None,
    warmup: int = 10,
    iters: int = 50,
    seed: int = 0,
    verify_plan_first: bool = True,
) -> dict[str, Any]:
    hardware = probe_hardware()
    loaded = load_graph(model_path, check=False)
    summary = summarize_graph(loaded)
    verified = verify_plan(plan, summary, hardware, backend_name=backend_name) if verify_plan_first else None
    active = verified.plan if verified and verified.accepted else plan
    numerics_kind = verified.numerics if verified else "exact"

    before_ir = graph_ir_snapshot(loaded.model)
    try:
        optimized, steps_applied = apply_plan(loaded.model, active)
        apply_error = None
    except Exception as exc:  # noqa: BLE001
        optimized = loaded.model
        steps_applied = []
        apply_error = f"{type(exc).__name__}: {exc}"
    after_ir = graph_ir_snapshot(optimized)
    graph_changed = before_ir["node_count"] != after_ir["node_count"] or before_ir["op_counts"] != after_ir["op_counts"]
    runtime_bytes = optimized.SerializeToString()
    artifact_meta = _save_artifact(kind, active.plan_id.replace("/", "_"), optimized, active)

    backend = get_backend(backend_name)
    available, skip_reason = backend.available()
    run_id = str(uuid.uuid4())
    record: dict[str, Any] = new_run_record(
        schema_version=2,
        run_id=run_id,
        model={"path": str(model_path), "sha256": loaded.sha256, "kind": kind, "opset": loaded.opset},
        graph=summary.to_dict(),
        graph_before=before_ir,
        graph_after=after_ir,
        graph_runtime=after_ir,
        graph_changed=bool(graph_changed),
        passes_applied=list(active.passes),
        steps_applied=steps_applied,
        plan=active.to_dict(),
        backend=backend_name,
        strategy={"strategy": active.name, "rationale": active.rationale, "source": active.source},
        host=_host_block(),
        hardware=hardware.to_dict(),
        artifact=artifact_meta,
        fps_claimed=False,
    )
    if apply_error:
        record["compile"] = {"ok": False, "ms": None, "error": apply_error}
        record["benchmark"] = None
        record["skip"] = None
        record["verification"] = {
            "accepted_plan": bool(verified.accepted) if verified else False,
            "rejections": list(verified.rejections) if verified else [],
            "numerics": numerics_kind,
            "passed": False,
            "error": apply_error,
        }
        _persist(record, results_dir)
        return record
    if not available:
        skip = BackendSkip(backend=backend_name, reason=skip_reason or "unavailable")
        record["compile"] = {"ok": False, "ms": None, "error": None}
        record["benchmark"] = None
        record["skip"] = skip.to_dict()
        _persist(record, results_dir)
        return record

    options = _options_from_plan(active)
    native_backend = get_backend(backend_name)
    compile_error = None
    compiled: CompiledModel | None = None
    native_compiled: CompiledModel | None = None
    t0 = time.perf_counter()
    try:
        compiled = backend.compile(runtime_bytes, options=options)
        native_compiled = native_backend.compile(loaded.model.SerializeToString(), options=BackendOptions(graph_opt="disable"))
        trial_feeds = _input_feeds(compiled, optimized, np.random.default_rng(seed))
        backend.infer(compiled, trial_feeds)
    except Exception as exc:  # noqa: BLE001
        compile_error = f"{type(exc).__name__}: {exc}"
        compiled = None
    compile_ms = (time.perf_counter() - t0) * 1000.0
    if compiled is None or native_compiled is None:
        record["compile"] = {"ok": False, "ms": compile_ms, "error": compile_error}
        record["benchmark"] = None
        record["skip"] = None
        record["verification"] = {
            "accepted_plan": bool(verified.accepted) if verified else True,
            "rejections": list(verified.rejections) if verified else [],
            "numerics": numerics_kind,
            "passed": False,
            "error": compile_error,
        }
        _persist(record, results_dir)
        return record

    feeds_list, inputs_source = _verification_feeds(compiled, optimized, task=task, seed=seed)
    compares: list[dict[str, Any]] = []
    agreements: list[dict[str, Any]] = []
    ref_scale = 1.0
    for feeds in feeds_list[:8]:
        ref = native_backend.infer(native_compiled, feeds)
        cand = backend.infer(compiled, feeds)
        compares.append(compare_outputs(ref, cand))
        agreements.append(task_agreement(ref, cand, task))
        ref_scale = max(ref_scale, float(np.max(np.abs(ref[0]))) if ref else 1.0)
    compare = compares[0] if compares else {"max_abs": 0.0, "cosine_min": 1.0, "per_output": []}
    if len(compares) > 1:
        compare = {
            "per_output": compares[0].get("per_output"),
            "max_abs": max(c["max_abs"] for c in compares),
            "cosine_min": min(c["cosine_min"] for c in compares),
        }
    agreement: dict[str, Any] = {}
    if agreements:
        if "matched_ratio" in agreements[0]:
            agreement = {
                "matched_ratio": float(sum(a.get("matched_ratio", 0) for a in agreements) / len(agreements)),
                "mean_iou": float(sum(a.get("mean_iou", 0) for a in agreements) / len(agreements)),
            }
        elif "top1_agreement" in agreements[0]:
            agreement = {
                "top1_agreement": float(sum(a.get("top1_agreement", 0) for a in agreements) / len(agreements)),
            }
        else:
            agreement = agreements[0]
    gate = evaluate_gates(
        numerics_kind=numerics_kind,
        compare=compare,
        agreement=agreement,
        task=task,
        ref_scale=ref_scale,
    )
    profile_feeds = feeds_list[0] if feeds_list else _input_feeds(compiled, optimized, np.random.default_rng(seed))
    rss_before = 0.0
    try:
        import psutil

        rss_before = float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:  # noqa: BLE001
        pass
    profile = profile_session(backend, compiled, profile_feeds, warmup=warmup, iters=iters)
    rss = dict(profile.rss_mb)
    rss["after_compile"] = rss.get("peak", rss_before)
    rss["before"] = rss_before or rss.get("before", 0.0)
    record["compile"] = {
        "ok": True,
        "ms": compile_ms,
        "error": None,
        "providers": list(compiled.providers),
        "graph_opt": active.ort_graph_opt,
        "runtime_inlined": False,
        "options": options.to_dict(),
    }
    record["benchmark"] = {
        "warmup": warmup,
        "iters": iters,
        "latency_ms": profile.latency_ms,
        "throughput_ips": profile.throughput_ips,
        "measured_at": utc_now_iso(),
        "fps_claimed": False,
    }
    record["profile"] = {
        "rss_mb": rss,
        "cpu_percent_mean": profile.cpu_percent_mean,
        "model_bytes": artifact_meta["bytes"],
        "compile_ms": compile_ms,
        "threads": profile.threads,
    }
    record["verification"] = {
        "accepted_plan": bool(verified.accepted) if verified else True,
        "rejections": list(verified.rejections) if verified else [],
        "numerics": compare,
        "agreement": agreement,
        "passed": bool(gate["passed"]),
        "inputs_source": inputs_source,
        "gate_used": gate["gate_used"],
        "numerics_kind": numerics_kind,
    }
    record["skip"] = None
    _persist(record, results_dir)
    return record


def compile_and_benchmark(
    model_path: Path,
    *,
    backend_name: str = "ort_cpu",
    strategy_name: str | None = None,
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
            host=_host_block(),
            fps_claimed=False,
        )
        _persist(record, results_dir)
        return record
    summary = summarize_graph(loaded)
    recommendation = recommend_strategy(summary, override=strategy_name)
    strategy = recommendation.strategy

    before_ir = graph_ir_snapshot(loaded.model)
    optimized = apply_strategy(loaded.model, strategy)
    after_ir = graph_ir_snapshot(optimized)
    runtime = optimized
    runtime_ir = graph_ir_snapshot(runtime)
    graph_changed = before_ir["node_count"] != after_ir["node_count"] or before_ir["op_counts"] != after_ir["op_counts"]
    runtime_bytes = runtime.SerializeToString()

    backend = get_backend(backend_name)
    available, skip_reason = backend.available()
    run_id = str(uuid.uuid4())

    record: dict[str, Any] = new_run_record(
        run_id=run_id,
        model={
            "path": str(model_path),
            "sha256": loaded.sha256,
            "kind": model_kind,
            "opset": loaded.opset,
        },
        graph=summary.to_dict(),
        graph_before=before_ir,
        graph_after=after_ir,
        graph_runtime=runtime_ir,
        graph_changed=bool(graph_changed),
        passes_applied=list(strategy.passes),
        backend=backend_name,
        strategy=recommendation.to_dict(),
        host=_host_block(),
        fps_claimed=False,
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
    t0 = time.perf_counter()
    try:
        compiled = backend.compile(runtime_bytes, graph_opt=strategy.ort_graph_opt)
        trial_rng = np.random.default_rng(seed)
        trial_feeds = _input_feeds(compiled, runtime, trial_rng)
        backend.infer(compiled, trial_feeds)
    except Exception as exc:  # noqa: BLE001
        compile_error = f"{type(exc).__name__}: {exc}"
        compiled = None
    compile_ms = (time.perf_counter() - t0) * 1000.0

    if compiled is None:
        record["compile"] = {"ok": False, "ms": compile_ms, "error": compile_error}
        record["benchmark"] = None
        record["skip"] = None
        _persist(record, results_dir)
        return record

    rng = np.random.default_rng(seed)
    feeds = _input_feeds(compiled, runtime, rng)
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
        "runtime_inlined": after_ir.get("op_counts") != runtime_ir.get("op_counts"),
    }
    record["benchmark"] = {
        "warmup": warmup,
        "iters": iters,
        "latency_ms": latency,
        "throughput_ips": throughput,
        "measured_at": utc_now_iso(),
        "fps_claimed": False,
    }
    record["skip"] = None
    _persist(record, results_dir)
    return record


def _persist(record: dict[str, Any], results_dir: Path | None) -> None:
    if results_dir is None:
        return
    run_path = results_dir / "runs" / f"{record['run_id']}.json"
    write_run_json(run_path, record)
    append_history(results_dir / "history.jsonl", record)


def analyze_only(model_path: Path) -> tuple[LoadedGraph, GraphSummary]:
    loaded = load_graph(model_path)
    return loaded, summarize_graph(loaded)
