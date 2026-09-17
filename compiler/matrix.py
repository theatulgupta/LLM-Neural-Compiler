"""Native (ORT baseline) vs allowlisted-strategy pairs. Numbers come from compile_and_benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from compiler.catalog import ModelSpec, REPO_ROOT, load_zoo
from compiler.exporters import skip_run_record, write_skip
from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph
from compiler.history import write_run_json
from compiler.llm.recommendation_engine import recommend_strategy
from compiler.pipeline import compile_and_benchmark
from nnc.probe import probe_host

PAPER_MATRIX_NAME = "paper_matrix.json"


def _slice(record: dict[str, Any]) -> dict[str, Any]:
    bench = record.get("benchmark") or {}
    latency = bench.get("latency_ms") or {}
    compile_block = record.get("compile") or {}
    return {
        "run_id": record.get("run_id"),
        "strategy": (record.get("strategy") or {}).get("strategy"),
        "strategy_source": (record.get("strategy") or {}).get("source"),
        "compile_ok": compile_block.get("ok"),
        "compile_ms": compile_block.get("ms"),
        "compile_error": compile_block.get("error"),
        "skip": record.get("skip"),
        "p50_ms": latency.get("p50"),
        "mean_ms": latency.get("mean"),
        "p95_ms": latency.get("p95"),
        "throughput_ips": bench.get("throughput_ips"),
        "warmup": bench.get("warmup"),
        "iters": bench.get("iters"),
        "fps_claimed": False,
        "graph_changed": record.get("graph_changed"),
        "graph_before_nodes": (record.get("graph_before") or {}).get("node_count"),
        "graph_after_nodes": (record.get("graph_after") or {}).get("node_count"),
        "passes_applied": record.get("passes_applied"),
    }


def _speedup(native_p50: object, opt_p50: object) -> float | None:
    if not isinstance(native_p50, (int, float)) or not isinstance(opt_p50, (int, float)):
        return None
    if opt_p50 <= 0:
        return None
    return float(native_p50) / float(opt_p50)


def measure_path(
    model_path: Path,
    *,
    kind: str,
    task: str | None = None,
    native_strategy: str = "baseline",
    warmup: int = 3,
    iters: int = 8,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    loaded = load_graph(model_path)
    summary = summarize_graph(loaded)
    recommendation = recommend_strategy(summary)
    native = compile_and_benchmark(
        model_path,
        backend_name="ort_cpu",
        strategy_name=native_strategy,
        warmup=warmup,
        iters=iters,
        results_dir=results_dir,
        model_kind=kind,
    )
    optimized = compile_and_benchmark(
        model_path,
        backend_name="ort_cpu",
        strategy_name=recommendation.strategy.name,
        warmup=warmup,
        iters=iters,
        results_dir=results_dir,
        model_kind=kind,
    )
    native_slice = _slice(native)
    opt_slice = _slice(optimized)
    native_p50 = native_slice.get("p50_ms")
    opt_p50 = opt_slice.get("p50_ms")
    compile_ms = opt_slice.get("compile_ms")
    pair = {
        "kind": kind,
        "task": task,
        "path": str(model_path),
        "sha256": loaded.sha256,
        "opset": loaded.opset,
        "recommend": recommendation.to_dict(),
        "native": native_slice,
        "optimized": opt_slice,
        "p50_speedup_native_over_optimized": _speedup(native_p50, opt_p50),
        "compile_ms_optimized": compile_ms,
        "fps_claimed": False,
        "same_host": True,
        "note": (
            "aarch64 QEMU ORT CPU only. Not TensorRT. Not cloud x86. "
            "throughput_ips is 1000/mean_ms from measured samples."
        ),
    }
    if results_dir is not None:
        out = results_dir / f"matrix_{kind.replace('/', '_')}.json"
        write_run_json(
            out,
            {
                "schema_version": 1,
                "run_id": f"matrix-{kind}",
                "created_at": native.get("created_at") or optimized.get("created_at"),
                "model": {"path": str(model_path), "kind": kind, "sha256": loaded.sha256},
                "backend": "ort_cpu",
                "compile": optimized.get("compile") or {"ok": False},
                "benchmark": optimized.get("benchmark"),
                "skip": optimized.get("skip"),
                "host": native.get("host") or optimized.get("host"),
                "fps_claimed": False,
                "pair": pair,
            },
            validate=False,
        )
        # Keep a schema-valid copy of each compile via compile_and_benchmark; pair JSON is extra.
    return pair


def measure_spec(
    spec: ModelSpec,
    *,
    warmup: int = 3,
    iters: int = 8,
    results_dir: Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    onnx_path = spec.onnx_path(root or REPO_ROOT)
    if not onnx_path.is_file():
        reason = (
            f"ONNX not present at {onnx_path}; run {' '.join(spec.export_cmd())}"
        )
        if results_dir is not None:
            write_skip(kind=spec.kind, path=str(onnx_path), reason=reason, stem=f"skip_{spec.kind}")
        record = skip_run_record(kind=spec.kind, path=str(onnx_path), reason=reason)
        return {
            "kind": spec.kind,
            "task": spec.task,
            "path": str(onnx_path),
            "sha256": None,
            "skip": record["skip"],
            "fps_claimed": False,
            "native": None,
            "optimized": None,
            "uav_role": spec.uav_role,
        }
    pair = measure_path(
        onnx_path,
        kind=spec.kind,
        task=spec.task,
        native_strategy=spec.native_strategy,
        warmup=warmup,
        iters=iters,
        results_dir=results_dir,
    )
    pair["uav_role"] = spec.uav_role
    pair["family"] = spec.family
    return pair


def run_zoo_matrix(
    *,
    kinds: list[str] | None = None,
    warmup: int = 3,
    iters: int = 8,
    results_dir: Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    host = probe_host()
    wanted = set(kinds) if kinds else None
    rows = []
    for spec in load_zoo():
        if wanted is not None and spec.kind not in wanted:
            continue
        rows.append(
            measure_spec(spec, warmup=warmup, iters=iters, results_dir=results_dir, root=root)
        )
    summary = {
        "host": {
            "machine": host["platform"]["machine"],
            "uname_m": host.get("uname_m"),
            "machine_id": host.get("machine_id"),
            "python": host["platform"]["python"],
            "nvidia": host["nvidia"],
        },
        "fps_claimed": False,
        "backend": "ort_cpu",
        "warmup": warmup,
        "iters": iters,
        "note": (
            "Native = unrewritten ONNX (strategy baseline, ORT graph opt disable). "
            "Optimized = schema-bound advisor pass set on the same host. "
            "graph_changed is node_count/op_counts on the compiler IR. "
            "Do not compare these numbers to cloud x86 or TensorRT."
        ),
        "models": rows,
    }
    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        out = results_dir / PAPER_MATRIX_NAME
        out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        summary["wrote"] = str(out)
    return summary
