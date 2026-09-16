"""Load an ONNX artifact through ONNX Runtime CPU and run honest timings.

This is runtime infrastructure (not the compiler). It never invents FPS:
throughput is 1000/mean_ms from samples actually measured on this process.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nnc.backends.ort_cpu import OrtCpuBackend


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    path: Path
    sha256: str
    bytes_len: int
    graph_opt: str
    compiled: Any
    input_name: str
    input_shape: tuple[int, ...]
    output_names: tuple[str, ...]
    providers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class InferSample:
    latency_ms: float
    output_shapes: tuple[tuple[int, ...], ...]


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _static_shape(meta: Any) -> tuple[int, ...]:
    shape: list[int] = []
    for dim in meta.shape:
        if isinstance(dim, int) and dim > 0:
            shape.append(dim)
        else:
            shape.append(1)
    return tuple(shape)


def load_ort_artifact(path: Path, *, graph_opt: str = "extended") -> LoadedArtifact:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"ONNX artifact not found: {path}")
    data = path.read_bytes()
    backend = OrtCpuBackend()
    ok, reason = backend.available()
    if not ok:
        raise RuntimeError(reason or "ort_cpu unavailable")
    compiled = backend.compile(data, graph_opt=graph_opt)
    inputs = compiled.session.get_inputs()
    if not inputs:
        raise RuntimeError(f"{path} has no graph inputs")
    first = inputs[0]
    return LoadedArtifact(
        path=path,
        sha256=_sha256(path),
        bytes_len=len(data),
        graph_opt=graph_opt,
        compiled=compiled,
        input_name=first.name,
        input_shape=_static_shape(first),
        output_names=tuple(compiled.output_names),
        providers=tuple(compiled.providers),
    )


def synthetic_feed(artifact: LoadedArtifact, rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    engine = rng or np.random.default_rng(0)
    return {artifact.input_name: engine.random(artifact.input_shape, dtype=np.float32)}


def infer_once(artifact: LoadedArtifact, feeds: dict[str, np.ndarray]) -> InferSample:
    backend = OrtCpuBackend()
    t0 = time.perf_counter()
    outputs = backend.infer(artifact.compiled, feeds)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    shapes = tuple(tuple(int(x) for x in arr.shape) for arr in outputs)
    return InferSample(latency_ms=latency_ms, output_shapes=shapes)


def benchmark_artifact(
    artifact: LoadedArtifact,
    *,
    warmup: int = 1,
    iters: int = 3,
    seed: int = 0,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    feeds = synthetic_feed(artifact, rng)
    for _ in range(max(0, warmup)):
        infer_once(artifact, feeds)
    samples: list[float] = []
    last_shapes: tuple[tuple[int, ...], ...] | None = None
    for _ in range(max(1, iters)):
        sample = infer_once(artifact, feeds)
        samples.append(sample.latency_ms)
        last_shapes = sample.output_shapes
    ordered = sorted(samples)
    mean = float(sum(ordered) / len(ordered))
    return {
        "model": {
            "path": str(artifact.path),
            "sha256": artifact.sha256,
            "bytes": artifact.bytes_len,
        },
        "backend": "ort_cpu",
        "graph_opt": artifact.graph_opt,
        "providers": list(artifact.providers),
        "input": {"name": artifact.input_name, "shape": list(artifact.input_shape)},
        "output_shapes": [list(s) for s in (last_shapes or ())],
        "warmup": warmup,
        "iters": iters,
        "latency_ms": {
            "mean": mean,
            "min": float(ordered[0]),
            "max": float(ordered[-1]),
            "p50": float(ordered[len(ordered) // 2]),
        },
        "throughput_ips": (1000.0 / mean) if mean > 0 else 0.0,
        "source": "synthetic",
    }
