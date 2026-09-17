"""Process RSS / CPU sampling around measured inference."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

from nnc.backends.base import Backend, CompiledModel


@dataclass(frozen=True, slots=True)
class ProfileResult:
    latency_ms: dict[str, float]
    throughput_ips: float
    rss_mb: dict[str, float]
    cpu_percent_mean: float
    threads: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rss_mb() -> float:
    try:
        import psutil

        return float(psutil.Process().memory_info().rss) / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return 0.0


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


def profile_session(
    backend: Backend,
    compiled: CompiledModel,
    feeds: dict[str, Any],
    *,
    warmup: int = 10,
    iters: int = 50,
) -> ProfileResult:
    import os

    try:
        import psutil

        proc = psutil.Process()
        proc.cpu_percent(None)
    except Exception:  # noqa: BLE001
        proc = None

    rss_before = _rss_mb()
    for _ in range(max(0, warmup)):
        backend.infer(compiled, feeds)

    samples: list[float] = []
    peak = rss_before
    cpu_samples: list[float] = []
    stop = threading.Event()

    def _sample() -> None:
        while not stop.wait(0.05):
            nonlocal peak
            rss = _rss_mb()
            if rss > peak:
                peak = rss
            if proc is not None:
                cpu_samples.append(float(proc.cpu_percent(None)))

    thread = threading.Thread(target=_sample, daemon=True)
    thread.start()
    try:
        for _ in range(max(1, iters)):
            t0 = time.perf_counter()
            backend.infer(compiled, feeds)
            samples.append((time.perf_counter() - t0) * 1000.0)
            rss = _rss_mb()
            if rss > peak:
                peak = rss
    finally:
        stop.set()
        thread.join(timeout=1.0)

    latency = _latency_stats(samples)
    mean = latency["mean"]
    cpu = float(sum(cpu_samples) / len(cpu_samples)) if cpu_samples else 0.0
    threads = int(os.cpu_count() or 1)
    try:
        import psutil

        threads = int(psutil.Process().num_threads())
    except Exception:  # noqa: BLE001
        pass
    return ProfileResult(
        latency_ms=latency,
        throughput_ips=(1000.0 / mean) if mean > 0 else 0.0,
        rss_mb={"before": rss_before, "after_compile": rss_before, "peak": peak},
        cpu_percent_mean=cpu,
        threads=threads,
    )
