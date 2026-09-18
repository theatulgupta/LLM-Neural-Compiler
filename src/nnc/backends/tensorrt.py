"""TensorRT backend. Compiles only when an NVIDIA GPU and TensorRT are present."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from nnc.backends.base import Backend, BackendOptions, CompiledModel
from nnc.backends.registry import register_backend


def nvidia_probe() -> tuple[bool, str]:
    """Return (present, evidence). Never invent a GPU."""

    smi = shutil.which("nvidia-smi")
    dev = Path("/dev/nvidia0")
    if smi is None and not dev.exists():
        return False, (
            "no NVIDIA device: nvidia-smi not on PATH and /dev/nvidia0 missing "
            "(this host is aarch64 QEMU with a Virtio GPU)"
        )
    return True, f"nvidia-smi={smi} nvidia0={dev.exists()}"


@register_backend
class TensorRtBackend(Backend):
    name = "tensorrt"

    def available(self) -> tuple[bool, str | None]:
        present, evidence = nvidia_probe()
        if not present:
            return False, evidence
        try:
            import tensorrt  # noqa: F401
        except ImportError:
            return False, f"NVIDIA GPU present ({evidence}) but the tensorrt Python package is not installed"
        return True, None

    def compile(
        self, model_bytes: bytes, *, options: BackendOptions | None = None, graph_opt: str | None = None
    ) -> CompiledModel:
        ok, reason = self.available()
        if not ok:
            raise RuntimeError(reason)
        raise RuntimeError("TensorRT compile path is not wired; GPU was present but no builder is configured")

    def infer(self, compiled: CompiledModel, feeds: dict[str, Any]) -> list[Any]:
        raise RuntimeError("TensorRT infer is unavailable on this host")
