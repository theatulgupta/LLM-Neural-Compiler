"""Host hardware facts the planner is allowed to see. Never invent a GPU."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from nnc.backends.tensorrt import nvidia_probe


def _cpuinfo() -> tuple[str, tuple[str, ...]]:
    model = "unknown"
    features: tuple[str, ...] = ()
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return model, features
    wanted = {"asimd", "asimddp", "fphp", "bf16", "fp16", "sve", "i8mm"}
    found: set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("model name") or line.startswith("CPU part") or line.startswith("Hardware"):
            _, _, value = line.partition(":")
            if value.strip():
                model = value.strip()
        if line.startswith("Features") or line.lower().startswith("flags"):
            _, _, value = line.partition(":")
            for token in value.split():
                if token in wanted:
                    found.add(token)
    return model, tuple(sorted(found))


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    arch: str
    cpu_count: int
    cpu_model: str
    features: tuple[str, ...]
    ram_total_mb: int
    ram_available_mb: int
    os: str
    python: str
    ort_version: str
    providers: tuple[str, ...]
    gpu_present: bool
    gpu_detail: str
    virtualized: bool
    fp16_execution: bool = field(default=False)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def probe_hardware() -> HardwareProfile:
    import os
    import platform
    import sys

    try:
        import psutil

        ram_total = int(psutil.virtual_memory().total / (1024 * 1024))
        ram_avail = int(psutil.virtual_memory().available / (1024 * 1024))
        cpu_count = int(psutil.cpu_count(logical=True) or os.cpu_count() or 1)
    except Exception:  # noqa: BLE001
        ram_total, ram_avail = 0, 0
        cpu_count = int(os.cpu_count() or 1)

    gpu_ok, gpu_reason = nvidia_probe()
    cpu_model, features = _cpuinfo()
    providers: tuple[str, ...] = ()
    ort_version = "missing"
    try:
        import onnxruntime as ort

        ort_version = str(ort.__version__)
        providers = tuple(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        pass

    virt = bool(
        Path("/sys/class/dmi/id/product_name").read_text(encoding="utf-8", errors="replace").lower().find("qemu") >= 0
        if Path("/sys/class/dmi/id/product_name").is_file()
        else "qemu" in platform.release().lower() or platform.machine() == "aarch64"
    )
    fp16 = gpu_ok or "CUDAExecutionProvider" in providers or "TensorrtExecutionProvider" in providers
    return HardwareProfile(
        arch=platform.machine(),
        cpu_count=cpu_count,
        cpu_model=cpu_model,
        features=features,
        ram_total_mb=ram_total,
        ram_available_mb=ram_avail,
        os=f"{platform.system()} {platform.release()}",
        python=sys.version.split()[0],
        ort_version=ort_version,
        providers=providers,
        gpu_present=gpu_ok,
        gpu_detail=gpu_reason,
        virtualized=virt,
        fp16_execution=fp16,
    )
