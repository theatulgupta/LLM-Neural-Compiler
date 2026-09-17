"""UAV companion model zoo. Pipeline never switches on YOLOv8-only names.

Records live in experiments/zoo.yaml. Adding a model means a new YAML entry,
an export script (or family exporter args), and the same compile CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from compiler.strategies import ALLOWED_STRATEGY_NAMES, get_strategy

REPO_ROOT = Path(__file__).resolve().parents[1]
ZOO_PATH = REPO_ROOT / "experiments" / "zoo.yaml"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    kind: str
    task: str
    family: str
    weights: str
    onnx: str
    export_script: str
    export_args: tuple[str, ...]
    default_strategy: str
    native_strategy: str
    imgsz: int
    opset: int
    uav_role: str
    justification: str

    def onnx_path(self, root: Path | None = None) -> Path:
        return (root or REPO_ROOT) / self.onnx

    def export_cmd(self, *, python: str = "python") -> list[str]:
        argv = [python, str(REPO_ROOT / self.export_script)]
        argv.extend(self.export_args)
        argv.extend(["--out", str(self.onnx_path())])
        return argv


def _scalar(raw: str) -> str | int:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    if text.isdigit():
        return int(text)
    return text


def _parse_zoo_yaml(text: str) -> list[dict[str, str | int]]:
    """Tiny YAML subset: a `models:` list of scalar maps. No nested objects."""

    models: list[dict[str, str | int]] = []
    current: dict[str, str | int] | None = None
    for raw in text.splitlines():
        if (not raw.strip()) or raw.lstrip().startswith("#"):
            continue
        if raw.startswith("models:"):
            continue
        if raw.startswith("  - "):
            if current:
                models.append(current)
            current = {}
            key, _, value = raw[4:].partition(":")
            current[key.strip()] = _scalar(value)
            continue
        if raw.startswith("    ") and current is not None:
            key, _, value = raw.strip().partition(":")
            current[key.strip()] = _scalar(value)
            continue
        raise ValueError(f"unsupported zoo.yaml line: {raw!r}")
    if current:
        models.append(current)
    return models


def _args(value: str | int) -> tuple[str, ...]:
    text = str(value).strip()
    if not text:
        return ()
    return tuple(text.split())


def load_zoo(path: Path | None = None) -> tuple[ModelSpec, ...]:
    zoo_path = path or ZOO_PATH
    rows = _parse_zoo_yaml(zoo_path.read_text(encoding="utf-8"))
    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for row in rows:
        kind = str(row["kind"])
        if kind in seen:
            raise ValueError(f"duplicate zoo kind {kind!r}")
        seen.add(kind)
        default_strategy = str(row["default_strategy"])
        native_strategy = str(row.get("native_strategy", "baseline"))
        get_strategy(default_strategy)
        get_strategy(native_strategy)
        if default_strategy not in ALLOWED_STRATEGY_NAMES:
            raise ValueError(f"{kind} default_strategy not allowlisted")
        specs.append(
            ModelSpec(
                kind=kind,
                task=str(row["task"]),
                family=str(row["family"]),
                weights=str(row["weights"]),
                onnx=str(row["onnx"]),
                export_script=str(row["export_script"]),
                export_args=_args(row.get("export_args", "")),
                default_strategy=default_strategy,
                native_strategy=native_strategy,
                imgsz=int(row["imgsz"]),
                opset=int(row["opset"]),
                uav_role=str(row["uav_role"]),
                justification=str(row["justification"]),
            )
        )
    return tuple(specs)


def get_model(kind: str, path: Path | None = None) -> ModelSpec:
    for spec in load_zoo(path):
        if spec.kind == kind:
            return spec
    known = [spec.kind for spec in load_zoo(path)]
    raise KeyError(f"unknown zoo kind {kind!r}; known={known}")


def zoo_kinds(path: Path | None = None) -> tuple[str, ...]:
    return tuple(spec.kind for spec in load_zoo(path))


def zoo_tasks(path: Path | None = None) -> tuple[str, ...]:
    return tuple(spec.task for spec in load_zoo(path))
