"""Shared ONNX export helpers. Per-model scripts stay thin; skip reasons are JSON."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from compiler.catalog import REPO_ROOT, ModelSpec, get_model
from compiler.history import new_run_record, write_run_json
from compiler.utils.hashing import sha256_file
from nnc.probe import probe_host, probe_machine_id

DEFAULT_RESULTS = REPO_ROOT / "experiments" / "results"


def host_facts() -> dict[str, Any]:
    host = probe_host()
    return {
        "machine": host["platform"]["machine"],
        "uname_m": host["platform"]["machine"],
        "machine_id": probe_machine_id(),
        "python": host["platform"]["python"],
        "system": host["platform"]["system"],
        "nvidia": host["nvidia"],
    }


def skip_run_record(
    *,
    kind: str,
    path: str,
    reason: str,
    backend: str = "ort_cpu",
) -> dict[str, Any]:
    return new_run_record(
        run_id=str(uuid.uuid4()),
        model={"path": path, "kind": kind},
        backend=backend,
        strategy={
            "strategy": "baseline",
            "rationale": "skipped before compile",
            "source": "export",
        },
        host=host_facts(),
        compile={"ok": False, "ms": None, "error": None},
        benchmark=None,
        skip={"backend": backend, "reason": reason},
        fps_claimed=False,
    )


def write_skip(
    *,
    kind: str,
    path: str,
    reason: str,
    results_dir: Path | None = None,
    stem: str | None = None,
) -> Path:
    results = results_dir or DEFAULT_RESULTS
    record = skip_run_record(kind=kind, path=path, reason=reason)
    name = stem or f"skip_{kind.replace('/', '_')}"
    out = results / f"{name}.json"
    write_run_json(out, record)
    from compiler.history import append_history

    append_history(results / "history.jsonl", record)
    return out


def finish_onnx(src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        dest.write_bytes(src.read_bytes())
        src.unlink(missing_ok=True)
    return sha256_file(dest)


def cleanup_cwd_weights(*names: str) -> None:
    cwd = Path.cwd().resolve()
    for name in names:
        path = Path(name)
        if path.is_file() and path.parent.resolve() == cwd:
            path.unlink(missing_ok=True)


def export_note(spec: ModelSpec, dest: Path, digest: str) -> str:
    return (
        f"wrote {dest} bytes={dest.stat().st_size} sha256={digest}\n"
        f"next: python -m compiler compile {dest} --backend ort_cpu "
        f"--strategy {spec.default_strategy} --kind {spec.kind}"
    )


def load_spec(kind: str) -> ModelSpec:
    return get_model(kind)
