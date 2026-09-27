"""Dashboard facts from experiments/results. No invented latency."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import Any

from compiler.catalog import REPO_ROOT, load_zoo
from compiler.history.store import read_history

_SECRET_MARKERS = ("gsk_", "sk-ant-", "sk-or-", "sk-proj-", "AIzaSy", "nvapi-")
UPLOADS = REPO_ROOT / "experiments" / "uploads"


def scrub(value: Any) -> Any:
    """Drop strings that look like API keys before they reach the page."""

    if isinstance(value, str):
        if any(marker in value for marker in _SECRET_MARKERS):
            return "[redacted]"
        return value
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, dict):
        return {str(key): scrub(item) for key, item in value.items()}
    return value


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _clip(text: object, limit: int = 400) -> str:
    raw = "" if text is None else str(text)
    if len(raw) <= limit:
        return raw
    return raw[: limit - 3] + "..."


def _speedup(baseline: object, chosen: object) -> float | None:
    try:
        base = float(baseline)  # type: ignore[arg-type]
        pick = float(chosen)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if pick <= 0:
        return None
    return base / pick


def host_strip() -> dict[str, Any]:
    """Machine and GPU facts. Same probes the CLI uses for NVIDIA and TensorRT."""

    from nnc.backends.tensorrt import TensorRtBackend, nvidia_probe

    present, detail = nvidia_probe()
    ready, reason = TensorRtBackend().available()
    machine = platform.machine()
    return {
        "machine": machine,
        "system": platform.system(),
        "python": platform.python_version(),
        "nvidia_present": present,
        "nvidia_detail": detail,
        "tensorrt_available": ready,
        "tensorrt_reason": None if ready else reason,
        "qemu_note": detail if "QEMU" in detail else None,
        "fps_claimed": False,
    }


def default_models() -> list[dict[str, str]]:
    rows = [
        {
            "kind": "fixture",
            "task": "classify",
            "path": str(REPO_ROOT / "fixtures" / "tiny_cnn.onnx"),
            "uav_role": "tiny CNN fixture; proves Conv-ReLU fusion",
            "exportable": "0",
        },
        {
            "kind": "tiny_depth",
            "task": "depth",
            "path": str(REPO_ROOT / "fixtures" / "tiny_depth.onnx"),
            "uav_role": "tiny depth fixture",
            "exportable": "0",
        },
    ]
    for spec in load_zoo():
        rows.append(
            {
                "kind": spec.kind,
                "task": spec.task,
                "path": str(spec.onnx_path()),
                "uav_role": spec.uav_role,
                "exportable": "1",
            }
        )
    return rows


def upload_models(uploads: Path) -> list[dict[str, str]]:
    if not uploads.is_dir():
        return []
    rows: list[dict[str, str]] = []
    for path in sorted(uploads.glob("*.onnx")):
        rows.append(
            {
                "kind": f"upload_{path.stem}",
                "task": "onnx",
                "path": str(path),
                "uav_role": "uploaded ONNX",
                "exportable": "0",
            }
        )
    return rows


def save_upload(uploads: Path, name: str, data: bytes) -> Path:
    """Store one ONNX under the uploads directory. Reject other names."""

    safe = Path(name).name
    if safe != name or not safe.endswith(".onnx") or safe.startswith("."):
        raise ValueError("upload must be a single .onnx file name")
    if not data:
        raise ValueError("empty upload")
    uploads.mkdir(parents=True, exist_ok=True)
    dest = uploads / safe
    dest.write_bytes(data)
    return dest


def _artifact_path(row: dict[str, Any]) -> str | None:
    artifact = row.get("artifact")
    if isinstance(artifact, dict) and artifact.get("path"):
        return str(artifact["path"])
    return None


def _candidate_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in payload.get("candidates") or []:
        if not isinstance(item, dict):
            continue
        passed = item.get("passed")
        p50 = item.get("p50_ms")
        rows.append(
            {
                "origin": item.get("origin"),
                "plan_id": item.get("plan_id"),
                "p50_ms": p50,
                "passed": passed,
                "nodes_before": item.get("nodes_before"),
                "nodes_after": item.get("nodes_after"),
                "graph_changed": item.get("graph_changed"),
                "same_as": item.get("same_as"),
                "artifact": _artifact_path(item),
                "illegal": p50 is not None and not passed,
            }
        )
    return rows


def _attempts(llm: dict[str, Any]) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for item in llm.get("attempts") or []:
        if not isinstance(item, dict):
            continue
        corrections = []
        for corr in item.get("corrections") or []:
            if isinstance(corr, dict):
                corrections.append(
                    {
                        "step": corr.get("step"),
                        "reason": _clip(corr.get("reason"), 180),
                        "level": corr.get("level"),
                    }
                )
        attempts.append(
            {
                "n": item.get("n"),
                "outcome": item.get("outcome"),
                "detail": _clip(item.get("detail")),
                "corrections": corrections,
            }
        )
    return attempts


def _followup(block: object) -> dict[str, Any] | None:
    if not isinstance(block, dict):
        return None
    if block.get("skipped"):
        return {"skipped": _clip(block.get("skipped")), "same_as": block.get("same_as")}
    if block.get("plan_id"):
        return {
            "plan_id": block.get("plan_id"),
            "p50_ms": block.get("p50_ms"),
            "passed": block.get("passed"),
            "rank": block.get("rank"),
        }
    return None


def _status(path: Path, chosen: dict[str, Any]) -> str:
    if not path.is_file():
        return "missing"
    if not chosen:
        return "unmeasured"
    if chosen.get("passed"):
        return "passed"
    return "failed"


def _baseline_p50(payload: dict[str, Any]) -> object:
    for cand in payload.get("candidates") or []:
        if not isinstance(cand, dict):
            continue
        origin = str(cand.get("origin") or "")
        if origin.endswith("baseline") or cand.get("plan_id") == "baseline":
            return cand.get("p50_ms")
    native = payload.get("native")
    if isinstance(native, dict):
        return native.get("p50_ms")
    return None


def _model_card(row: dict[str, str], results: Path) -> dict[str, Any]:
    kind = row["kind"]
    path = Path(row["path"])
    payload = _read_json(results / f"optimize_{kind.replace('/', '_')}.json") or {}
    chosen = payload.get("chosen") if isinstance(payload.get("chosen"), dict) else {}
    llm = payload.get("llm") if isinstance(payload.get("llm"), dict) else {}
    baseline = _baseline_p50(payload)
    host = payload.get("host") if isinstance(payload.get("host"), dict) else None
    return {
        "kind": kind,
        "task": row.get("task") or payload.get("task"),
        "uav_role": row.get("uav_role") or "",
        "exportable": row.get("exportable") == "1",
        "path": str(path),
        "onnx_present": path.is_file(),
        "status": _status(path, chosen),
        "baseline_p50_ms": baseline,
        "speedup": _speedup(baseline, chosen.get("p50_ms") if chosen else None),
        "host": host,
        "llm_rank": payload.get("llm_rank"),
        "chosen": {
            "origin": chosen.get("origin"),
            "plan_id": chosen.get("plan_id"),
            "p50_ms": chosen.get("p50_ms"),
            "passed": chosen.get("passed"),
            "artifact": _artifact_path(chosen),
        }
        if chosen
        else None,
        "candidates": _candidate_rows(payload),
        "llm": {
            "source": llm.get("source"),
            "fallback_reason": _clip(llm.get("fallback_reason")),
            "attempts": _attempts(llm),
            "revised": _followup(llm.get("revised")),
            "improved": _followup(llm.get("improved")),
            "same_as": llm.get("same_as"),
        }
        if llm
        else None,
        "fps_claimed": False,
    }


def _report_from_optimize(payload: dict[str, Any]) -> dict[str, Any]:
    chosen = payload.get("chosen") if isinstance(payload.get("chosen"), dict) else {}
    llm = payload.get("llm") if isinstance(payload.get("llm"), dict) else {}
    baseline = _baseline_p50(payload)
    follow = _followup(llm.get("improved")) or _followup(llm.get("revised"))
    follow_text = None
    if isinstance(follow, dict):
        if follow.get("skipped"):
            follow_text = f"skipped: {follow['skipped']}"
        elif follow.get("plan_id"):
            follow_text = f"{follow.get('plan_id')} p50={follow.get('p50_ms')}"
    host = payload.get("host") if isinstance(payload.get("host"), dict) else None
    return {
        "kind": payload.get("kind"),
        "task": payload.get("task"),
        "baseline_p50_ms": baseline,
        "chosen_plan": chosen.get("plan_id"),
        "chosen_origin": chosen.get("origin"),
        "chosen_p50_ms": chosen.get("p50_ms"),
        "speedup": _speedup(baseline, chosen.get("p50_ms")),
        "passed": chosen.get("passed"),
        "llm_source": llm.get("source"),
        "llm_rank": payload.get("llm_rank"),
        "followup": follow_text,
        "host": host,
        "fps_claimed": False,
    }


def report_rows(results: Path) -> dict[str, Any]:
    """Comparison rows from optimize JSON, filled in with paper_matrix when present."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    if results.is_dir():
        for path in sorted(results.glob("optimize_*.json")):
            payload = _read_json(path) or {}
            if "chosen" not in payload and "candidates" not in payload:
                continue
            row = _report_from_optimize(payload)
            kind = str(row.get("kind") or path.stem.removeprefix("optimize_"))
            row["kind"] = kind
            rows.append(row)
            seen.add(kind)
    matrix = _read_json(results / "paper_matrix.json") or {}
    for item in matrix.get("models") or []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "")
        if kind in seen:
            continue
        wrapped = dict(item)
        if "kind" not in wrapped:
            wrapped["kind"] = kind
        rows.append(_report_from_optimize(wrapped))
    hosts = [
        {"kind": row.get("kind"), "machine": (row.get("host") or {}).get("machine")}
        for row in rows
        if isinstance(row.get("host"), dict)
    ]
    machines = {item["machine"] for item in hosts if item.get("machine")}
    return scrub(
        {
            "rows": rows,
            "hosts": hosts,
            "mixed_hosts": len(machines) > 1,
            "fps_claimed": False,
        }
    )


def build_dashboard(
    results: Path,
    *,
    models: list[dict[str, str]] | None = None,
    uploads: Path | None = None,
) -> dict[str, Any]:
    """One card per fixture, zoo model, or upload, plus host facts."""

    if models is None:
        catalog = default_models() + upload_models(uploads or UPLOADS)
    else:
        catalog = models
    _rows, skipped = read_history(results / "history.jsonl")
    return scrub(
        {
            "results": str(results),
            "fps_claimed": False,
            "history_skipped": skipped,
            "host": host_strip(),
            "models": [_model_card(row, results) for row in catalog],
        }
    )
