"""One background worker for operator actions. A second job waits until this one finishes."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from compiler.catalog import REPO_ROOT, get_model
from compiler.errors import FrontendSkip
from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph
from compiler.hardware.profile import probe_hardware
from compiler.history import history_for_model
from compiler.llm.context_builder import build_context
from compiler.llm.prompting import _applicable, build_messages
from compiler.llm.recommendation_engine import recommend_plan
from compiler.pipeline.compile import compile_verify_profile
from compiler.pipeline.optimize import optimize_model
from compiler.planner.plan import get_plan
from compiler.report.report_generator import write_report
from compiler.schema import SchemaError
from compiler.ui.view import _read_json, report_rows, scrub
from nnc.artifact import benchmark_artifact, load_artifact_with_sidecar

_ACTIONS = {
    "analyze",
    "plan",
    "optimize",
    "report",
    "emit-fixture",
    "export",
    "prove-fusion",
    "test",
}
_NEEDS_MODEL = {"analyze", "plan", "optimize", "prove-fusion"}


class JobBusy(RuntimeError):
    """A compile or plan job is already queued or running."""


def _run(action: str, params: dict[str, Any], results: Path) -> dict[str, Any]:
    if action == "report":
        path = write_report(results)
        body = report_rows(results)
        body["wrote"] = str(path)
        return body
    if action == "emit-fixture":
        return _emit_fixture()
    if action == "export":
        return _export(str(params.get("kind") or ""))
    if action == "test":
        return _remeasure(
            str(params.get("kind") or ""),
            results,
            warmup=int(params.get("warmup") or 1),
            iters=int(params.get("iters") or 5),
        )
    model = Path(str(params.get("model") or ""))
    if action in _NEEDS_MODEL and not model.is_file():
        raise FileNotFoundError(f"model not found: {model}")
    kind = str(params.get("kind") or "onnx")
    task = str(params.get("task") or "classify")
    if action == "analyze":
        _loaded, summary = _summarize(model)
        data = summary.to_dict()
        return {"summary": data, "notes": list(summary.notes), "fps_claimed": False}
    if action == "plan":
        return _plan_view(model, kind)
    if action == "optimize":
        warmup = int(params.get("warmup") or 2)
        iters = int(params.get("iters") or 5)
        payload = optimize_model(
            model,
            kind=kind,
            task=task,
            mode=str(params.get("candidates") or "default"),
            warmup=warmup,
            iters=iters,
            results_dir=results,
        )
        return {
            "wrote": payload.get("wrote"),
            "chosen": payload.get("chosen"),
            "llm": payload.get("llm"),
            "llm_rank": payload.get("llm_rank"),
            "ranking": payload.get("ranking"),
            "fps_claimed": False,
        }
    if action == "prove-fusion":
        return _prove_fusion(model, results, kind=kind, task=task)
    raise ValueError(f"unknown action {action!r}")


def _emit_fixture() -> dict[str, Any]:
    from compiler.parsers.tiny_cnn import write_tiny_cnn

    path = REPO_ROOT / "fixtures" / "tiny_cnn.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_tiny_cnn(path)
    return {"path": str(path), "onnx_present": path.is_file(), "fps_claimed": False}


def _export(kind: str) -> dict[str, Any]:
    spec = get_model(kind)
    cmd = spec.export_cmd(python=sys.executable)
    completed = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False, capture_output=True, text=True)
    present = spec.onnx_path().is_file()
    if completed.returncode != 0 or not present:
        tail = (completed.stderr or completed.stdout or "export failed").strip()[-400:]
        raise RuntimeError(f"export {kind} failed ({completed.returncode}): {tail}")
    return {"kind": kind, "path": str(spec.onnx_path()), "onnx_present": True, "fps_claimed": False}


def _plan_view(model: Path, kind: str) -> dict[str, Any]:
    loaded, summary = _summarize(model)
    hardware = probe_hardware()
    history = history_for_model(kind)
    rec = recommend_plan(summary, hardware, {}, history=history)
    outcome = rec.outcome
    context = build_context(summary, hardware, {}, history)
    applicable, skipped = _applicable(summary, context)
    prompt = build_messages(summary, context, feedback=None)[1]["content"]
    plan = rec.strategy
    fallback = None if outcome is None else outcome.fallback_reason
    source = rec.source
    return {
        "sha256": loaded.sha256,
        "notes": list(summary.notes),
        "applicable": applicable,
        "skipped_atoms": skipped,
        "prompt": prompt,
        "source": source,
        "fallback_reason": fallback,
        "heuristic_fallback": bool(fallback) and source == "heuristic",
        "offline_heuristic": source == "heuristic" and not fallback,
        "accepted": None if outcome is None else outcome.accepted,
        "attempts": [] if outcome is None else [item.to_dict() for item in outcome.attempts],
        "plan": {
            "plan_id": plan.plan_id,
            "steps": [str(step.get("atom")) for step in plan.steps],
            "options": dict(plan.options),
            "rationale": plan.rationale,
        },
        "fps_claimed": False,
    }


def _prove_fusion(model: Path, results: Path, *, kind: str, task: str) -> dict[str, Any]:
    record = compile_verify_profile(
        model,
        plan=get_plan("graph_fuse"),
        kind=kind,
        task=task,
        results_dir=results,
        warmup=0,
        iters=1,
    )
    before = (record.get("graph_before") or {}).get("op_counts") or {}
    after = (record.get("graph_after") or {}).get("op_counts") or {}
    return {
        "relu_before": int(before.get("Relu") or 0),
        "relu_after": int(after.get("Relu") or 0),
        "fused_conv": int(after.get("FusedConv") or 0),
        "passed": bool((record.get("verification") or {}).get("passed")),
        "fps_claimed": False,
    }


def _remeasure(kind: str, results: Path, *, warmup: int, iters: int) -> dict[str, Any]:
    payload = _read_json(results / f"optimize_{kind.replace('/', '_')}.json") or {}
    chosen = payload.get("chosen") if isinstance(payload.get("chosen"), dict) else {}
    artifact = chosen.get("artifact") if isinstance(chosen, dict) else None
    path = artifact.get("path") if isinstance(artifact, dict) else None
    if not path:
        for row in payload.get("candidates") or []:
            if isinstance(row, dict) and row.get("origin") == chosen.get("origin"):
                found = row.get("artifact")
                if isinstance(found, dict) and found.get("path"):
                    path = found["path"]
                    break
    if not path or not Path(str(path)).is_file():
        raise FileNotFoundError("no artifact yet; optimize this model first")
    loaded = load_artifact_with_sidecar(Path(str(path)))
    measured = benchmark_artifact(loaded, warmup=warmup, iters=iters)
    latency = measured.get("latency_ms") or {}
    return {
        "artifact": str(path),
        "ranked_p50_ms": chosen.get("p50_ms"),
        "gate_passed": chosen.get("passed"),
        "p50_ms": latency.get("p50"),
        "p95_ms": latency.get("p95"),
        "fps_claimed": False,
    }


def _summarize(model: Path):
    loaded = load_graph(model)
    return loaded, summarize_graph(loaded)


class JobRunner:
    """Single worker. submit() returns immediately with status queued."""

    def __init__(self, results: Path) -> None:
        self.results = results
        self._jobs: dict[str, dict[str, Any]] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="nnc-ui-job", daemon=True)
        self._thread.start()

    def submit(self, action: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if action not in _ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        with self._lock:
            if any(item["status"] in {"queued", "running"} for item in self._jobs.values()):
                raise JobBusy("a job is already running")
            job_id = uuid.uuid4().hex[:12]
            job = {
                "id": job_id,
                "action": action,
                "status": "queued",
                "detail": "",
                "result": None,
                "params": dict(params or {}),
            }
            self._jobs[job_id] = job
            self._order.append(job_id)
        self._queue.put(job_id)
        return self.get(job_id)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return scrub(_public(job))

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [scrub(_public(self._jobs[job_id])) for job_id in self._order]

    def close(self) -> None:
        self._queue.put(None)

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    continue
                job["status"] = "running"
                action = str(job["action"])
                params = dict(job["params"])
            self._execute(job_id, action, params)

    def _execute(self, job_id: str, action: str, params: dict[str, Any]) -> None:
        previous_llm = os.environ.get("NNC_LLM")
        previous_model = os.environ.get("NNC_LLM_MODEL")
        _apply_llm_choice(params)
        try:
            result = _run(action, params, self.results)
            status = "done"
            detail = "ok"
        except (FrontendSkip, SchemaError, OSError, ValueError, RuntimeError, KeyError) as exc:
            result = None
            status = "error"
            detail = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            result = None
            status = "error"
            detail = f"{type(exc).__name__}: {exc}"
        finally:
            _restore_env("NNC_LLM", previous_llm)
            _restore_env("NNC_LLM_MODEL", previous_model)
        with self._lock:
            job = self._jobs[job_id]
            job["status"] = status
            job["detail"] = detail
            job["result"] = result


def _apply_llm_choice(params: dict[str, Any]) -> None:
    """Point this job at one coded host and model id. Does not read or store a key."""

    choice = str(params.get("llm") or "heuristic").strip().lower()
    model_id = str(params.get("model_id") or "").strip()
    if choice in {"", "heuristic"}:
        os.environ["NNC_LLM"] = "heuristic"
        return
    os.environ["NNC_LLM"] = choice
    if model_id:
        os.environ["NNC_LLM_MODEL"] = model_id
    else:
        os.environ.pop("NNC_LLM_MODEL", None)


def _restore_env(name: str, previous: str | None) -> None:
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous


def _public(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": job["id"],
        "action": job["action"],
        "status": job["status"],
        "detail": job["detail"],
        "result": job["result"],
    }
