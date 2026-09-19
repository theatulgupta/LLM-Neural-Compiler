"""Rank candidate plans on a model. History feeds the next prompt."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from compiler.catalog import REPO_ROOT
from compiler.errors import FrontendSkip
from compiler.exporters import skip_run_record
from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph
from compiler.hardware.profile import probe_hardware
from compiler.history import history_for_model
from compiler.llm.llm_client import LlmClient, build_client
from compiler.llm.session import propose_plan
from compiler.pipeline.compile import compile_verify_profile
from compiler.planner.candidates import generate_candidates, json_key
from compiler.planner.plan import Plan
from nnc.probe import probe_host

DEFAULT_RESULTS = REPO_ROOT / "experiments" / "results"
_OFFLINE_LLM = frozenset({"heuristic", "mock"})


def _row_from_record(cand, record: dict[str, Any], *, measured: bool) -> dict[str, Any]:
    p50 = ((record.get("benchmark") or {}).get("latency_ms") or {}).get("p50")
    passed = bool((record.get("verification") or {}).get("passed"))
    return {
        "origin": cand.origin,
        "plan_id": cand.plan.plan_id,
        "plan": cand.plan.to_dict(),
        "p50_ms": p50,
        "passed": passed,
        "compile_ok": bool((record.get("compile") or {}).get("ok")),
        "run_id": record.get("run_id"),
        "graph_changed": record.get("graph_changed"),
        "verification": record.get("verification"),
        "profile": record.get("profile"),
        "benchmark": record.get("benchmark"),
        "artifact": record.get("artifact"),
        "nodes_before": (record.get("graph_before") or {}).get("node_count"),
        "nodes_after": (record.get("graph_after") or {}).get("node_count"),
        "same_as": cand.same_as,
        "measured": measured,
    }


def _copy_row(base: dict[str, Any], cand) -> dict[str, Any]:
    row = dict(base)
    row["origin"] = cand.origin
    row["plan_id"] = cand.plan.plan_id
    row["plan"] = cand.plan.to_dict()
    row["same_as"] = cand.same_as
    row["measured"] = False
    return row


def _rank(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    passed = [row for row in rows if row.get("passed") and row.get("p50_ms")]
    passed.sort(key=lambda row: float(row["p50_ms"]))
    return passed


def _origin_rank(ranked: list[dict[str, Any]], origin: str) -> int | None:
    for index, row in enumerate(ranked, 1):
        if row.get("origin") == origin:
            return index
    return None


def _strictly_slower(llm_row: dict[str, Any], best: dict[str, Any]) -> bool:
    try:
        return float(llm_row["p50_ms"]) > float(best["p50_ms"])
    except (TypeError, ValueError, KeyError):
        return False


def _followup_kind(
    *,
    source: str | None,
    llm_row: dict[str, Any] | None,
    ranked: list[dict[str, Any]],
) -> str | None:
    """Return 'revised', 'improved', or None. HTTP only; at most one extra round."""

    if llm_row is None or source is None or source in _OFFLINE_LLM:
        return None
    if not llm_row.get("compile_ok") or not llm_row.get("passed"):
        return "revised"
    if ranked and _strictly_slower(llm_row, ranked[0]):
        return "improved"
    return None


def _plan_atoms(plan: dict[str, Any] | None) -> str:
    names: list[str] = []
    for step in (plan or {}).get("steps") or []:
        if isinstance(step, dict):
            atom = str(step.get("atom") or "")
        else:
            atom = str(step)
        if atom:
            names.append(atom)
    return ",".join(names) or "-"


def _plan_options(plan: dict[str, Any] | None) -> str:
    opts = (plan or {}).get("options") or {}
    parts = []
    for key in ("ort_graph_opt", "execution_mode", "intra_op_threads"):
        if key in opts:
            parts.append(f"{key}={opts[key]}")
    return " ".join(parts) or "-"


def _gate_error(row: dict[str, Any]) -> str:
    ver = row.get("verification") or {}
    err = ver.get("error")
    if err:
        return str(err)[:120]
    if not row.get("passed") or not row.get("compile_ok"):
        return f"passed={row.get('passed')} compile_ok={row.get('compile_ok')}"
    return ""


def _measured_table(rows: list[dict[str, Any]], *, limit: int = 10) -> str:
    lines = ["measured_table origin plan_id p50 passed atoms options"]
    for row in rows[:limit]:
        plan = row.get("plan") if isinstance(row.get("plan"), dict) else {}
        extra = _gate_error(row)
        line = (
            f"{row.get('origin')} {row.get('plan_id')} p50={row.get('p50_ms')} "
            f"passed={row.get('passed')} atoms={_plan_atoms(plan)} opts={_plan_options(plan)}"
        )
        if extra:
            line += f" gate={extra}"
        lines.append(line)
    return "\n".join(lines)


def _llm_block(
    cand,
    row: dict[str, Any] | None,
    ranked: list[dict[str, Any]],
    revised: dict[str, Any] | None,
    improved: dict[str, Any] | None = None,
) -> dict[str, Any]:
    outcome = cand.outcome
    p50 = None if row is None else row.get("p50_ms")
    oracle = float(ranked[0]["p50_ms"]) if ranked else None
    gap = None
    if oracle and p50:
        gap = (float(p50) - oracle) / oracle * 100.0
    return {
        "source": None if outcome is None else outcome.source,
        "plan_id": cand.plan.plan_id,
        "attempts": [] if outcome is None else [item.to_dict() for item in outcome.attempts],
        "fallback_reason": None if outcome is None else outcome.fallback_reason,
        "same_as": cand.same_as,
        "rank": _origin_rank(ranked, "llm"),
        "gap_pct": gap,
        "passed": None if row is None else row.get("passed"),
        "p50_ms": p50,
        "revised": revised,
        "improved": improved,
    }


class _Followup:
    """Minimal candidate-shaped row for compile_verify_profile follow-ups."""

    def __init__(self, origin: str, plan: Plan) -> None:
        self.origin = origin
        self.plan = plan
        self.same_as = None


def _known_keys(candidates, rows: list[dict[str, Any]]) -> dict[str, str]:
    keys: dict[str, str] = {}
    for item in candidates:
        keys.setdefault(json_key(item.plan), item.origin)
    for row in rows:
        plan = row.get("plan")
        origin = str(row.get("origin") or "unknown")
        if isinstance(plan, dict):
            keys.setdefault(json_key(Plan.from_dict(plan)), origin)
    return keys


def _compile_followup(
    *,
    summary,
    hardware,
    history: list[dict[str, Any]],
    client: LlmClient,
    candidates,
    rows: list[dict[str, Any]],
    model_path: Path,
    kind: str,
    task: str,
    results: Path,
    warmup: int,
    iters: int,
    origin: str,
    feedback: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    extra = propose_plan(
        summary,
        hardware,
        {},
        history,
        client,
        max_attempts=1,
        feedback=feedback,
    )
    if extra.source in _OFFLINE_LLM:
        return rows, {"skipped": f"fallback: {extra.fallback_reason or extra.source}"}
    if not extra.accepted:
        return rows, {"skipped": extra.fallback_reason or "not accepted"}
    rkey = json_key(extra.plan)
    existing = _known_keys(candidates, rows)
    if rkey in existing:
        return rows, {"skipped": "same as existing candidate", "same_as": existing[rkey]}
    record = compile_verify_profile(
        model_path,
        plan=extra.plan,
        kind=kind,
        task=task,
        results_dir=results,
        warmup=warmup,
        iters=iters,
    )
    follow = _Followup(origin, extra.plan)
    extra_row = _row_from_record(follow, record, measured=True)
    rows = [*rows, extra_row]
    ranked = _rank(rows)
    return rows, {
        "plan_id": extra.plan.plan_id,
        "p50_ms": extra_row.get("p50_ms"),
        "passed": extra_row.get("passed"),
        "rank": _origin_rank(ranked, origin),
    }


def optimize_model(
    model_path: Path,
    *,
    kind: str,
    task: str = "detect",
    mode: str = "default",
    warmup: int = 5,
    iters: int = 20,
    results_dir: Path | None = None,
    client: LlmClient | None = None,
) -> dict[str, Any]:
    results = results_dir or DEFAULT_RESULTS
    try:
        loaded = load_graph(model_path)
    except FrontendSkip as exc:
        record = skip_run_record(kind=kind, path=str(model_path), reason=exc.reason, backend="ort_cpu")
        skip = dict(record.get("skip") or {})
        skip["frontend"] = exc.frontend
        record["skip"] = skip
        out = results / f"optimize_{kind.replace('/', '_')}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        record["wrote"] = str(out)
        record["chosen"] = None
        record["fps_claimed"] = False
        return record
    summary = summarize_graph(loaded)
    hardware = probe_hardware()
    history = history_for_model(kind)
    active = client or build_client()
    candidates = generate_candidates(summary, hardware, {}, active, history, mode=mode)
    rows: list[dict[str, Any]] = []
    by_origin: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        if cand.same_as and cand.same_as in by_origin:
            row = _copy_row(by_origin[cand.same_as], cand)
            rows.append(row)
            by_origin[cand.origin] = row
            continue
        record = compile_verify_profile(
            model_path,
            plan=cand.plan,
            kind=kind,
            task=task,
            results_dir=results,
            warmup=warmup,
            iters=iters,
        )
        row = _row_from_record(cand, record, measured=True)
        rows.append(row)
        by_origin[cand.origin] = row

    ranked = _rank(rows)
    llm_cand = next((item for item in candidates if item.origin == "llm"), None)
    llm_row = by_origin.get("llm")
    revised_meta: dict[str, Any] | None = None
    improved_meta: dict[str, Any] | None = None
    outcome = None if llm_cand is None else llm_cand.outcome
    source = None if outcome is None else outcome.source
    kind_follow = _followup_kind(source=source, llm_row=llm_row, ranked=ranked)
    table = _measured_table(rows)
    follow_kwargs = {
        "summary": summary,
        "hardware": hardware,
        "history": history,
        "client": active,
        "candidates": candidates,
        "rows": rows,
        "model_path": model_path,
        "kind": kind,
        "task": task,
        "results": results,
        "warmup": warmup,
        "iters": iters,
    }
    if kind_follow == "revised" and llm_row is not None:
        err = _gate_error(llm_row)
        prefix = f"llm_failed {err}\n" if err else ""
        rows, revised_meta = _compile_followup(
            **follow_kwargs,
            origin="llm_revised",
            feedback=[{"n": 1, "outcome": "measured_failure", "detail": f"{prefix}{table}"}],
        )
    elif kind_follow == "improved" and llm_row is not None and ranked:
        best = ranked[0]
        gap = None
        if best.get("p50_ms") and llm_row.get("p50_ms"):
            gap = (float(llm_row["p50_ms"]) - float(best["p50_ms"])) / float(best["p50_ms"]) * 100.0
        header = (
            f"chosen_origin={best.get('origin')} chosen_plan={best.get('plan_id')} "
            f"chosen_p50={best.get('p50_ms')} llm_plan={llm_row.get('plan_id')} "
            f"llm_p50={llm_row.get('p50_ms')} gap_pct={gap}"
        )
        rows, improved_meta = _compile_followup(
            **follow_kwargs,
            origin="llm_improved",
            feedback=[{"n": 1, "outcome": "measured_not_best", "detail": f"{header}\n{table}"}],
        )

    ranked = _rank(rows)
    chosen = ranked[0] if ranked else None
    llm_payload = None
    if llm_cand is not None:
        llm_payload = _llm_block(llm_cand, llm_row, ranked, revised_meta, improved_meta)
    payload = {
        "kind": kind,
        "task": task,
        "path": str(model_path),
        "sha256": loaded.sha256,
        "candidates": rows,
        "ranking": [row["origin"] for row in ranked],
        "chosen": chosen,
        "llm": llm_payload,
        "llm_rank": None if llm_payload is None else llm_payload["rank"],
        "heuristic_rank": _origin_rank(ranked, "heuristic"),
        "llm_vs_oracle_gap_pct": None if llm_payload is None else llm_payload["gap_pct"],
        "llm_confidence": (llm_cand.plan.confidence if llm_cand else None),
        "fps_claimed": False,
        "host": probe_host()["platform"],
    }
    out = results / f"optimize_{kind.replace('/', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    payload["wrote"] = str(out)
    return payload
