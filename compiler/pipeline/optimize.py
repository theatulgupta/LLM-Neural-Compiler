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


def _llm_block(
    cand,
    row: dict[str, Any] | None,
    ranked: list[dict[str, Any]],
    revised: dict[str, Any] | None,
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
    chosen = ranked[0] if ranked else None
    llm_cand = next((item for item in candidates if item.origin == "llm"), None)
    llm_row = by_origin.get("llm")
    revised_meta: dict[str, Any] | None = None
    outcome = None if llm_cand is None else llm_cand.outcome
    llm_failed = bool(
        llm_row
        and outcome is not None
        and outcome.source not in _OFFLINE_LLM
        and (not llm_row.get("compile_ok") or not llm_row.get("passed"))
    )
    if llm_failed and llm_cand is not None:
        detail = ((llm_row.get("verification") or {}).get("error")) or (
            f"passed={llm_row.get('passed')} compile_ok={llm_row.get('compile_ok')}"
        )
        revised = propose_plan(
            summary,
            hardware,
            {},
            history,
            active,
            max_attempts=1,
            feedback=[
                {
                    "n": 1,
                    "outcome": "measured_failure",
                    "detail": str(detail),
                }
            ],
        )
        rkey = json_key(revised.plan)
        existing = {json_key(item.plan) for item in candidates}
        if not revised.accepted:
            revised_meta = {"skipped": revised.fallback_reason or "not accepted"}
        elif rkey in existing:
            revised_meta = {"skipped": "same as existing candidate"}
        else:
            record = compile_verify_profile(
                model_path,
                plan=revised.plan,
                kind=kind,
                task=task,
                results_dir=results,
                warmup=warmup,
                iters=iters,
            )

            class _Rev:
                origin = "llm_revised"
                plan = revised.plan
                same_as = None

            rev_row = _row_from_record(_Rev(), record, measured=True)
            rows.append(rev_row)
            ranked = _rank(rows)
            revised_meta = {
                "plan_id": revised.plan.plan_id,
                "p50_ms": rev_row.get("p50_ms"),
                "passed": rev_row.get("passed"),
                "rank": _origin_rank(ranked, "llm_revised"),
            }

    llm_payload = None
    if llm_cand is not None:
        llm_payload = _llm_block(llm_cand, llm_row, ranked, revised_meta)
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
