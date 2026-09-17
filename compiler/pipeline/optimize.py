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
from compiler.llm.llm_client import build_client
from compiler.pipeline.compile import compile_verify_profile
from compiler.planner.candidates import generate_candidates
from nnc.probe import probe_host

DEFAULT_RESULTS = REPO_ROOT / "experiments" / "results"


def optimize_model(
    model_path: Path,
    *,
    kind: str,
    task: str = "detect",
    mode: str = "default",
    warmup: int = 5,
    iters: int = 20,
    results_dir: Path | None = None,
) -> dict[str, Any]:
    results = results_dir or DEFAULT_RESULTS
    try:
        loaded = load_graph(model_path)
    except FrontendSkip as exc:
        record = skip_run_record(
            kind=kind, path=str(model_path), reason=exc.reason, backend="ort_cpu"
        )
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
    client = build_client()
    candidates = generate_candidates(summary, hardware, {}, client, history, mode=mode)
    rows = []
    llm_rank = None
    for index, cand in enumerate(candidates):
        record = compile_verify_profile(
            model_path,
            plan=cand.plan,
            kind=kind,
            task=task,
            results_dir=results,
            warmup=warmup,
            iters=iters,
        )
        p50 = ((record.get("benchmark") or {}).get("latency_ms") or {}).get("p50")
        passed = bool((record.get("verification") or {}).get("passed"))
        row = {
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
        }
        rows.append(row)
        if cand.origin == "llm":
            llm_rank = index
    passed_rows = [row for row in rows if row["passed"] and row["p50_ms"]]
    passed_rows.sort(key=lambda row: float(row["p50_ms"]))
    chosen = passed_rows[0] if passed_rows else None
    ranking = [row["origin"] for row in passed_rows]
    oracle_p50 = float(passed_rows[0]["p50_ms"]) if passed_rows else None
    llm_rows = [row for row in rows if row["origin"] == "llm"]
    llm_p50 = float(llm_rows[0]["p50_ms"]) if llm_rows and llm_rows[0]["p50_ms"] else None
    gap = None
    if oracle_p50 and llm_p50:
        gap = (llm_p50 - oracle_p50) / oracle_p50 * 100.0
    payload = {
        "kind": kind,
        "task": task,
        "path": str(model_path),
        "sha256": loaded.sha256,
        "candidates": rows,
        "ranking": ranking,
        "chosen": chosen,
        "llm_rank": (ranking.index("llm") + 1) if "llm" in ranking else llm_rank,
        "llm_vs_oracle_gap_pct": gap,
        "llm_confidence": (llm_rows[0]["plan"].get("confidence") if llm_rows else None),
        "fps_claimed": False,
        "host": probe_host()["platform"],
    }
    out = results / f"optimize_{kind.replace('/', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    payload["wrote"] = str(out)
    return payload
