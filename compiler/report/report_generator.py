"""Markdown report from measured JSON only."""

from __future__ import annotations

import json
from pathlib import Path

from compiler.catalog import REPO_ROOT

DEFAULT_RESULTS = REPO_ROOT / "experiments" / "results"


def _speedup(native, chosen) -> str:
    try:
        n = float(native)
        c = float(chosen)
    except (TypeError, ValueError):
        return ""
    if c <= 0:
        return ""
    return f"{n / c:.3f}"


def write_report(results_dir: Path | None = None) -> Path:
    results = results_dir or DEFAULT_RESULTS
    matrix_path = results / "paper_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8")) if matrix_path.is_file() else {}
    lines = [
        "# LLM-guided compiler report",
        "",
        "Numbers below are measured on this host. `fps_claimed` is false.",
        "",
    ]
    host = matrix.get("host") or {}
    lines += ["## Hardware", "", f"```json\n{json.dumps(host, indent=2)}\n```", ""]

    models = matrix.get("models") or []
    lines += [
        "## Summary",
        "",
        "| kind | task | baseline p50 | chosen plan | chosen p50 | speedup | passed | llm_source | llm_rank | llm_gap_pct | inputs_source |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in models:
        chosen = row.get("chosen") or {}
        native_p50 = None
        inputs_source = None
        for cand in row.get("candidates") or []:
            origin = str(cand.get("origin") or "")
            if origin.endswith("baseline") or cand.get("plan_id") == "baseline":
                native_p50 = cand.get("p50_ms")
            ver = cand.get("verification") or {}
            if ver.get("inputs_source"):
                inputs_source = ver.get("inputs_source")
        if row.get("native"):
            native_p50 = (row.get("native") or {}).get("p50_ms", native_p50)
        chosen_p50 = chosen.get("p50_ms")
        plan_id = chosen.get("plan_id") or ((chosen.get("plan") or {}).get("plan_id"))
        llm = row.get("llm") or {}
        lines.append(
            "| {kind} | {task} | {base} | {plan} | {p50} | {sp} | {ok} | {src_llm} | {rank} | {gap} | {src} |".format(
                kind=row.get("kind"),
                task=row.get("task") or "",
                base=native_p50 if native_p50 is not None else "",
                plan=plan_id or "",
                p50=chosen_p50 if chosen_p50 is not None else "",
                sp=_speedup(native_p50, chosen_p50),
                ok=chosen.get("passed"),
                src_llm=llm.get("source") or "",
                rank=row.get("llm_rank"),
                gap=row.get("llm_vs_oracle_gap_pct"),
                src=inputs_source or "",
            )
        )
    lines += ["", "## Models", ""]
    for row in models:
        kind = row.get("kind")
        lines.append(f"### {kind}")
        chosen = row.get("chosen") or {}
        lines.append(
            f"- chosen origin `{chosen.get('origin')}` plan `{chosen.get('plan_id')}` "
            f"p50_ms `{chosen.get('p50_ms')}` passed `{chosen.get('passed')}`"
        )
        llm = row.get("llm") or {}
        lines.append(
            f"- llm source `{llm.get('source')}` plan `{llm.get('plan_id')}` "
            f"rank `{row.get('llm_rank')}` oracle_gap_pct `{row.get('llm_vs_oracle_gap_pct')}` "
            f"fallback `{llm.get('fallback_reason')}`"
        )
        lines.append("")
        lines.append("| origin | plan | p50_ms | passed | nodes | bytes |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for cand in row.get("candidates") or []:
            nodes = f"{cand.get('nodes_before')}->{cand.get('nodes_after')}"
            bytes_ = (cand.get("profile") or {}).get("model_bytes")
            lines.append(
                f"| {cand.get('origin')} | {cand.get('plan_id')} | {cand.get('p50_ms')} | {cand.get('passed')} | {nodes} | {bytes_} |"
            )
        if row.get("native") and not row.get("candidates"):
            native = row["native"]
            opt = row.get("optimized") or {}
            lines.append(
                f"| native | {native.get('strategy')} | {native.get('p50_ms')} | {native.get('compile_ok')} | "
                f"{native.get('graph_before_nodes')}->{native.get('graph_after_nodes')} |  |"
            )
            lines.append(
                f"| optimized | {opt.get('strategy')} | {opt.get('p50_ms')} | {opt.get('compile_ok')} | "
                f"{opt.get('graph_before_nodes')}->{opt.get('graph_after_nodes')} |  |"
            )
        lines.append("")

    lines += [
        "## Gazebo camera loop",
        "",
        "| file | kind | variant | frames_rx | inferences | e2e p50 | e2e p95 | infer p50 | infer p95 | detections_total | ok |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    cam_files = sorted(results.glob("sim_inference_*.json"))
    if not cam_files:
        lines.append("| (none) |  |  |  |  |  |  |  |  |  |  |")
    for path in cam_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        stem = path.stem.removeprefix("sim_inference_")
        kind, _, variant = stem.rpartition("_")
        if not kind:
            kind, variant = stem, ""
        e2e = payload.get("e2e_ms") or {}
        infer = payload.get("infer_ms") or {}
        lines.append(
            "| {name} | {kind} | {var} | {fr} | {inf} | {e50} | {e95} | {i50} | {i95} | {det} | {ok} |".format(
                name=path.name,
                kind=kind,
                var=variant,
                fr=payload.get("frames_rx"),
                inf=payload.get("inferences"),
                e50=(e2e.get("p50") if isinstance(e2e, dict) else ""),
                e95=(e2e.get("p95") if isinstance(e2e, dict) else ""),
                i50=(infer.get("p50") if isinstance(infer, dict) else ""),
                i95=(infer.get("p95") if isinstance(infer, dict) else ""),
                det=payload.get("detections_total"),
                ok=payload.get("ok"),
            )
        )
    lines += [
        "",
        "## Not measured",
        "",
        "- TensorRT / NVIDIA GPU",
        "- Board power",
        "- COCO mAP (agreement vs FP32 reference is reported instead)",
        "",
    ]
    out = results / "report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
