"""Markdown report from measured JSON only."""

from __future__ import annotations

import json
from pathlib import Path

from compiler.catalog import REPO_ROOT

DEFAULT_RESULTS = REPO_ROOT / "experiments" / "results"


def write_report(results_dir: Path | None = None) -> Path:
    results = results_dir or DEFAULT_RESULTS
    matrix_path = results / "paper_matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8")) if matrix_path.is_file() else {}
    lines = ["# LLM-guided compiler report", "", "Numbers below are measured on this host. `fps_claimed` is false.", ""]
    host = matrix.get("host") or {}
    lines += ["## Hardware", "", f"```json\n{json.dumps(host, indent=2)}\n```", ""]
    lines += ["## Models", ""]
    models = matrix.get("models") or []
    for row in models:
        kind = row.get("kind")
        lines.append(f"### {kind}")
        chosen = row.get("chosen") or {}
        lines.append(
            f"- chosen origin `{chosen.get('origin')}` plan `{chosen.get('plan_id')}` "
            f"p50_ms `{chosen.get('p50_ms')}` passed `{chosen.get('passed')}`"
        )
        lines.append(f"- llm_rank `{row.get('llm_rank')}` oracle_gap_pct `{row.get('llm_vs_oracle_gap_pct')}`")
        lines.append("")
        lines.append("| origin | plan | p50_ms | passed | nodes | bytes |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for cand in row.get("candidates") or []:
            nodes = f"{cand.get('nodes_before')}->{cand.get('nodes_after')}"
            bytes_ = ((cand.get("profile") or {}).get("model_bytes"))
            lines.append(
                f"| {cand.get('origin')} | {cand.get('plan_id')} | {cand.get('p50_ms')} | {cand.get('passed')} | {nodes} | {bytes_} |"
            )
        lines.append("")
    lines += [
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
