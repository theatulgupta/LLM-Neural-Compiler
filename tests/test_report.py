from __future__ import annotations

import json

from compiler.report.report_generator import write_report


def test_report_from_matrix(tmp_path) -> None:
    matrix = {
        "host": {"machine": "aarch64"},
        "models": [
            {
                "kind": "fixture",
                "chosen": {"origin": "preset:baseline", "plan_id": "baseline", "p50_ms": 1.0, "passed": True},
                "candidates": [
                    {
                        "origin": "preset:baseline",
                        "plan_id": "baseline",
                        "p50_ms": 1.0,
                        "passed": True,
                        "nodes_before": 5,
                        "nodes_after": 5,
                        "profile": {"model_bytes": 100},
                    }
                ],
                "llm_rank": 1,
                "llm_vs_oracle_gap_pct": 0,
            }
        ],
    }
    (tmp_path / "paper_matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
    path = write_report(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "fixture" in text
    assert "aarch64" in text
    assert "Not measured" in text
