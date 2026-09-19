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
    assert "## Summary" in text
    assert "## Gazebo camera loop" in text
    assert "(none)" in text


def test_report_summary_row(tmp_path) -> None:
    matrix = {
        "host": {"machine": "aarch64"},
        "models": [
            {
                "kind": "fixture",
                "task": "classify",
                "chosen": {"origin": "preset:baseline", "plan_id": "baseline", "p50_ms": 1.0, "passed": True},
                "candidates": [
                    {
                        "origin": "preset:baseline",
                        "plan_id": "baseline",
                        "p50_ms": 2.0,
                        "passed": True,
                        "nodes_before": 5,
                        "nodes_after": 5,
                        "profile": {"model_bytes": 100},
                        "verification": {"inputs_source": "assets"},
                    }
                ],
                "llm_rank": 1,
                "llm_vs_oracle_gap_pct": 0,
                "llm": {
                    "source": "openai",
                    "plan_id": "baseline",
                    "improved": {"plan_id": "uav_try_threads", "p50_ms": 0.8, "passed": True, "rank": 1},
                    "revised": None,
                },
            }
        ],
    }
    (tmp_path / "paper_matrix.json").write_text(json.dumps(matrix), encoding="utf-8")
    (tmp_path / "sim_inference_yolov8n_native.json").write_text(
        json.dumps(
            {
                "ok": True,
                "frames_rx": 3,
                "inferences": 2,
                "e2e_ms": {"p50": 10.0, "p95": 12.0},
                "infer_ms": {"p50": 8.0, "p95": 9.0},
                "detections_total": 0,
                "fps_claimed": False,
            }
        ),
        encoding="utf-8",
    )
    path = write_report(tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "assets" in text
    assert "2.0" in text
    assert "sim_inference_yolov8n_native.json" in text
    assert "Gazebo camera loop" in text
    assert "chosen origin" in text
    assert "llm.improved" in text
    assert "uav_try_threads" in text
    assert "llm_followup" in text
