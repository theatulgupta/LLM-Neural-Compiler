from __future__ import annotations

from compiler.history import append_history, history_for_model, new_run_record


def test_history_feedback(tmp_path) -> None:
    record = new_run_record(
        run_id="h1",
        backend="ort_cpu",
        model={"path": "x.onnx", "kind": "yolov8n", "sha256": "abc"},
        compile={"ok": False, "ms": None, "error": "n/a"},
        benchmark={"latency_ms": {"p50": 12.0}, "throughput_ips": 1.0},
        skip=None,
        verification={"passed": True},
        plan={"plan_id": "graph_fuse"},
    )
    path = tmp_path / "history.jsonl"
    append_history(path, record)
    rows = history_for_model("yolov8n", history_path=path)
    assert rows[0]["plan_id"] == "graph_fuse"
    assert rows[0]["p50_ms"] == 12.0
