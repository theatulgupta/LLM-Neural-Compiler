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
    assert "origin" in rows[0]


def test_corrupt_history_line_is_counted(tmp_path, tiny_path) -> None:
    from compiler.graph.graph_loader import load_graph
    from compiler.graph.graph_summary import summarize_graph
    from compiler.hardware.profile import probe_hardware
    from compiler.history.store import load_history, read_history
    from compiler.llm.context_builder import build_context
    from compiler.llm.prompting import build_messages

    record = new_run_record(
        run_id="h2",
        backend="ort_cpu",
        model={"path": "x.onnx", "kind": "yolov8n", "sha256": "abc"},
        compile={"ok": True, "ms": 1.0, "error": None},
        benchmark={"latency_ms": {"p50": 9.0}, "throughput_ips": 1.0},
        skip=None,
        verification={"passed": True},
        plan={"plan_id": "graph_fuse"},
    )
    path = tmp_path / "history.jsonl"
    append_history(path, record)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    rows, skipped = read_history(path)
    assert len(rows) == 1
    assert skipped == 1
    assert load_history(path)
    assert load_history.last_skipped == 1  # type: ignore[attr-defined]
    projected = history_for_model("yolov8n", history_path=path)
    assert projected[0]["plan_id"] == "graph_fuse"
    assert projected[0]["p50_ms"] == 9.0
    assert projected[0]["history_skipped"] == 1
    summary = summarize_graph(load_graph(tiny_path))
    text = build_messages(summary, build_context(summary, probe_hardware(), {}, projected))[1]["content"]
    assert "graph_fuse" in text
    assert "9.0" in text
    assert "1 corrupt history lines were skipped" in text
