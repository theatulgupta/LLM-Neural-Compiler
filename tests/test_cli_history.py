from __future__ import annotations

import json

from compiler.cli import main
from compiler.history import append_history, new_run_record, write_run_json
from compiler.parsers.tiny_cnn import flatten_features


def test_history_roundtrip(tmp_path) -> None:
    record = new_run_record(
        run_id="abc",
        backend="ort_cpu",
        model={"path": "tiny.onnx", "kind": "fixture"},
        compile={"ok": False, "ms": None, "error": "unit-test incomplete record"},
        benchmark=None,
        skip=None,
    )
    path = tmp_path / "run.json"
    write_run_json(path, record)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["run_id"] == "abc"
    history = tmp_path / "history.jsonl"
    append_history(history, record)
    lines = history.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1


def test_plan_show_prompt_includes_history(tiny_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        "compiler.cli.app.history_for_model",
        lambda kind: [{"plan_id": "graph_fuse", "p50_ms": 1.5, "passed": True}],
    )
    assert main(["plan", str(tiny_path), "--show-prompt", "--kind", "fixture"]) == 0
    text = capsys.readouterr().out
    assert "graph_fuse" in text
    assert "1.5" in text


def test_cli_emit_and_analyze(tmp_path) -> None:
    fixture = tmp_path / "tiny.onnx"
    assert main(["emit-fixture", "--out", str(fixture)]) == 0
    assert fixture.is_file()
    assert flatten_features() == 64
    assert main(["analyze", str(fixture)]) == 0
    assert main(["recommend", str(fixture)]) == 0
    assert main(["infer", str(fixture), "--graph-opt", "disable", "--warmup", "0", "--iters", "2"]) == 0
