"""Operator console: dashboard facts, loopback bind, one job at a time."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from compiler.ui.jobs import JobBusy, JobRunner, _run
from compiler.ui.server import require_loopback, start_background
from compiler.ui.view import build_dashboard, report_rows, save_upload


def test_dashboard_reads_results_and_scrubs_secrets(tmp_path) -> None:
    payload = {
        "kind": "fixture",
        "task": "classify",
        "candidates": [{"origin": "preset:baseline", "plan_id": "baseline", "p50_ms": 1.2, "passed": True}],
        "chosen": {"origin": "preset:baseline", "plan_id": "baseline", "p50_ms": 1.2, "passed": True},
        "llm": {
            "source": "heuristic",
            "attempts": [
                {
                    "n": 1,
                    "outcome": "rejected",
                    "detail": "fuse_conv_relu: pattern absent",
                    "corrections": [
                        {"step": "fuse_conv_relu", "reason": "pattern conv_relu absent", "level": "drop"}
                    ],
                }
            ],
            "fallback_reason": "leak gsk_x",
            "revised": {"skipped": "fallback: transport"},
            "improved": None,
        },
        "fps_claimed": False,
    }
    (tmp_path / "optimize_fixture.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "history.jsonl").write_text("{not-json\n", encoding="utf-8")
    dash = build_dashboard(
        tmp_path,
        models=[{"kind": "fixture", "task": "classify", "path": str(tmp_path / "missing.onnx")}],
    )
    card = dash["models"][0]
    assert card["onnx_present"] is False
    assert card["baseline_p50_ms"] == 1.2
    assert card["chosen"]["plan_id"] == "baseline"
    assert card["llm"]["attempts"][0]["corrections"][0]["level"] == "drop"
    assert card["llm"]["revised"]["skipped"].startswith("fallback")
    assert card["llm"]["fallback_reason"] == "[redacted]"
    assert dash["history_skipped"] == 1
    assert dash["fps_claimed"] is False
    assert "gsk_" not in json.dumps(dash)


def test_report_rows_without_matrix(tmp_path) -> None:
    payload = {
        "kind": "fixture",
        "task": "classify",
        "host": {"machine": "aarch64"},
        "candidates": [{"origin": "preset:baseline", "plan_id": "baseline", "p50_ms": 2.0, "passed": True}],
        "chosen": {"origin": "preset:graph_fuse", "plan_id": "graph_fuse", "p50_ms": 1.0, "passed": True},
        "llm": {"source": "heuristic", "revised": {"skipped": "same as existing candidate"}},
        "fps_claimed": False,
    }
    (tmp_path / "optimize_fixture.json").write_text(json.dumps(payload), encoding="utf-8")
    report = report_rows(tmp_path)
    assert report["rows"][0]["chosen_plan"] == "graph_fuse"
    assert report["rows"][0]["speedup"] == 2.0
    assert report["rows"][0]["followup"].startswith("skipped")
    assert report["fps_claimed"] is False
    assert report["mixed_hosts"] is False


def test_upload_rejects_non_onnx(tmp_path) -> None:
    with pytest.raises(ValueError, match="onnx"):
        save_upload(tmp_path, "notes.txt", b"not a model")


def test_http_upload_rejects_non_onnx(tmp_path) -> None:
    server, app = start_background("127.0.0.1", 0, tmp_path)
    port = server.server_address[1]
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/upload?name=notes.txt",
            data=b"not a model",
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request)
        assert caught.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        app.close()


def test_analyze_notes_mention_flatten(tiny_path, tmp_path) -> None:
    result = _run("analyze", {"model": str(tiny_path), "kind": "fixture"}, tmp_path)
    assert any("Flatten" in note for note in result["notes"])
    assert result["fps_claimed"] is False


def test_prove_fusion_on_fixture(tiny_path, tmp_path) -> None:
    result = _run(
        "prove-fusion",
        {"model": str(tiny_path), "kind": "fixture", "task": "classify"},
        tmp_path,
    )
    assert result["fused_conv"] >= 1
    assert result["relu_after"] == 0
    assert result["fps_claimed"] is False


def test_remeasure_returns_latency(tiny_path, tmp_path) -> None:
    payload = {
        "kind": "fixture",
        "chosen": {
            "origin": "preset:baseline",
            "plan_id": "baseline",
            "p50_ms": 1.0,
            "passed": True,
            "artifact": {"path": str(tiny_path)},
        },
    }
    (tmp_path / "optimize_fixture.json").write_text(json.dumps(payload), encoding="utf-8")
    result = _run("test", {"kind": "fixture", "warmup": 0, "iters": 2}, tmp_path)
    assert result["p50_ms"] > 0
    assert result["p95_ms"] > 0
    assert result["gate_passed"] is True
    assert result["fps_claimed"] is False


def test_job_sets_provider_and_model_without_a_key(tmp_path, monkeypatch) -> None:
    import os

    seen: dict[str, str | None] = {}

    def fake(_action, _params, _results):
        seen["llm"] = os.environ.get("NNC_LLM")
        seen["model"] = os.environ.get("NNC_LLM_MODEL")
        return {"fps_claimed": False}

    monkeypatch.setattr("compiler.ui.jobs._run", fake)
    runner = JobRunner(tmp_path)
    try:
        job = runner.submit("report", {"llm": "openai", "model_id": "gpt-4o-mini"})
        current = job
        for _ in range(40):
            current = runner.get(job["id"])
            if current["status"] in {"done", "error"}:
                break
            time.sleep(0.05)
        assert current["status"] == "done"
        assert seen["llm"] == "openai"
        assert seen["model"] == "gpt-4o-mini"
        blob = json.dumps(current)
        assert "sk-" not in blob
        assert "gsk_" not in blob
    finally:
        runner.close()


def test_page_names_graph_and_fusion() -> None:
    page = Path("compiler/ui/static/index.html").read_text(encoding="utf-8")
    assert "Prove fusion" in page
    assert "FusedConv" in page
    assert "fps_claimed" in page
    assert "key not set" in page
    assert "model id" in page
    assert "console is not reachable" in page


def test_rejects_non_loopback() -> None:
    assert require_loopback("localhost") == "127.0.0.1"
    with pytest.raises(ValueError, match="127.0.0.1"):
        require_loopback("0.0.0.0")


def test_second_job_is_rejected(tmp_path, monkeypatch) -> None:
    started = threading.Event()
    release = threading.Event()

    def slow(_action, _params, _results):
        started.set()
        assert release.wait(2)
        return {"fps_claimed": False}

    monkeypatch.setattr("compiler.ui.jobs._run", slow)
    runner = JobRunner(tmp_path)
    try:
        first = runner.submit("report", {})
        assert first["status"] in {"queued", "running"}
        assert started.wait(1)
        with pytest.raises(JobBusy):
            runner.submit("analyze", {})
        release.set()
        for _ in range(40):
            if runner.get(first["id"])["status"] == "done":
                break
            time.sleep(0.05)
        assert runner.get(first["id"])["status"] == "done"
    finally:
        release.set()
        runner.close()


def test_http_analyze_reaches_done(tiny_path, tmp_path) -> None:
    server, app = start_background("127.0.0.1", 0, tmp_path)
    port = server.server_address[1]
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as response:
            page = response.read().decode("utf-8")
        assert "NNC operator" in page
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/jobs",
            data=json.dumps(
                {"action": "analyze", "model": str(tiny_path), "kind": "fixture", "llm": "heuristic"}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            queued = json.loads(response.read().decode("utf-8"))
        assert queued["status"] in {"queued", "running", "done"}
        current = queued
        for _ in range(50):
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/jobs/{queued['id']}") as response:
                current = json.loads(response.read().decode("utf-8"))
            if current["status"] in {"done", "error"}:
                break
            time.sleep(0.05)
        assert current["status"] == "done"
        assert current["result"]["summary"]["node_count"] > 0
        assert current["result"]["fps_claimed"] is False
        blob = json.dumps(current)
        assert "gsk_" not in blob
        assert ".config/nnc" not in blob
    finally:
        server.shutdown()
        server.server_close()
        app.close()
