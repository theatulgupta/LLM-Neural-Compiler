"""HTTP follow-up: llm_improved / llm_revised, bounded, provider-agnostic."""

from __future__ import annotations

import json
from typing import Any

from compiler.llm.llm_client import MockLlmClient
from compiler.llm.provider import HttpLlmClient
from compiler.pipeline.optimize import _followup_kind, optimize_model
from compiler.planner.plan import PRESETS


def _dumps(plan_id: str, steps: list, options: dict, rationale: str = "ok") -> str:
    return json.dumps(
        {
            "plan_id": plan_id,
            "steps": steps,
            "options": options,
            "rationale": rationale,
            "confidence": 0.5,
        }
    )


def _record(*, p50: float | None, passed: bool = True, compile_ok: bool = True, error: str | None = None) -> dict[str, Any]:
    return {
        "run_id": "t",
        "compile": {"ok": compile_ok, "ms": 1.0, "error": None if compile_ok else (error or "boom")},
        "benchmark": {"latency_ms": {"p50": p50}, "throughput_ips": 1.0},
        "verification": {"passed": passed, "error": error, "inputs_source": "assets"},
        "profile": {"model_bytes": 10},
        "graph_changed": False,
        "artifact": None,
        "graph_before": {"node_count": 5},
        "graph_after": {"node_count": 5},
    }


LLM_FIRST = _dumps(
    "llm_first",
    [],
    {"ort_graph_opt": "disable", "execution_mode": "sequential", "intra_op_threads": 1},
)
IMPROVED = _dumps(
    "uav_try_threads",
    [dict(step) for step in PRESETS["graph_fuse"].steps],
    {"ort_graph_opt": "disable", "execution_mode": "sequential", "intra_op_threads": 2},
)
GRAPH_FUSE = _dumps(
    "graph_fuse",
    [dict(step) for step in PRESETS["graph_fuse"].steps],
    dict(PRESETS["graph_fuse"].options),
)


def _chat(replies: list[str], seen: list) -> Any:
    def fake(messages, model):
        seen.append(messages)
        return replies[min(len(seen) - 1, len(replies) - 1)]

    return fake


def _p50_cvp(table: dict[str, float], *, failed: str | None = None):
    def fake(model_path, *, plan, **kwargs):
        if failed and plan.plan_id == failed:
            return _record(p50=3.0, passed=False, compile_ok=True, error="cosine")
        return _record(p50=table.get(plan.plan_id, 15.0))

    return fake


def test_followup_kind_predicates() -> None:
    best = {"p50_ms": 5.0, "origin": "preset:graph_fuse"}
    ranked = [best]
    slow = {"compile_ok": True, "passed": True, "p50_ms": 9.0}
    assert _followup_kind(source="heuristic", llm_row=slow, ranked=ranked) is None
    assert _followup_kind(source="mock", llm_row=slow, ranked=ranked) is None
    assert _followup_kind(source=None, llm_row=slow, ranked=ranked) is None
    assert (
        _followup_kind(
            source="openai",
            llm_row={"compile_ok": False, "passed": False, "p50_ms": 3.0},
            ranked=ranked,
        )
        == "revised"
    )
    assert _followup_kind(source="openai", llm_row=slow, ranked=ranked) == "improved"
    assert (
        _followup_kind(
            source="openai",
            llm_row={"compile_ok": True, "passed": True, "p50_ms": 5.0},
            ranked=ranked,
        )
        is None
    )
    assert (
        _followup_kind(
            source="openai",
            llm_row={"compile_ok": True, "passed": True, "p50_ms": 4.0},
            ranked=ranked,
        )
        is None
    )


def test_http_improved_unique_slower(monkeypatch, tiny_path, tmp_path) -> None:
    seen: list = []
    monkeypatch.setattr(
        "compiler.pipeline.optimize.compile_verify_profile",
        _p50_cvp(
            {
                "baseline": 10.0,
                "ort_default": 11.0,
                "graph_fuse": 8.0,
                "llm_first": 20.0,
                "uav_try_threads": 5.0,
            }
        ),
    )
    client = HttpLlmClient(provider="openai", chat=_chat([LLM_FIRST, IMPROVED], seen), api_key="unused")
    payload = optimize_model(
        tiny_path,
        kind="fixture",
        task="classify",
        mode="default",
        warmup=1,
        iters=2,
        results_dir=tmp_path,
        client=client,
    )
    assert len(seen) == 2
    second = seen[1][1]["content"]
    assert "measured_not_best" in second
    assert "measured_table" in second
    assert payload["llm"]["source"] == "openai"
    assert payload["fps_claimed"] is False
    improved = [row for row in payload["candidates"] if row["origin"] == "llm_improved"]
    assert len(improved) == 1
    assert improved[0]["plan_id"] == "uav_try_threads"
    assert payload["chosen"]["origin"] == "llm_improved"
    assert payload["llm"]["improved"]["plan_id"] == "uav_try_threads"
    assert payload["llm"]["revised"] is None


def test_http_tie_skips_improved(monkeypatch, tiny_path, tmp_path) -> None:
    seen: list = []
    monkeypatch.setattr(
        "compiler.pipeline.optimize.compile_verify_profile",
        _p50_cvp({"baseline": 10.0, "ort_default": 11.0, "graph_fuse": 8.0}),
    )
    client = HttpLlmClient(provider="openai", chat=_chat([GRAPH_FUSE], seen), api_key="unused")
    payload = optimize_model(
        tiny_path,
        kind="fixture",
        task="classify",
        mode="default",
        warmup=1,
        iters=2,
        results_dir=tmp_path,
        client=client,
    )
    assert len(seen) == 1
    assert payload["llm"]["improved"] is None
    assert payload["llm"]["revised"] is None
    assert payload["llm"]["same_as"] == "preset:graph_fuse"


def test_http_failed_gates_revised(monkeypatch, tiny_path, tmp_path) -> None:
    seen: list = []
    monkeypatch.setattr(
        "compiler.pipeline.optimize.compile_verify_profile",
        _p50_cvp(
            {"baseline": 10.0, "ort_default": 11.0, "graph_fuse": 8.0, "llm_first": 3.0, "uav_try_threads": 7.0},
            failed="llm_first",
        ),
    )
    client = HttpLlmClient(provider="openai", chat=_chat([LLM_FIRST, IMPROVED], seen), api_key="unused")
    payload = optimize_model(
        tiny_path,
        kind="fixture",
        task="classify",
        mode="default",
        warmup=1,
        iters=2,
        results_dir=tmp_path,
        client=client,
    )
    assert len(seen) == 2
    assert "measured_failure" in seen[1][1]["content"]
    assert payload["llm"]["revised"]["plan_id"] == "uav_try_threads"
    assert payload["llm"]["improved"] is None
    assert any(row["origin"] == "llm_revised" for row in payload["candidates"])


def test_http_followup_garbage_skips_compile(monkeypatch, tiny_path, tmp_path) -> None:
    seen: list = []
    monkeypatch.setattr(
        "compiler.pipeline.optimize.compile_verify_profile",
        _p50_cvp({"baseline": 10.0, "ort_default": 11.0, "graph_fuse": 8.0, "llm_first": 20.0}),
    )
    client = HttpLlmClient(provider="openai", chat=_chat([LLM_FIRST, "not a plan"], seen), api_key="unused")
    payload = optimize_model(
        tiny_path,
        kind="fixture",
        task="classify",
        mode="default",
        warmup=1,
        iters=2,
        results_dir=tmp_path,
        client=client,
    )
    assert len(seen) == 2
    assert str(payload["llm"]["improved"].get("skipped") or "").startswith("fallback")
    assert not any(row["origin"] == "llm_improved" for row in payload["candidates"])


def test_http_followup_same_key_records_same_as(monkeypatch, tiny_path, tmp_path) -> None:
    seen: list = []
    monkeypatch.setattr(
        "compiler.pipeline.optimize.compile_verify_profile",
        _p50_cvp({"baseline": 10.0, "ort_default": 11.0, "graph_fuse": 8.0, "llm_first": 20.0}),
    )
    client = HttpLlmClient(provider="openai", chat=_chat([LLM_FIRST, GRAPH_FUSE], seen), api_key="unused")
    payload = optimize_model(
        tiny_path,
        kind="fixture",
        task="classify",
        mode="default",
        warmup=1,
        iters=2,
        results_dir=tmp_path,
        client=client,
    )
    assert payload["llm"]["improved"]["skipped"] == "same as existing candidate"
    assert payload["llm"]["improved"]["same_as"] == "preset:graph_fuse"
    assert not any(row["origin"] == "llm_improved" for row in payload["candidates"])


def test_mock_skips_followup(monkeypatch, tiny_path, tmp_path) -> None:
    class Counting(MockLlmClient):
        def __init__(self) -> None:
            super().__init__({**PRESETS["graph_fuse"].to_dict(), "source": "mock"})
            self.n = 0

        def propose(self, summary, context=None, feedback=None):
            self.n += 1
            return super().propose(summary, context, feedback)

    monkeypatch.setattr(
        "compiler.pipeline.optimize.compile_verify_profile",
        _p50_cvp({"baseline": 10.0, "ort_default": 11.0, "graph_fuse": 8.0}),
    )
    client = Counting()
    payload = optimize_model(
        tiny_path,
        kind="fixture",
        task="classify",
        mode="default",
        warmup=1,
        iters=2,
        results_dir=tmp_path,
        client=client,
    )
    assert client.n == 1
    assert payload["llm"]["improved"] is None
    assert payload["llm"]["source"] == "mock"


def test_real_compile_bounded(tiny_path, tmp_path) -> None:
    seen: list = []
    client = HttpLlmClient(provider="openai", chat=_chat([LLM_FIRST, IMPROVED], seen), api_key="unused")
    payload = optimize_model(
        tiny_path,
        kind="fixture",
        task="classify",
        mode="default",
        warmup=1,
        iters=2,
        results_dir=tmp_path,
        client=client,
    )
    assert payload["chosen"] is not None
    assert payload["fps_claimed"] is False
    assert len(seen) <= 2
    origins = {row["origin"] for row in payload["candidates"]}
    assert not ({"llm_revised", "llm_improved"} <= origins)
