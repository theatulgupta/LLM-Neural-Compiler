"""Bounded LLM correction loop: retries, fallback, transport, ranking."""

from __future__ import annotations

import json
import urllib.error

from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph
from compiler.hardware.profile import probe_hardware
from compiler.llm.groq_client import GroqLlmClient, groq_chat
from compiler.llm.llm_client import MockLlmClient
from compiler.llm.session import propose_plan
from compiler.pipeline.optimize import optimize_model
from compiler.planner.plan import PRESETS
from compiler.schema import SchemaError


def test_propose_plan_retries_garbage_then_unknown_atom(tiny_path) -> None:
    replies = [
        "not json at all",
        json.dumps(
            {
                "plan_id": "bad",
                "steps": [{"atom": "magic_int4"}],
                "options": {},
                "rationale": "invented atom",
                "confidence": 0.4,
                "source": "groq",
            }
        ),
        json.dumps({"strategy": "graph_fuse", "rationale": "conv graph", "source": "groq"}),
    ]
    seen: list[list[dict[str, str]]] = []

    def fake_chat(messages, model):
        seen.append(messages)
        return replies[len(seen) - 1]

    summary = summarize_graph(load_graph(tiny_path))
    client = GroqLlmClient(chat=fake_chat, api_key="unused")
    outcome = propose_plan(summary, probe_hardware(), {}, [], client, max_attempts=3)
    assert outcome.accepted is True
    assert outcome.source == "groq"
    assert len(outcome.attempts) == 3
    assert outcome.attempts[0].outcome == "schema_error"
    assert outcome.attempts[1].outcome == "schema_error"
    assert outcome.attempts[2].outcome == "accepted"
    assert "json" in seen[1][1]["content"].lower() or "attempt 1" in seen[1][1]["content"]
    assert "magic_int4" in seen[2][1]["content"] or "not allowlisted" in seen[2][1]["content"].lower()
    assert "Reply again with a corrected plan" in seen[1][1]["content"]
    assert "Reply again with a corrected plan" in seen[2][1]["content"]


def test_all_bad_replies_fall_back_and_optimize_still_chooses(tiny_path, tmp_path) -> None:
    def fake_chat(messages, model):
        return "definitely not a plan"

    client = GroqLlmClient(chat=fake_chat, api_key="unused")
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
    assert payload.get("chosen") is not None
    llm = payload["llm"]
    assert llm["source"] == "heuristic"
    assert llm["fallback_reason"]
    assert llm["attempts"]
    assert all(item["outcome"] == "schema_error" for item in llm["attempts"])


def test_groq_chat_retries_urlerror(tiny_path) -> None:
    calls = {"n": 0}
    sleeps: list[float] = []
    content = json.dumps({"strategy": "baseline", "rationale": "ok", "source": "groq"})
    body = json.dumps({"choices": [{"message": {"content": content}}]})

    class _Resp:
        def read(self) -> bytes:
            return body.encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

    def fake_urlopen(request, timeout=30.0):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("timeout")
        return _Resp()

    text = groq_chat(
        [{"role": "user", "content": "hi"}],
        "openai/gpt-oss-20b",
        api_key="unused",
        sleep=sleeps.append,
        urlopen=fake_urlopen,
    )
    assert calls["n"] == 3
    assert sleeps == [1.0, 2.0]
    assert "baseline" in text


def test_optimize_keeps_llm_row_when_plan_matches_preset(tiny_path, tmp_path) -> None:
    client = MockLlmClient(PRESETS["graph_fuse"].to_dict())
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
    llm_rows = [row for row in payload["candidates"] if row["origin"] == "llm"]
    assert len(llm_rows) == 1
    assert llm_rows[0]["same_as"] == "preset:graph_fuse"
    assert llm_rows[0]["measured"] is False
    assert isinstance(payload["llm_rank"], int)
    assert payload["llm"]["same_as"] == "preset:graph_fuse"
    assert payload.get("heuristic_rank") is None or isinstance(payload["heuristic_rank"], int)


def test_groq_http_400_does_not_retry() -> None:
    class _Err(urllib.error.HTTPError):
        def __init__(self) -> None:
            super().__init__(url="http://x", code=400, msg="bad", hdrs=None, fp=None)

        def read(self) -> bytes:
            return b"nope"

    def fake_urlopen(request, timeout=30.0):
        raise _Err()

    try:
        groq_chat(
            [{"role": "user", "content": "hi"}],
            "openai/gpt-oss-20b",
            api_key="unused",
            sleep=lambda _s: None,
            urlopen=fake_urlopen,
        )
    except SchemaError as exc:
        assert "400" in str(exc)
        return
    raise AssertionError("expected SchemaError")
