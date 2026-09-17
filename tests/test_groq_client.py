from __future__ import annotations

import json

import pytest

from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph
from compiler.llm.groq_client import GroqLlmClient, extract_json_object, strip_invented_metrics
from compiler.llm.recommendation_engine import recommend_strategy
from compiler.schema import SchemaError


def test_extract_json_object_from_fences() -> None:
    payload = extract_json_object(
        'noise\n```json\n{"strategy": "graph_fuse", "rationale": "conv graph", "source": "groq"}\n```\n'
    )
    assert payload["strategy"] == "graph_fuse"


def test_extract_json_rejects_non_object() -> None:
    with pytest.raises(SchemaError):
        extract_json_object("[1, 2]")


def test_strip_invented_metrics() -> None:
    cleaned, stripped = strip_invented_metrics("use fused conv; 120 fps on Jetson")
    assert stripped is True
    assert "fps" not in cleaned.lower()


def test_groq_client_uses_injected_chat(tiny_path) -> None:
    def fake_chat(messages, model):
        assert messages[0]["role"] == "system"
        assert "allowlist" in messages[1]["content"].lower() or "Allowlisted" in messages[1]["content"]
        return json.dumps(
            {"strategy": "graph_fuse", "rationale": "many convs; no numbers", "source": "ignored"}
        )

    client = GroqLlmClient(chat=fake_chat, api_key="unused")
    summary = summarize_graph(load_graph(tiny_path))
    rec = recommend_strategy(summary, client=client)
    assert rec.source == "groq"
    assert rec.strategy.name == "graph_fuse"


def test_groq_client_rejects_unknown_strategy(tiny_path) -> None:
    def fake_chat(messages, model):
        return '{"strategy": "magic_int4", "rationale": "nope", "source": "groq"}'

    client = GroqLlmClient(chat=fake_chat, api_key="unused")
    summary = summarize_graph(load_graph(tiny_path))
    with pytest.raises(SchemaError):
        client.propose(summary)
