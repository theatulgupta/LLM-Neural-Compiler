from __future__ import annotations

import pytest

from compiler.errors import UnknownStrategyError
from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph
from compiler.llm.llm_client import MockLlmClient, proposal_from_dict
from compiler.llm.recommendation_engine import recommend_strategy
from compiler.planner import ALLOWED_STRATEGY_NAMES, get_strategy
from compiler.schema import LLM_PROPOSAL_SCHEMA_PATH, SchemaError, load_schema, validate_llm_proposal


def test_schema_enum_matches_allowlist() -> None:
    schema = load_schema(LLM_PROPOSAL_SCHEMA_PATH)
    assert tuple(schema["properties"]["strategy"]["enum"]) == ALLOWED_STRATEGY_NAMES


def test_mock_client_is_schema_bound(tiny_path) -> None:
    summary = summarize_graph(load_graph(tiny_path))
    rec = recommend_strategy(summary, client=MockLlmClient())
    assert rec.source == "mock"
    assert rec.strategy.name == "baseline"
    validate_llm_proposal(rec.proposal.to_dict())


def test_mock_client_rejects_unknown_strategy(tiny_path) -> None:
    summary = summarize_graph(load_graph(tiny_path))
    client = MockLlmClient({"strategy": "invent_new_op", "rationale": "should fail", "source": "mock"})
    with pytest.raises(SchemaError):
        client.propose(summary)


def test_proposal_from_dict_rejects_empty_rationale() -> None:
    with pytest.raises(SchemaError):
        proposal_from_dict({"strategy": "baseline", "rationale": "  ", "source": "mock"})


def test_unknown_strategy_still_raises() -> None:
    with pytest.raises(UnknownStrategyError):
        get_strategy("not_a_strategy")


def test_heuristic_recommendation_is_allowlisted(tiny_path) -> None:
    summary = summarize_graph(load_graph(tiny_path))
    rec = recommend_strategy(summary)
    assert rec.strategy.name in ALLOWED_STRATEGY_NAMES
    assert rec.source == "heuristic"
