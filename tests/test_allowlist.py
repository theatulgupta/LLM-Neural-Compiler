from __future__ import annotations

from compiler.errors import UnknownStrategyError
from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph
from compiler.llm.recommendation_engine import recommend_strategy
from compiler.optimization.passes import apply_pass
from compiler.strategies import ALLOWED_STRATEGY_NAMES, get_strategy


def test_unknown_strategy_is_rejected() -> None:
    try:
        get_strategy("invent_new_op")
    except UnknownStrategyError as exc:
        assert exc.name == "invent_new_op"
        assert "baseline" in exc.allowed
    else:
        raise AssertionError("expected UnknownStrategyError")


def test_heuristic_recommendation_is_allowlisted(tiny_path) -> None:
    summary = summarize_graph(load_graph(tiny_path))
    rec = recommend_strategy(summary)
    assert rec.strategy.name in ALLOWED_STRATEGY_NAMES
    assert rec.source == "heuristic"


def test_unknown_pass_is_rejected(tiny_path) -> None:
    model = load_graph(tiny_path).model
    try:
        apply_pass("quantize_int4_magic", model)
    except ValueError as exc:
        assert "quantize_int4_magic" in str(exc)
    else:
        raise AssertionError("expected ValueError")
