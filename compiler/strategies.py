"""Allowlisted compile strategies. LLM output is rejected if it leaves this set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OrtGraphOpt = Literal["disable", "basic", "extended", "all"]


@dataclass(frozen=True, slots=True)
class Strategy:
    name: str
    description: str
    ort_graph_opt: OrtGraphOpt
    passes: tuple[str, ...]


ALLOWED_STRATEGIES: dict[str, Strategy] = {
    "baseline": Strategy(
        name="baseline",
        description="ORT CPU with graph optimizations disabled (correctness baseline).",
        ort_graph_opt="disable",
        passes=(),
    ),
    "ort_basic": Strategy(
        name="ort_basic",
        description="ORT CPU with basic graph optimizations (fuse, constant folding).",
        ort_graph_opt="basic",
        passes=("onnx_shape_infer",),
    ),
    "ort_extended": Strategy(
        name="ort_extended",
        description="ORT CPU with extended graph optimizations.",
        ort_graph_opt="extended",
        passes=("onnx_shape_infer",),
    ),
    "ort_all": Strategy(
        name="ort_all",
        description="ORT CPU with all graph optimizations enabled.",
        ort_graph_opt="all",
        passes=("onnx_shape_infer",),
    ),
}

ALLOWED_STRATEGY_NAMES: tuple[str, ...] = tuple(ALLOWED_STRATEGIES)


def get_strategy(name: str) -> Strategy:
    from compiler.errors import UnknownStrategyError

    try:
        return ALLOWED_STRATEGIES[name]
    except KeyError as exc:
        raise UnknownStrategyError(name, ALLOWED_STRATEGY_NAMES) from exc
