"""Allowlisted compile strategies. LLM output is rejected if it leaves this set.

A strategy is a named **graph pass set** plus an ORT session graph-opt level.
The LLM may only pick the name. The transformation engine applies the passes.
"""

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
        description="No graph rewrites. ORT graph optimizations disabled (native DAG).",
        ort_graph_opt="disable",
        passes=(),
    ),
    "graph_simplify": Strategy(
        name="graph_simplify",
        description="Shape infer, drop Identity/nop Dropout, constant-fold. ORT opts off.",
        ort_graph_opt="disable",
        passes=("onnx_shape_infer", "eliminate_identity", "constant_folding"),
    ),
    "graph_fuse": Strategy(
        name="graph_fuse",
        description="Simplify plus Conv-BN and Conv-ReLU fusion on the ONNX DAG. ORT opts off.",
        ort_graph_opt="disable",
        passes=(
            "onnx_shape_infer",
            "eliminate_identity",
            "constant_folding",
            "fuse_bn_into_conv",
            "fuse_conv_relu",
        ),
    ),
    "graph_fuse_ort": Strategy(
        name="graph_fuse_ort",
        description="Same DAG fusions as graph_fuse, then ORT extended (ablation vs our passes).",
        ort_graph_opt="extended",
        passes=(
            "onnx_shape_infer",
            "eliminate_identity",
            "constant_folding",
            "fuse_bn_into_conv",
            "fuse_conv_relu",
        ),
    ),
}

ALLOWED_STRATEGY_NAMES: tuple[str, ...] = tuple(ALLOWED_STRATEGIES)


def get_strategy(name: str) -> Strategy:
    from compiler.errors import UnknownStrategyError

    if name not in ALLOWED_STRATEGIES:
        raise UnknownStrategyError(name, ALLOWED_STRATEGY_NAMES)
    return ALLOWED_STRATEGIES[name]
