"""Compact graph summary consumed by the recommendation engine (and tests)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from compiler.graph.graph_analyzer import analyze_graph
from compiler.graph.graph_loader import LoadedGraph


@dataclass(frozen=True, slots=True)
class GraphSummary:
    source: str
    sha256: str
    opset: int
    ir_version: int
    producer: str
    node_count: int
    op_counts: dict[str, int]
    param_count: int
    initializer_bytes: int
    estimated_flops: int | None
    flops_total: int
    flops_by_op: dict[str, int]
    top_nodes: list[dict[str, Any]]
    patterns: dict[str, int]
    memory: dict[str, Any]
    static_shapes: bool
    inputs: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_graph(loaded: LoadedGraph) -> GraphSummary:
    stats = analyze_graph(loaded)
    notes = _notes(stats)
    return GraphSummary(
        source=loaded.source,
        sha256=loaded.sha256,
        opset=int(stats["opset"]),
        ir_version=int(stats["ir_version"]),
        producer=str(stats["producer"]),
        node_count=int(stats["node_count"]),
        op_counts=dict(stats["op_counts"]),  # type: ignore[arg-type]
        param_count=int(stats["param_count"]),
        initializer_bytes=int(stats["initializer_bytes"]),
        estimated_flops=stats["estimated_flops"],  # type: ignore[arg-type]
        flops_total=int(stats["flops_total"]),
        flops_by_op=dict(stats["flops_by_op"]),  # type: ignore[arg-type]
        top_nodes=list(stats["top_nodes"]),  # type: ignore[arg-type]
        patterns=dict(stats["patterns"]),  # type: ignore[arg-type]
        memory=dict(stats["memory"]),  # type: ignore[arg-type]
        static_shapes=bool(stats["static_shapes"]),
        inputs=list(stats["inputs"]),  # type: ignore[arg-type]
        outputs=list(stats["outputs"]),  # type: ignore[arg-type]
        notes=notes,
    )


def _notes(stats: dict[str, object]) -> tuple[str, ...]:
    op_counts = stats["op_counts"]
    patterns = stats["patterns"]
    assert isinstance(op_counts, dict)
    assert isinstance(patterns, dict)
    notes: list[str] = []
    if op_counts.get("Flatten") and op_counts.get("Gemm"):
        notes.append("classifier head is Flatten+Gemm; Gemm K must match Flatten width")
    if int(patterns.get("conv_bn", 0)) == 0:
        notes.append("conv without BatchNormalization; fuse_bn_into_conv not applicable")
    if int(patterns.get("conv_silu", 0)) > 0:
        notes.append("no fused CPU SiLU kernel; quantization/threads are the levers")
    if int(stats["param_count"]) < 10_000:  # type: ignore[arg-type]
        notes.append("tiny parameter count; CPU latency is dominated by overhead")
    return tuple(notes)
