"""LLM clients return a plan dict. Heuristic works offline."""

from __future__ import annotations

import os
from typing import Any, Protocol

from compiler.graph.graph_summary import GraphSummary
from compiler.hardware.profile import HardwareProfile, probe_hardware
from compiler.llm.context_builder import build_context
from compiler.planner.plan import Plan
from compiler.schema_validate import SchemaError, validate_llm_plan


class LlmProposal:
    """Back-compat wrapper: strategy name + rationale."""

    def __init__(self, strategy: str, rationale: str, source: str, plan: dict[str, Any] | None = None) -> None:
        self.strategy = strategy
        self.rationale = rationale
        self.source = source
        self.plan = plan or {
            "plan_id": strategy,
            "steps": [],
            "options": {"ort_graph_opt": "disable"},
            "rationale": rationale,
            "confidence": 0.5,
            "source": source,
        }

    def to_dict(self) -> dict[str, str]:
        return {"strategy": self.strategy, "rationale": self.rationale, "source": self.source}


class LlmClient(Protocol):
    name: str

    def propose(self, summary: GraphSummary, context: dict[str, Any] | None = None) -> LlmProposal:
        ...


def proposal_from_dict(payload: dict[str, object]) -> LlmProposal:
    if "plan_id" in payload or "steps" in payload:
        plan = validate_llm_plan(dict(payload))
        return LlmProposal(
            strategy=str(plan["plan_id"]),
            rationale=str(plan["rationale"]),
            source=str(plan["source"]),
            plan=plan,
        )
    strategy = str(payload.get("strategy", ""))
    rationale = str(payload.get("rationale", ""))
    source = str(payload.get("source", "unknown"))
    if not strategy or not rationale.strip():
        raise SchemaError("proposal missing strategy or rationale")
    from compiler.planner.plan import PRESETS

    if strategy not in PRESETS:
        raise SchemaError(f"strategy {strategy!r} is not allowlisted")
    base = PRESETS[strategy]
    preset = Plan(
        plan_id=strategy,
        steps=base.steps,
        options=dict(base.options),
        rationale=rationale,
        expected_effects=base.expected_effects,
        confidence=base.confidence,
        source=source,
    )
    return LlmProposal(strategy=strategy, rationale=rationale, source=source, plan=preset.to_dict())


def _context(summary: GraphSummary, context: dict[str, Any] | None) -> dict[str, Any]:
    if context is not None:
        return context
    return build_context(summary, probe_hardware(), {}, [])


class HeuristicLlmClient:
    """No network. Maps graph features onto allowlisted atoms."""

    name = "heuristic"

    def propose(self, summary: GraphSummary, context: dict[str, Any] | None = None) -> LlmProposal:
        ctx = _context(summary, context)
        hardware = ctx.get("hardware") or {}
        features = set(hardware.get("features") or [])
        cpu_count = int(hardware.get("cpu_count") or 1)
        patterns = summary.patterns or {}
        steps: list[dict[str, Any]] = [{"atom": "onnx_shape_infer", "params": {}}]
        if int(patterns.get("identity_nodes", 0) or 0) > 0:
            steps.append({"atom": "eliminate_identity", "params": {}})
        if int(patterns.get("dropout_nodes", 0) or 0) > 0:
            steps.append({"atom": "eliminate_dropout", "params": {}})
        steps.append({"atom": "constant_folding", "params": {}})
        if int(patterns.get("conv_bn", 0) or 0) > 0:
            steps.append({"atom": "fuse_bn_into_conv", "params": {}})
        if int(patterns.get("conv_relu", 0) or 0) > 0:
            steps.append({"atom": "fuse_conv_relu", "params": {}})
        if int(patterns.get("matmul_add", 0) or 0) > 0:
            steps.append({"atom": "fuse_matmul_add_gemm", "params": {}})
        if int(summary.flops_total or 0) > 1_000_000_000 and "asimddp" in features:
            steps.append({"atom": "quantize_dynamic_int8", "params": {"per_channel": True, "weight_type": "qint8"}})
        if summary.node_count <= 8:
            payload = {
                "plan_id": "baseline",
                "steps": [],
                "options": {"ort_graph_opt": "disable", "execution_mode": "sequential"},
                "rationale": "Tiny graph; keep the native DAG so the Flatten/Gemm contract is visible.",
                "expected_effects": ["none"],
                "confidence": 0.9,
                "source": self.name,
            }
        elif int(summary.op_counts.get("Conv", 0) or 0) >= 1:
            payload = {
                "plan_id": "graph_fuse",
                "steps": steps,
                "options": {
                    "ort_graph_opt": "extended",
                    "intra_op_threads": min(4, cpu_count),
                    "execution_mode": "sequential",
                },
                "rationale": "Convolutional DAG: fuse Conv-BN and Conv-ReLU when present; fold constants.",
                "expected_effects": ["fewer_nodes", "lower_latency"],
                "confidence": 0.7,
                "source": self.name,
            }
        else:
            payload = {
                "plan_id": "graph_simplify",
                "steps": [s for s in steps if s["atom"] in {"onnx_shape_infer", "eliminate_identity", "eliminate_dropout", "constant_folding"}],
                "options": {"ort_graph_opt": "disable", "execution_mode": "sequential"},
                "rationale": "Generic DAG: shape infer, dead-node elim, constant fold; no extra fusion.",
                "expected_effects": ["fewer_nodes"],
                "confidence": 0.6,
                "source": self.name,
            }
        return proposal_from_dict(payload)


class MockLlmClient:
    name = "mock"

    def __init__(self, proposal: dict[str, object] | None = None) -> None:
        self._proposal = proposal or {
            "plan_id": "baseline",
            "steps": [],
            "options": {"ort_graph_opt": "disable"},
            "rationale": "mock",
            "confidence": 1.0,
            "source": "mock",
        }

    def propose(self, summary: GraphSummary, context: dict[str, Any] | None = None) -> LlmProposal:
        return proposal_from_dict(dict(self._proposal))


def build_client() -> LlmClient:
    mode = os.environ.get("NNC_LLM", "").strip().lower()
    if mode == "mock":
        return MockLlmClient()
    if mode == "heuristic":
        return HeuristicLlmClient()
    if mode == "groq":
        from compiler.llm.groq_client import GroqLlmClient

        return GroqLlmClient()
    from compiler.llm.groq_client import load_groq_api_key

    if load_groq_api_key() and mode != "heuristic":
        if mode == "groq" or not mode:
            # Default stays heuristic unless NNC_LLM=groq so offline tests are stable.
            return HeuristicLlmClient()
    return HeuristicLlmClient()
