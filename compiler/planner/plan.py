"""Named optimization plans. Presets keep the old --strategy CLI working."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from compiler.planner.atoms import PASS_ATOMS


@dataclass(frozen=True, slots=True)
class Plan:
    plan_id: str
    steps: tuple[dict[str, Any], ...]
    options: dict[str, Any]
    rationale: str
    expected_effects: tuple[str, ...] = ()
    confidence: float = 1.0
    source: str = "preset"

    @property
    def name(self) -> str:
        return self.plan_id

    @property
    def description(self) -> str:
        return self.rationale

    @property
    def passes(self) -> tuple[str, ...]:
        return tuple(str(step["atom"]) for step in self.steps if str(step.get("atom")) in PASS_ATOMS)

    @property
    def ort_graph_opt(self) -> str:
        return str(self.options.get("ort_graph_opt", "disable"))

    @property
    def intra_op_threads(self) -> int | None:
        value = self.options.get("intra_op_threads")
        return int(value) if value is not None else None

    @property
    def execution_mode(self) -> str:
        return str(self.options.get("execution_mode", "sequential"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "steps": [dict(step) for step in self.steps],
            "options": dict(self.options),
            "rationale": self.rationale,
            "expected_effects": list(self.expected_effects),
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Plan:
        steps = tuple(
            dict(step) if isinstance(step, dict) else {"atom": str(step)}
            for step in payload.get("steps") or ()
        )
        options = dict(payload.get("options") or {})
        effects = payload.get("expected_effects") or ()
        return cls(
            plan_id=str(payload.get("plan_id") or "unnamed"),
            steps=steps,
            options=options,
            rationale=str(payload.get("rationale") or "unspecified"),
            expected_effects=tuple(str(item) for item in effects),
            confidence=float(payload.get("confidence") or 0.0),
            source=str(payload.get("source") or "unknown"),
        )


def _steps(*names: str) -> tuple[dict[str, Any], ...]:
    return tuple({"atom": name, "params": {}} for name in names)


PRESETS: dict[str, Plan] = {
    "baseline": Plan(
        plan_id="baseline",
        steps=(),
        options={"ort_graph_opt": "disable", "execution_mode": "sequential"},
        rationale="No graph rewrites. ORT graph optimizations disabled (native DAG).",
        expected_effects=("none",),
        source="preset",
    ),
    "ort_default": Plan(
        plan_id="ort_default",
        steps=(),
        options={"ort_graph_opt": "all", "execution_mode": "sequential"},
        rationale="Unrewritten ONNX with ORT graph optimizations enabled.",
        expected_effects=("lower_latency",),
        source="preset",
    ),
    "graph_simplify": Plan(
        plan_id="graph_simplify",
        steps=_steps("onnx_shape_infer", "eliminate_identity", "eliminate_dropout", "constant_folding"),
        options={"ort_graph_opt": "disable", "execution_mode": "sequential"},
        rationale="Shape infer, drop Identity/nop Dropout, constant-fold.",
        expected_effects=("fewer_nodes",),
        source="preset",
    ),
    "graph_fuse": Plan(
        plan_id="graph_fuse",
        steps=_steps(
            "onnx_shape_infer",
            "eliminate_identity",
            "constant_folding",
            "fuse_bn_into_conv",
            "fuse_conv_relu",
        ),
        options={"ort_graph_opt": "disable", "execution_mode": "sequential"},
        rationale="Simplify plus Conv-BN and Conv-ReLU fusion on the ONNX DAG.",
        expected_effects=("fewer_nodes", "lower_latency"),
        source="preset",
    ),
    "graph_fuse_ort": Plan(
        plan_id="graph_fuse_ort",
        steps=_steps(
            "onnx_shape_infer",
            "eliminate_identity",
            "constant_folding",
            "fuse_bn_into_conv",
            "fuse_conv_relu",
        ),
        options={"ort_graph_opt": "extended", "execution_mode": "sequential"},
        rationale="Same DAG fusions as graph_fuse, then ORT extended.",
        expected_effects=("fewer_nodes", "lower_latency"),
        source="preset",
    ),
    "int8_dynamic": Plan(
        plan_id="int8_dynamic",
        steps=_steps("onnx_shape_infer", "quantize_dynamic_int8"),
        options={"ort_graph_opt": "basic", "execution_mode": "sequential"},
        rationale="Dynamic INT8 weights for CPU (asimddp) after shape infer.",
        expected_effects=("smaller_model", "lower_latency", "lower_memory"),
        source="preset",
        confidence=0.6,
    ),
    "int8_static": Plan(
        plan_id="int8_static",
        steps=(
            {"atom": "onnx_shape_infer", "params": {}},
            {
                "atom": "quantize_static_int8",
                "params": {"calibration": "gz_frames", "per_channel": True, "format": "qdq"},
            },
        ),
        options={"ort_graph_opt": "basic", "execution_mode": "sequential"},
        rationale="Static INT8 with calibration frames.",
        expected_effects=("smaller_model", "lower_latency"),
        source="preset",
        confidence=0.5,
    ),
}

ALLOWED_STRATEGY_NAMES: tuple[str, ...] = tuple(PRESETS)


def get_plan(name: str) -> Plan:
    from compiler.errors import UnknownStrategyError

    if name not in PRESETS:
        raise UnknownStrategyError(name, ALLOWED_STRATEGY_NAMES)
    return PRESETS[name]


# Facade for the old Strategy name.
Strategy = Plan
ALLOWED_STRATEGIES = PRESETS
get_strategy = get_plan
