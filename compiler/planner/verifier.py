"""Verify an LLM plan against graph patterns, hardware, and the atom allowlist."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compiler.graph.graph_summary import GraphSummary
from compiler.hardware.profile import HardwareProfile
from compiler.planner.atoms import ATOMS, PASS_ATOMS
from compiler.planner.plan import Plan
from compiler.schema_validate import SchemaError, validate_llm_plan

_STRUCTURAL = {
    "eliminate_identity",
    "eliminate_dropout",
    "constant_folding",
    "fuse_bn_into_conv",
    "fuse_conv_relu",
    "fuse_matmul_add_gemm",
}
_QUANT = {"quantize_dynamic_int8", "quantize_static_int8", "convert_fp16"}


@dataclass(frozen=True, slots=True)
class VerifiedPlan:
    plan: Plan
    accepted: bool
    rejections: tuple[dict[str, str], ...] = ()
    numerics: str = "exact"


def verify_plan(
    plan: Plan,
    summary: GraphSummary,
    hardware: HardwareProfile,
    backend_name: str = "ort_cpu",
    constraints: dict[str, Any] | None = None,
) -> VerifiedPlan:
    constraints = constraints or {}
    rejections: list[dict[str, str]] = []
    try:
        validate_llm_plan(plan.to_dict())
    except SchemaError as exc:
        return VerifiedPlan(plan=plan, accepted=False, rejections=({"step": "*", "reason": str(exc), "level": "reject"},), numerics="exact")

    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    numerics = "exact"
    patterns = summary.patterns or {}
    cpu_count = max(1, int(hardware.cpu_count))

    for step in plan.steps:
        atom_name = str(step.get("atom", ""))
        atom = ATOMS.get(atom_name)
        if atom is None or atom.kind != "pass":
            rejections.append({"step": atom_name, "reason": "unknown atom", "level": "reject"})
            return VerifiedPlan(plan=plan, accepted=False, rejections=tuple(rejections), numerics="exact")
        if atom_name in seen:
            rejections.append({"step": atom_name, "reason": "duplicate atom dropped", "level": "drop"})
            continue
        skip = False
        for key in atom.requires:
            if int(patterns.get(key, 0) or 0) <= 0:
                rejections.append({"step": atom_name, "reason": f"pattern {key} absent", "level": "drop"})
                skip = True
                break
        if skip:
            continue
        for flag in atom.hardware_requires:
            hw_ok = bool(getattr(hardware, flag, False))
            if flag == "fp16_execution" and backend_name == "ort_cpu":
                hw_ok = False
            if not hw_ok:
                rejections.append({"step": atom_name, "reason": f"hardware {flag} unavailable", "level": "drop"})
                skip = True
                break
        if skip:
            continue
        seen.add(atom_name)
        if atom.numerics == "approx":
            numerics = "approx"
        kept.append({"atom": atom_name, "params": dict(step.get("params") or {})})

    names = [step["atom"] for step in kept]
    if any(name in _STRUCTURAL for name in names) and "onnx_shape_infer" not in names:
        kept.insert(0, {"atom": "onnx_shape_infer", "params": {}})
        names = [step["atom"] for step in kept]
        rejections.append({"step": "onnx_shape_infer", "reason": "forced first for structural passes", "level": "drop"})

    rest = [step for step in kept if step["atom"] not in _QUANT and step["atom"] != "onnx_shape_infer"]
    quants = [step for step in kept if step["atom"] in _QUANT]
    infer = [step for step in kept if step["atom"] == "onnx_shape_infer"]
    kept = infer + rest + quants

    options = dict(plan.options)
    threads = options.get("intra_op_threads")
    if threads is not None:
        try:
            value = int(threads)
        except (TypeError, ValueError):
            value = cpu_count
        if value > cpu_count:
            rejections.append({"step": "intra_op_threads", "reason": f"clamped {value} -> {cpu_count}", "level": "drop"})
            value = cpu_count
        if value < 1:
            value = 1
        options["intra_op_threads"] = value
    options.setdefault("ort_graph_opt", "disable")
    options.setdefault("execution_mode", "sequential")

    verified = Plan(
        plan_id=plan.plan_id,
        steps=tuple(kept),
        options=options,
        rationale=plan.rationale,
        expected_effects=plan.expected_effects,
        confidence=plan.confidence,
        source=plan.source,
    )
    return VerifiedPlan(plan=verified, accepted=True, rejections=tuple(rejections), numerics=numerics)
