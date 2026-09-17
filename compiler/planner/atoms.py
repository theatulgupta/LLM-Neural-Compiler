"""Allowlisted plan atoms. LLM may name these; the engine executes them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Atom:
    name: str
    kind: Literal["pass", "option"]
    params_schema: dict
    requires: tuple[str, ...] = ()
    hardware_requires: tuple[str, ...] = ()
    numerics: Literal["exact", "approx"] = "exact"


ATOMS: dict[str, Atom] = {
    "onnx_shape_infer": Atom("onnx_shape_infer", "pass", {}),
    "eliminate_identity": Atom("eliminate_identity", "pass", {}),
    "eliminate_dropout": Atom("eliminate_dropout", "pass", {}),
    "constant_folding": Atom("constant_folding", "pass", {}),
    "fuse_bn_into_conv": Atom("fuse_bn_into_conv", "pass", {}, requires=("conv_bn",)),
    "fuse_conv_relu": Atom("fuse_conv_relu", "pass", {}, requires=("conv_relu",)),
    "fuse_matmul_add_gemm": Atom("fuse_matmul_add_gemm", "pass", {}, requires=("matmul_add",)),
    "quantize_dynamic_int8": Atom(
        "quantize_dynamic_int8",
        "pass",
        {"per_channel": {"type": "boolean"}, "weight_type": {"enum": ["qint8", "quint8"]}},
        numerics="approx",
    ),
    "quantize_static_int8": Atom(
        "quantize_static_int8",
        "pass",
        {
            "calibration": {"enum": ["gz_frames", "assets"]},
            "per_channel": {"type": "boolean"},
            "format": {"enum": ["qdq", "qoperator"]},
        },
        numerics="approx",
    ),
    "convert_fp16": Atom(
        "convert_fp16",
        "pass",
        {},
        hardware_requires=("fp16_execution",),
        numerics="approx",
    ),
    "ort_graph_opt": Atom(
        "ort_graph_opt",
        "option",
        {"value": {"enum": ["disable", "basic", "extended", "all"]}},
    ),
    "intra_op_threads": Atom("intra_op_threads", "option", {"value": {"type": "integer"}}),
    "execution_mode": Atom(
        "execution_mode",
        "option",
        {"value": {"enum": ["sequential", "parallel"]}},
    ),
}

PASS_ATOMS: tuple[str, ...] = tuple(name for name, atom in ATOMS.items() if atom.kind == "pass")
OPTION_ATOMS: tuple[str, ...] = tuple(name for name, atom in ATOMS.items() if atom.kind == "option")
