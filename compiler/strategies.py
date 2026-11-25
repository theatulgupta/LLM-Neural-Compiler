"""Allowlisted compile strategies. LLM output is rejected if it leaves this set."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Literal

OrtGraphOpt = Literal["disable", "basic", "extended", "all"]
ExecutionProvider = Literal[
    "CPUExecutionProvider",
    "XNNPACKExecutionProvider",
    "TensorRTExecutionProvider",
]
PrecisionMode = Literal["fp32", "int8_dynamic", "int8_static", "fp16"]
ExecutionMode = Literal["sequential", "parallel"]

ALLOWED_GRAPH_OPT_LEVELS: tuple[OrtGraphOpt, ...] = ("disable", "basic", "extended", "all")
ALLOWED_EXECUTION_PROVIDERS: tuple[ExecutionProvider, ...] = (
    "CPUExecutionProvider",
    "XNNPACKExecutionProvider",
    "TensorRTExecutionProvider",
)
ALLOWED_PRECISIONS: tuple[PrecisionMode, ...] = ("fp32", "int8_dynamic", "int8_static", "fp16")
ALLOWED_EXECUTION_MODES: tuple[ExecutionMode, ...] = ("sequential", "parallel")
ALLOWED_INTRA_OP_THREADS: tuple[int, ...] = (1, 2, 4, 8)
ALLOWED_INTER_OP_THREADS: tuple[int, ...] = (1, 2)
ALLOWED_ENABLE_MEM_PATTERN: tuple[bool, ...] = (True, False)


@dataclass(frozen=True, slots=True)
class Strategy:
    name: str
    description: str
    ort_graph_opt: OrtGraphOpt
    passes: tuple[str, ...]
    execution_provider: ExecutionProvider = "CPUExecutionProvider"
    precision: PrecisionMode = "fp32"
    execution_mode: ExecutionMode = "sequential"
    intra_op_num_threads: int = 0
    inter_op_num_threads: int = 0
    enable_mem_pattern: bool = True

    def config_dict(self) -> dict[str, Any]:
        return {
            "execution_provider": self.execution_provider,
            "graph_optimization_level": self.ort_graph_opt,
            "precision": self.precision,
            "execution_mode": self.execution_mode,
            "intra_op_num_threads": self.intra_op_num_threads,
            "inter_op_num_threads": self.inter_op_num_threads,
            "enable_mem_pattern": self.enable_mem_pattern,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ALLOWED_STRATEGIES: dict[str, Strategy] = {
    "baseline": Strategy(
        name="baseline",
        description="ORT CPU FP32 with graph optimizations disabled (correctness baseline).",
        ort_graph_opt="disable",
        passes=(),
        execution_provider="CPUExecutionProvider",
        precision="fp32",
        execution_mode="sequential",
        intra_op_num_threads=0,
        inter_op_num_threads=0,
        enable_mem_pattern=True,
    ),
    "ort_basic": Strategy(
        name="ort_basic",
        description="ORT CPU FP32 with basic graph optimizations (constant folding, redundant node removal).",
        ort_graph_opt="basic",
        passes=("onnx_shape_infer",),
        execution_provider="CPUExecutionProvider",
        precision="fp32",
        execution_mode="sequential",
        intra_op_num_threads=0,
        inter_op_num_threads=0,
        enable_mem_pattern=True,
    ),
    "ort_extended": Strategy(
        name="ort_extended",
        description="ORT CPU FP32 with extended graph optimizations (complex operator fusions).",
        ort_graph_opt="extended",
        passes=("onnx_shape_infer",),
        execution_provider="CPUExecutionProvider",
        precision="fp32",
        execution_mode="sequential",
        intra_op_num_threads=0,
        inter_op_num_threads=0,
        enable_mem_pattern=True,
    ),
    "ort_all": Strategy(
        name="ort_all",
        description="ORT CPU FP32 with all graph optimizations enabled (including NCHWc layout).",
        ort_graph_opt="all",
        passes=("onnx_shape_infer",),
        execution_provider="CPUExecutionProvider",
        precision="fp32",
        execution_mode="sequential",
        intra_op_num_threads=0,
        inter_op_num_threads=0,
        enable_mem_pattern=True,
    ),
}

ALLOWED_STRATEGY_NAMES: tuple[str, ...] = tuple(ALLOWED_STRATEGIES)


def get_strategy(name: str) -> Strategy:
    from compiler.errors import UnknownStrategyError

    if name not in ALLOWED_STRATEGIES:
        raise UnknownStrategyError(name, ALLOWED_STRATEGY_NAMES)
    return ALLOWED_STRATEGIES[name]


def validate_strategy_config(config: dict[str, Any]) -> dict[str, Any]:
    """Verify every dimension of a multi-dimensional strategy config is strictly allowlisted."""

    required = {
        "execution_provider",
        "graph_optimization_level",
        "precision",
        "execution_mode",
        "intra_op_num_threads",
        "inter_op_num_threads",
        "enable_mem_pattern",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"strategy config missing fields: {missing}")
    extra = sorted(set(config) - required)
    if extra:
        raise ValueError(f"strategy config has unknown fields: {extra}")

    ep = config["execution_provider"]
    opt = config["graph_optimization_level"]
    prec = config["precision"]
    mode = config["execution_mode"]
    intra = config["intra_op_num_threads"]
    inter = config["inter_op_num_threads"]
    mem = config["enable_mem_pattern"]

    if ep not in ALLOWED_EXECUTION_PROVIDERS:
        raise ValueError(f"execution_provider {ep!r} not allowlisted: {ALLOWED_EXECUTION_PROVIDERS}")
    if opt not in ALLOWED_GRAPH_OPT_LEVELS:
        raise ValueError(f"graph_optimization_level {opt!r} not allowlisted: {ALLOWED_GRAPH_OPT_LEVELS}")
    if prec not in ALLOWED_PRECISIONS:
        raise ValueError(f"precision {prec!r} not allowlisted: {ALLOWED_PRECISIONS}")
    if mode not in ALLOWED_EXECUTION_MODES:
        raise ValueError(f"execution_mode {mode!r} not allowlisted: {ALLOWED_EXECUTION_MODES}")
    if not isinstance(intra, int) or isinstance(intra, bool) or intra not in (0,) + ALLOWED_INTRA_OP_THREADS:
        raise ValueError(f"intra_op_num_threads {intra!r} not in {(0,) + ALLOWED_INTRA_OP_THREADS}")
    if not isinstance(inter, int) or isinstance(inter, bool) or inter not in (0,) + ALLOWED_INTER_OP_THREADS:
        raise ValueError(f"inter_op_num_threads {inter!r} not in {(0,) + ALLOWED_INTER_OP_THREADS}")
    if not isinstance(mem, bool):
        raise ValueError(f"enable_mem_pattern must be bool, got {type(mem).__name__}")
    return dict(config)


def strategy_from_config(base_name: str, config: dict[str, Any]) -> Strategy:
    base = get_strategy(base_name)
    checked = validate_strategy_config(config)
    opt: OrtGraphOpt = checked["graph_optimization_level"]
    passes: tuple[str, ...] = () if opt == "disable" else ("onnx_shape_infer",)
    label = (
        f"{base.name}[{checked['execution_provider']}|{opt}|{checked['precision']}|"
        f"{checked['execution_mode']}|intra={checked['intra_op_num_threads']}|"
        f"inter={checked['inter_op_num_threads']}|mem={int(checked['enable_mem_pattern'])}]"
    )
    return Strategy(
        name=base.name,
        description=label,
        ort_graph_opt=opt,
        passes=passes,
        execution_provider=checked["execution_provider"],
        precision=checked["precision"],
        execution_mode=checked["execution_mode"],
        intra_op_num_threads=checked["intra_op_num_threads"],
        inter_op_num_threads=checked["inter_op_num_threads"],
        enable_mem_pattern=checked["enable_mem_pattern"],
    )


def enumerate_strategy_space(
    *,
    include_tensorrt: bool = False,
    include_fp16: bool = False,
    include_int8_static: bool = False,
) -> list[Strategy]:
    """Generate the closed, multi-dimensional allowlisted strategy universe."""

    eps: list[ExecutionProvider] = ["CPUExecutionProvider", "XNNPACKExecutionProvider"]
    if include_tensorrt:
        eps.append("TensorRTExecutionProvider")

    precs: list[PrecisionMode] = ["fp32", "int8_dynamic"]
    if include_int8_static:
        precs.append("int8_static")
    if include_fp16:
        precs.append("fp16")

    strategies: list[Strategy] = []
    for ep, opt, prec, mode, intra, inter, mem in product(
        eps,
        ALLOWED_GRAPH_OPT_LEVELS,
        precs,
        ALLOWED_EXECUTION_MODES,
        ALLOWED_INTRA_OP_THREADS,
        ALLOWED_INTER_OP_THREADS,
        ALLOWED_ENABLE_MEM_PATTERN,
    ):
        base_name = "baseline" if opt == "disable" else f"ort_{opt}"
        cfg = {
            "execution_provider": ep,
            "graph_optimization_level": opt,
            "precision": prec,
            "execution_mode": mode,
            "intra_op_num_threads": intra,
            "inter_op_num_threads": inter,
            "enable_mem_pattern": mem,
        }
        strategies.append(strategy_from_config(base_name, cfg))
    return strategies
