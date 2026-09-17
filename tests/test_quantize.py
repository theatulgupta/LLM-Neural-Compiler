from __future__ import annotations

from compiler.graph.graph_loader import load_graph
from compiler.optimization.passes import apply_pass, graph_ir_snapshot
from compiler.planner.plan import get_plan
from compiler.optimization.optimizer import apply_plan
from nnc.backends.ort_cpu import OrtCpuBackend
import numpy as np


def test_int8_dynamic_tiny(tiny_path) -> None:
    loaded = load_graph(tiny_path)
    quantized = apply_pass("quantize_dynamic_int8", loaded.model, {"per_channel": True, "weight_type": "qint8"})
    ops = graph_ir_snapshot(quantized)["op_counts"]
    assert "ConvInteger" in ops or "MatMulInteger" in ops or "QLinearConv" in ops or "DynamicQuantizeLinear" in ops
    backend = OrtCpuBackend()
    compiled = backend.compile(quantized.SerializeToString(), graph_opt="disable")
    x = np.random.default_rng(0).standard_normal((1, 1, 8, 8), dtype=np.float32)
    out = backend.infer(compiled, {compiled.input_names[0]: x})
    assert out[0].shape == (1, 8)


def test_int8_dynamic_preset(tiny_path) -> None:
    loaded = load_graph(tiny_path)
    model, steps = apply_plan(loaded.model, get_plan("int8_dynamic"))
    assert steps
    assert graph_ir_snapshot(model)["node_count"] >= 1
