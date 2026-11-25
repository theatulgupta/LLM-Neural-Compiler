from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from compiler.errors import DimensionalContractError
from compiler.evaluation.baselines import compute_regret, run_oracle_and_baselines
from compiler.evaluation.fidelity import compare_outputs, evaluate_top1_agreement
from compiler.graph.contracts import verify_dimensional_contracts
from compiler.graph.graph_analyzer import analyze_graph
from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph
from compiler.llm.llm_client import MockLlmClient, _extract_json_object
from compiler.llm.prompts import user_prompt
from compiler.llm.recommendation_engine import recommend_strategy
from compiler.optimization.quantizer import quantize_model_bytes
from compiler.pipeline import compile_and_benchmark
from compiler.schema_validate import SchemaError, validate_llm_proposal
from compiler.strategies import (
    enumerate_strategy_space,
    strategy_from_config,
    validate_strategy_config,
)


def test_shape_aware_flops_on_tiny_cnn(tiny_path) -> None:
    stats = analyze_graph(load_graph(tiny_path))
    # Tiny CNN: Conv(4608 + 36*0? bias=4*64=256) + ReLU(256) + MaxPool(256) + Gemm(1024+8) = 6408
    assert stats["estimated_flops"] is not None
    assert stats["estimated_flops"] >= 5632
    assert stats["estimated_macs"] == stats["estimated_flops"] // 2


def test_dimensional_contracts_pass_and_catch_k16(tiny_path, broken_k16_path) -> None:
    report = verify_dimensional_contracts(load_graph(tiny_path).model)
    assert report["passed"] is True
    assert report["checked_count"] >= 2

    broken_model = load_graph(broken_k16_path, check=False).model
    with pytest.raises(DimensionalContractError):
        verify_dimensional_contracts(broken_model)


def test_multidim_strategy_space_and_schema() -> None:
    space = enumerate_strategy_space(include_int8_static=True)
    # 2 EPs * 4 opts * 3 precisions * 2 modes * 4 intra * 2 inter * 2 mem = 768
    assert len(space) == 768

    cfg = {
        "execution_provider": "CPUExecutionProvider",
        "graph_optimization_level": "extended",
        "precision": "int8_dynamic",
        "execution_mode": "sequential",
        "intra_op_num_threads": 4,
        "inter_op_num_threads": 1,
        "enable_mem_pattern": True,
    }
    validate_strategy_config(cfg)
    strat = strategy_from_config("ort_extended", cfg)
    assert strat.precision == "int8_dynamic"
    assert strat.intra_op_num_threads == 4

    payload = {
        "strategy": "ort_extended",
        "config": cfg,
        "rationale": "INT8 dynamic + 4 threads fits ARM64 CPU",
        "source": "unit-test",
    }
    validate_llm_proposal(payload)

    bad_cfg = dict(cfg, intra_op_num_threads=99)
    with pytest.raises(SchemaError):
        validate_llm_proposal(dict(payload, config=bad_cfg))


def test_recommendation_deterministic_fallback(tiny_path) -> None:
    summary = summarize_graph(load_graph(tiny_path))
    bad_client = MockLlmClient(
        {"strategy": "hallucinated_pass", "rationale": "bad", "source": "mock"}
    )
    rec = recommend_strategy(summary, client=bad_client, fallback_on_error=True)
    assert rec.fallback_triggered is True
    assert rec.strategy.name == "baseline"
    assert rec.rejection_reason is not None


def test_quantize_dynamic_and_fidelity(tiny_path, tmp_path) -> None:
    raw = tiny_path.read_bytes()
    q_bytes = quantize_model_bytes(raw, "int8_dynamic")
    assert len(q_bytes) > 0

    cfg = {
        "execution_provider": "CPUExecutionProvider",
        "graph_optimization_level": "extended",
        "precision": "int8_dynamic",
        "execution_mode": "sequential",
        "intra_op_num_threads": 2,
        "inter_op_num_threads": 1,
        "enable_mem_pattern": True,
    }
    strat = strategy_from_config("ort_extended", cfg)
    rec = compile_and_benchmark(
        tiny_path,
        strategy_obj=strat,
        check_fidelity=True,
        warmup=1,
        iters=3,
        results_dir=tmp_path / "results",
    )
    assert rec["compile"]["ok"] is True
    assert rec["fidelity"]["cosine_similarity"] >= 0.90
    assert rec["benchmark"]["latency_ms"]["p99"] > 0


def test_fidelity_helpers_and_prompt_history(tiny_path) -> None:
    a = [np.array([[1.0, 2.0, 0.5]], dtype=np.float32)]
    b = [np.array([[1.0, 2.1, 0.5]], dtype=np.float32)]
    metrics = compare_outputs(a, b)
    assert metrics["max_abs_error"] == pytest.approx(0.1, abs=1e-5)
    assert metrics["cosine_similarity"] > 0.99
    assert evaluate_top1_agreement(a, b) == 1.0

    summary = summarize_graph(load_graph(tiny_path))
    prompt = user_prompt(
        summary,
        history_feedback=[
            {
                "strategy": {"strategy": "ort_all", "config": {"precision": "fp32"}},
                "compile": {"ok": True},
                "benchmark": {"latency_ms": {"mean": 66.2, "p95": 81.0}},
                "fidelity": {"cosine_similarity": 1.0},
            }
        ],
    )
    assert "closed-loop context" in prompt.lower()
    assert "66.2" in prompt


def test_extract_json_and_regret_math() -> None:
    parsed = _extract_json_object('```json\n{"strategy": "ort_all", "rationale": "ok", "source": "x"}\n```')
    assert parsed["strategy"] == "ort_all"
    assert compute_regret(12.0, 10.0) == pytest.approx(0.2)
