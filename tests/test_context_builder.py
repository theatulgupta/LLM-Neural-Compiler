from __future__ import annotations

from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph
from compiler.hardware.profile import probe_hardware
from compiler.llm.context_builder import build_context, render_user_prompt
from compiler.llm.prompting import build_messages


def test_context_contains_patterns_and_hardware(tiny_path) -> None:
    summary = summarize_graph(load_graph(tiny_path))
    hw = probe_hardware()
    ctx = build_context(
        summary, hw, {"latency_ms": "50"}, [{"plan_id": "baseline", "p50_ms": 1.2, "passed": True}]
    )
    prompt = render_user_prompt(ctx)
    assert "conv_relu" in prompt
    assert "cpu_count" in prompt
    assert "baseline" in prompt
    assert hw.cpu_count > 0
    assert hw.arch
    messages = build_messages(summary, ctx)
    assert messages[0]["role"] == "system"
    assert "allowlisted" in messages[1]["content"].lower()
