from __future__ import annotations

from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph
from compiler.hardware.profile import probe_hardware
from compiler.planner.plan import Plan, get_plan
from compiler.planner.verifier import verify_plan


def test_verifier_drops_fp16_on_cpu(tiny_path) -> None:
    summary = summarize_graph(load_graph(tiny_path))
    hw = probe_hardware()
    plan = Plan.from_dict(
        {
            "plan_id": "fp16",
            "steps": [{"atom": "convert_fp16", "params": {}}],
            "options": {"ort_graph_opt": "disable", "intra_op_threads": 999},
            "rationale": "try fp16",
            "confidence": 0.2,
            "source": "test",
        }
    )
    verified = verify_plan(plan, summary, hw, backend_name="ort_cpu")
    assert verified.accepted is True
    assert all(step["atom"] != "convert_fp16" for step in verified.plan.steps)
    assert verified.plan.options["intra_op_threads"] <= hw.cpu_count
    assert any(row["step"] == "convert_fp16" for row in verified.rejections)


def test_verifier_drops_bn_when_absent(tiny_path) -> None:
    summary = summarize_graph(load_graph(tiny_path))
    hw = probe_hardware()
    verified = verify_plan(get_plan("graph_fuse"), summary, hw)
    assert verified.accepted is True
    assert all(step["atom"] != "fuse_bn_into_conv" for step in verified.plan.steps)


def test_verifier_rejects_unknown_atom(tiny_path) -> None:
    summary = summarize_graph(load_graph(tiny_path))
    hw = probe_hardware()
    plan = Plan(
        plan_id="bad",
        steps=({"atom": "not_an_atom", "params": {}},),
        options={"ort_graph_opt": "disable"},
        rationale="bad",
        confidence=0.1,
        source="test",
    )
    verified = verify_plan(plan, summary, hw)
    assert verified.accepted is False
