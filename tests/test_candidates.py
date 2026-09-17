from __future__ import annotations

from compiler.graph.graph_loader import load_graph
from compiler.graph.graph_summary import summarize_graph
from compiler.hardware.profile import probe_hardware
from compiler.planner.candidates import generate_candidates


def test_candidates_include_presets(tiny_path) -> None:
    summary = summarize_graph(load_graph(tiny_path))
    cands = generate_candidates(summary, probe_hardware(), mode="default")
    origins = {c.origin for c in cands}
    assert any(o.startswith("preset:baseline") for o in origins)
    assert any("graph_fuse" in o or o == "heuristic" for o in origins)
