from __future__ import annotations

from pathlib import Path

from compiler.pipeline import compile_verify_profile
from compiler.planner.plan import get_plan


def test_pipeline_v2_graph_fuse_tiny(tiny_path, tmp_path) -> None:
    record = compile_verify_profile(
        tiny_path,
        plan=get_plan("graph_fuse"),
        kind="fixture",
        task="classify",
        results_dir=tmp_path,
        warmup=1,
        iters=2,
    )
    assert record["compile"]["ok"] is True
    assert record["graph_runtime"]["op_counts"].get("FusedConv", 0) == 1
    assert record["verification"]["passed"] is True
    assert Path(record["artifact"]["path"]).is_file()
    assert record["schema_version"] == 2


def test_pipeline_v2_int8_does_not_raise(tiny_path, tmp_path) -> None:
    record = compile_verify_profile(
        tiny_path,
        plan=get_plan("int8_dynamic"),
        kind="fixture",
        task="classify",
        results_dir=tmp_path,
        warmup=0,
        iters=1,
    )
    assert "compile" in record
    assert record.get("fps_claimed") is False


def test_reuses_loaded_graph_and_native_session(tiny_path, tmp_path, monkeypatch) -> None:
    from compiler.graph.graph_loader import load_graph
    from nnc.backends.base import BackendOptions
    from nnc.backends.ort_cpu import OrtCpuBackend

    loaded = load_graph(tiny_path)
    calls = {"n": 0}
    real = OrtCpuBackend.compile

    def wrapped(self, model_bytes, **kwargs):
        calls["n"] += 1
        return real(self, model_bytes, **kwargs)

    monkeypatch.setattr(OrtCpuBackend, "compile", wrapped)
    native = OrtCpuBackend().compile(
        loaded.model.SerializeToString(), options=BackendOptions(graph_opt="disable")
    )
    before = calls["n"]

    def refuse_reload(*_args, **_kwargs):
        raise AssertionError("load_graph should not run when loaded= is set")

    monkeypatch.setattr("compiler.pipeline.compile.load_graph", refuse_reload)
    for name in ("baseline", "graph_fuse"):
        record = compile_verify_profile(
            tiny_path,
            plan=get_plan(name),
            kind="fixture",
            task="classify",
            results_dir=tmp_path,
            warmup=0,
            iters=1,
            loaded=loaded,
            native=native,
        )
        assert record["compile"]["ok"] is True
        assert record["verification"]["passed"] is True
    assert calls["n"] == before + 2
