from __future__ import annotations

from typing import Any

import pytest

from nnc.backends import get_backend, known_backends, register_backend, unregister_backend
from nnc.backends.base import Backend, CompiledModel, TensorSpec


def test_builtin_backends_are_registered() -> None:
    names = known_backends()
    assert "ort_cpu" in names
    assert "tensorrt" in names
    assert get_backend("ort_cpu").name == "ort_cpu"
    assert get_backend("tensorrt").name == "tensorrt"


def test_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("not_a_backend")


def test_new_backend_registers_without_pipeline_edit() -> None:
    class DummyBackend(Backend):
        name = "dummy_test_only"

        def available(self) -> tuple[bool, str | None]:
            return True, None

        def compile(self, model_bytes: bytes, *, graph_opt: str) -> CompiledModel:
            spec = TensorSpec(name="x", shape=(1, 1, 8, 8), dtype="float32")
            return CompiledModel(
                backend=self.name,
                session=object(),
                input_names=("x",),
                output_names=("y",),
                providers=(),
                inputs=(spec,),
            )

        def infer(self, compiled: CompiledModel, feeds: dict[str, Any]) -> list[Any]:
            return []

    register_backend(DummyBackend)
    try:
        backend = get_backend("dummy_test_only")
        compiled = backend.compile(b"", graph_opt="disable")
        assert compiled.inputs[0].numpy_shape() == (1, 1, 8, 8)
        assert "dummy_test_only" in known_backends()
    finally:
        unregister_backend("dummy_test_only")
    assert "dummy_test_only" not in known_backends()
