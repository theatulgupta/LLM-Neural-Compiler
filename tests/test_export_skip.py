from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path

from compiler.catalog import REPO_ROOT


def _load_export_ultralytics():
    path = REPO_ROOT / "scripts" / "export_ultralytics.py"
    spec = importlib.util.spec_from_file_location("export_ultralytics_mod", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ultralytics_export_skips_without_package(tmp_path, monkeypatch) -> None:
    module = _load_export_ultralytics()
    monkeypatch.setattr(module, "write_skip", lambda **kwargs: tmp_path / "skip.json")
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ultralytics" or name.startswith("ultralytics"):
            raise ImportError("ultralytics missing in unit test")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    out = tmp_path / "yolov8n.onnx"
    code = module.export_ultralytics("yolov8n", out)
    assert code == 2
    assert not out.is_file()
