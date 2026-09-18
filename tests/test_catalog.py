from __future__ import annotations

from compiler.catalog import get_model, load_zoo, zoo_kinds, zoo_tasks
from compiler.planner import ALLOWED_STRATEGY_NAMES


def test_zoo_has_distinct_uav_tasks() -> None:
    specs = load_zoo()
    kinds = zoo_kinds()
    tasks = zoo_tasks()
    assert "yolov8n" in kinds
    assert "yolo11n" in kinds
    assert "yolov8n-pose" in kinds
    assert "ssdlite_mobilenetv3" in kinds
    assert "mobilenetv3_small" in kinds
    assert "midas_small" in kinds
    assert "detect" in tasks
    assert "pose" in tasks
    assert "classify" in tasks
    assert "detect-lite" in tasks
    assert "depth" in tasks
    assert len(kinds) == len(set(kinds))
    assert len(specs) >= 6
    # Two detectors are allowed (YOLOv8n vs YOLO11n) but must not share the same weights.
    detect_weights = {spec.weights for spec in specs if spec.task == "detect"}
    assert len(detect_weights) >= 2


def test_zoo_strategies_are_allowlisted() -> None:
    for spec in load_zoo():
        assert spec.native_strategy in ALLOWED_STRATEGY_NAMES
        assert spec.default_strategy in ALLOWED_STRATEGY_NAMES
        assert spec.native_strategy == "baseline"
        assert spec.export_script.endswith(".py")


def test_cli_zoo_json_has_imgsz(capsys) -> None:
    import json

    from compiler.cli import main

    assert main(["zoo"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models"]
    for row in payload["models"]:
        assert isinstance(row["imgsz"], int)
        assert row["imgsz"] > 0
    for spec in load_zoo():
        assert spec.imgsz in {224, 256, 320, 640}


def test_get_model_unknown_raises() -> None:
    try:
        get_model("not-a-real-uav-net")
    except KeyError as exc:
        assert "yolov8n" in str(exc)
    else:
        raise AssertionError("expected KeyError")
