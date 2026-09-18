from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nnc.postprocess import (
    boxes_to_original,
    decode,
    ssdlite_default_boxes,
    yolo_decode_nms,
)


def test_yolo_detect_shape_and_positive() -> None:
    arr = np.zeros((1, 84, 8400), dtype=np.float32)
    arr[0, 0, 0] = 100.0
    arr[0, 1, 0] = 100.0
    arr[0, 2, 0] = 20.0
    arr[0, 3, 0] = 20.0
    arr[0, 4, 0] = 0.9
    out = yolo_decode_nms(arr, num_classes=80, conf=0.25, iou=0.45)
    assert out["boxes"].shape[1] == 4
    assert len(out["boxes"]) == 1
    assert out["classes"][0] == 0
    assert out["scores"][0] > 0.8


def test_yolo_pose_keypoints() -> None:
    arr = np.zeros((1, 56, 80), dtype=np.float32)
    arr[0, 0, 0] = 50
    arr[0, 1, 0] = 50
    arr[0, 2, 0] = 10
    arr[0, 3, 0] = 10
    arr[0, 4, 0] = 0.99
    arr[0, 5:56, 0] = np.arange(51)
    out = decode("pose", [arr], conf=0.25)
    assert out["keypoints"].shape == (1, 17, 3)
    assert out["boxes"].shape == (1, 4)


def test_yolo_segment_boxes_only() -> None:
    arr = np.zeros((1, 116, 8), dtype=np.float32)
    out = decode("segment", [arr], conf=0.99)
    assert out["boxes"].shape == (0, 4)
    assert out["mask_decoded"] is False


def test_ssdlite_prior_count() -> None:
    boxes = ssdlite_default_boxes()
    assert len(boxes) == 3234
    assert boxes.shape == (3234, 4)


def test_boxes_to_original_roundtrip() -> None:
    meta = {"pad": (10, 20), "scale": 0.5}
    original = np.asarray([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32)
    letterboxed = original.copy()
    letterboxed[:, [0, 2]] = original[:, [0, 2]] * 0.5 + 10
    letterboxed[:, [1, 3]] = original[:, [1, 3]] * 0.5 + 20
    recovered = boxes_to_original(letterboxed, meta)
    assert np.allclose(recovered, original, atol=1e-4)


def test_ssdlite_onnx_bus_jpg() -> None:
    onnx = Path("experiments/models/ssdlite_mobilenetv3.onnx")
    if not onnx.is_file():
        pytest.skip("ssdlite_mobilenetv3.onnx not exported")
    try:
        from PIL import Image
    except ImportError:
        pytest.skip("PIL missing")
    assets = [
        Path(".venv/lib/python3.12/site-packages/ultralytics/assets/bus.jpg"),
        Path.home() / "LLM-Neural-Compiler/.venv/lib/python3.12/site-packages/ultralytics/assets/bus.jpg",
    ]
    jpg = next((p for p in assets if p.is_file()), None)
    if jpg is None:
        pytest.skip("ultralytics bus.jpg missing")
    from nnc.artifact import infer_once, load_ort_artifact
    from nnc.preprocess import letterbox

    rgb = np.asarray(Image.open(jpg).convert("RGB"), dtype=np.uint8)
    nchw, _meta = letterbox(rgb, 320, rgb=True)
    art = load_ort_artifact(onnx, graph_opt="disable")
    sample = infer_once(art, {art.input_name: nchw})
    det = decode("detect-lite", list(sample.outputs or []), conf=0.3)
    labels = set(int(c) for c in det["classes"])
    assert labels & {1, 6}, f"expected person(1) or bus(6), got {labels}"
