"""Dataset-level object detection accuracy (mAP@0.5 and mAP@0.5:0.95) for ONNX YOLO models."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def evaluate_yolo_map(
    model_path: Path | str,
    *,
    data_yaml: Path | str = Path("experiments/data/coco128/coco128.yaml"),
    imgsz: int = 640,
) -> dict[str, Any]:
    """Run Ultralytics validation on coco128 for an ONNX model and return mAP metrics."""

    from ultralytics import YOLO

    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.is_file():
        # Create a local coco128.yaml pointing to experiments/data/coco128
        root = yaml_path.parent.resolve()
        yaml_path.write_text(
            f"path: {root}\ntrain: images/train2017\nval: images/train2017\nnc: 80\n",
            encoding="utf-8",
        )

    model = YOLO(str(model_path), task="detect")
    metrics = model.val(data=str(yaml_path), imgsz=imgsz, verbose=False, plots=False)
    box = getattr(metrics, "box", None)
    map50 = float(getattr(box, "map50", 0.0)) if box is not None else 0.0
    map50_95 = float(getattr(box, "map", 0.0)) if box is not None else 0.0
    return {
        "map50": map50,
        "map50_95": map50_95,
        "dataset": str(yaml_path),
    }
