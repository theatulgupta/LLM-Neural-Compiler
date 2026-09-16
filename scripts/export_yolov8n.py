#!/usr/bin/env python3
"""Export YOLOv8n to ONNX. Fails honestly if ultralytics/torch are missing."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLOv8n to ONNX")
    parser.add_argument("--out", default="experiments/models/yolov8n.onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=13)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(
            "skip: ultralytics is not installed. "
            "Install with: pip install '.[yolo]'\n"
            f"import error: {exc}",
            file=sys.stderr,
        )
        return 2

    model = YOLO("yolov8n.pt")
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=False)
    exported_path = Path(str(exported))
    if exported_path.resolve() != out.resolve():
        out.write_bytes(exported_path.read_bytes())
        if exported_path.name == "yolov8n.onnx" and exported_path.parent == Path.cwd():
            exported_path.unlink(missing_ok=True)
    weights = Path("yolov8n.pt")
    if weights.is_file() and weights.parent.resolve() == Path.cwd().resolve():
        weights.unlink(missing_ok=True)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out} bytes={out.stat().st_size} sha256={digest}")
    print("next: python -m compiler compile", out, "--backend ort_cpu --strategy ort_extended --kind yolov8n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
