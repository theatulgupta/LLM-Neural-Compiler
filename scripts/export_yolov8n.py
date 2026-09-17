#!/usr/bin/env python3
"""Back-compat wrapper. Prefer: python scripts/export_ultralytics.py --kind yolov8n"""

from __future__ import annotations

import argparse
from pathlib import Path

from compiler.catalog import REPO_ROOT, get_model
from export_ultralytics import export_ultralytics


def main() -> int:
    spec = get_model("yolov8n")
    parser = argparse.ArgumentParser(description="Export YOLOv8n to ONNX")
    parser.add_argument("--out", default=str(spec.onnx_path(REPO_ROOT)))
    parser.add_argument("--imgsz", type=int, default=spec.imgsz)
    parser.add_argument("--opset", type=int, default=spec.opset)
    args = parser.parse_args()
    _ = args.imgsz, args.opset
    return export_ultralytics("yolov8n", Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
