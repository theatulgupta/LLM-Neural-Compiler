#!/usr/bin/env python3
"""Export an Ultralytics nano/pose/seg checkpoint to ONNX. Skip JSON on failure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from compiler.catalog import REPO_ROOT, get_model
from compiler.exporters import cleanup_cwd_weights, export_note, finish_onnx, write_skip


def export_ultralytics(kind: str, out: Path) -> int:
    spec = get_model(kind)
    if spec.family != "ultralytics":
        print(f"skip: {kind} is family {spec.family}, not ultralytics", file=sys.stderr)
        write_skip(kind=kind, path=str(out), reason=f"not an ultralytics model ({spec.family})")
        return 2
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        reason = f"ultralytics is not installed ({exc}). Install with: pip install '.[yolo]'"
        print(f"skip: {reason}", file=sys.stderr)
        write_skip(kind=kind, path=str(out), reason=reason)
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        model = YOLO(spec.weights)
        exported = model.export(format="onnx", imgsz=spec.imgsz, opset=spec.opset, simplify=False)
    except Exception as exc:  # noqa: BLE001 — record the real export/download failure
        reason = f"{type(exc).__name__}: {exc}"
        print(f"skip: {reason}", file=sys.stderr)
        write_skip(kind=kind, path=str(out), reason=reason)
        return 2

    exported_path = Path(str(exported))
    digest = finish_onnx(exported_path, out)
    cleanup_cwd_weights(spec.weights)
    print(export_note(spec, out, digest))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Export an Ultralytics zoo model to ONNX")
    parser.add_argument("--kind", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    spec = get_model(args.kind)
    out = Path(args.out) if args.out else spec.onnx_path(REPO_ROOT)
    return export_ultralytics(args.kind, out)


if __name__ == "__main__":
    raise SystemExit(main())
