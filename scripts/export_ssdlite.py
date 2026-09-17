#!/usr/bin/env python3
"""Export SSDLite320 MobileNetV3-Large heads to ONNX. Skip JSON on failure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from compiler.catalog import REPO_ROOT, get_model
from compiler.exporters import export_note, finish_onnx, write_skip


class SSDLiteHeads(torch.nn.Module):
    """Backbone + box/class heads. NMS stays off-graph (same as default YOLO export)."""

    def __init__(self, detector: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = detector.backbone
        self.head = detector.head

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(images)
        if isinstance(features, dict):
            features = list(features.values())
        elif isinstance(features, torch.Tensor):
            features = [features]
        outputs = self.head(features)
        return outputs["bbox_regression"], outputs["cls_logits"]


def _onnx_export(model: torch.nn.Module, dummy: torch.Tensor, dest: Path, opset: int) -> None:
    kwargs = {
        "input_names": ["images"],
        "output_names": ["bbox_regression", "cls_logits"],
        "opset_version": opset,
    }
    try:
        torch.onnx.export(model, dummy, str(dest), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(dest), **kwargs)


def export_ssdlite(out: Path) -> int:
    spec = get_model("ssdlite_mobilenetv3")
    try:
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )
    except ImportError as exc:
        reason = f"torchvision detection is not installed ({exc})"
        print(f"skip: {reason}", file=sys.stderr)
        write_skip(kind=spec.kind, path=str(out), reason=reason)
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.onnx")
    try:
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        detector = ssdlite320_mobilenet_v3_large(weights=weights)
        detector.eval()
        wrapped = SSDLiteHeads(detector)
        wrapped.eval()
        dummy = torch.zeros(1, 3, spec.imgsz, spec.imgsz)
        with torch.inference_mode():
            _onnx_export(wrapped, dummy, tmp, spec.opset)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        reason = f"{type(exc).__name__}: {exc}"
        print(f"skip: {reason}", file=sys.stderr)
        write_skip(kind=spec.kind, path=str(out), reason=reason)
        return 2

    digest = finish_onnx(tmp, out)
    print(export_note(spec, out, digest))
    return 0


def main() -> int:
    spec = get_model("ssdlite_mobilenetv3")
    parser = argparse.ArgumentParser(description="Export SSDLite MobileNetV3 to ONNX")
    parser.add_argument("--out", default=str(spec.onnx_path(REPO_ROOT)))
    args = parser.parse_args()
    return export_ssdlite(Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
