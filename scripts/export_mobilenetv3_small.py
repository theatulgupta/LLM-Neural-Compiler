#!/usr/bin/env python3
"""Export ImageNet MobileNetV3-small to ONNX. Skip JSON on failure."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from compiler.catalog import REPO_ROOT, get_model
from compiler.exporters import export_note, finish_onnx, write_skip


def _onnx_export(model: torch.nn.Module, dummy: torch.Tensor, dest: Path, opset: int) -> None:
    kwargs = {
        "input_names": ["images"],
        "output_names": ["logits"],
        "opset_version": opset,
    }
    try:
        torch.onnx.export(model, dummy, str(dest), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, dummy, str(dest), **kwargs)


def export_mobilenetv3_small(out: Path) -> int:
    spec = get_model("mobilenetv3_small")
    try:
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    except ImportError as exc:
        reason = f"torchvision is not installed ({exc})"
        print(f"skip: {reason}", file=sys.stderr)
        write_skip(kind=spec.kind, path=str(out), reason=reason)
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.onnx")
    try:
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = mobilenet_v3_small(weights=weights)
        model.eval()
        dummy = torch.zeros(1, 3, spec.imgsz, spec.imgsz)
        with torch.inference_mode():
            _onnx_export(model, dummy, tmp, spec.opset)
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
    spec = get_model("mobilenetv3_small")
    parser = argparse.ArgumentParser(description="Export MobileNetV3-small to ONNX")
    parser.add_argument("--out", default=str(spec.onnx_path(REPO_ROOT)))
    args = parser.parse_args()
    return export_mobilenetv3_small(Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
