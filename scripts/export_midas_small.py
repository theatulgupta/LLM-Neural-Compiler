#!/usr/bin/env python3
"""Export MiDaS small monocular depth to ONNX. Skip JSON if hub/export is messy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from compiler.catalog import REPO_ROOT, get_model
from compiler.exporters import export_note, finish_onnx, write_skip


def export_midas_small(out: Path) -> int:
    spec = get_model("midas_small")
    try:
        import timm  # noqa: F401  — MiDaS_small backbone
        import torch
    except ImportError as exc:
        reason = f"torch/timm missing for MiDaS ({exc}). Install timm or skip depth."
        print(f"skip: {reason}", file=sys.stderr)
        write_skip(kind=spec.kind, path=str(out), reason=reason)
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.onnx")
    try:
        hub_dir = Path(torch.hub.get_dir())
        trusted = hub_dir / "trusted_list"
        trusted.parent.mkdir(parents=True, exist_ok=True)
        owners = {"intel-isl", "isl-org", "rwightman"}
        existing = set()
        if trusted.is_file():
            existing = {
                line.strip() for line in trusted.read_text(encoding="utf-8").splitlines() if line.strip()
            }
        trusted.write_text("\n".join(sorted(existing | owners)) + "\n", encoding="utf-8")
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True, trust_repo=True)
        model.eval()
        dummy = torch.zeros(1, 3, spec.imgsz, spec.imgsz)
        kwargs = {
            "input_names": ["image"],
            "output_names": ["depth"],
            "opset_version": spec.opset,
        }
        with torch.inference_mode():
            try:
                torch.onnx.export(model, dummy, str(tmp), dynamo=False, **kwargs)
            except TypeError:
                torch.onnx.export(model, dummy, str(tmp), **kwargs)
    except Exception as exc:  # noqa: BLE001 — do not invent a depth net
        tmp.unlink(missing_ok=True)
        reason = f"{type(exc).__name__}: {exc}"
        print(f"skip: {reason}", file=sys.stderr)
        write_skip(kind=spec.kind, path=str(out), reason=reason)
        return 2

    digest = finish_onnx(tmp, out)
    print(export_note(spec, out, digest))
    return 0


def main() -> int:
    spec = get_model("midas_small")
    parser = argparse.ArgumentParser(description="Export MiDaS small to ONNX")
    parser.add_argument("--out", default=str(spec.onnx_path(REPO_ROOT)))
    args = parser.parse_args()
    return export_midas_small(Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
