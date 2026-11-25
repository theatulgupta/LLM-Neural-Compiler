"""Export a diverse 6-8 model UAV edge perception ONNX Model Zoo."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def export_torchvision_models(out_dir: Path, names: list[str]) -> list[str]:
    import torch
    import torchvision.models as models

    factories = {
        "mobilenet_v2": (lambda: models.mobilenet_v2(weights=None), (1, 3, 224, 224)),
        "mobilenet_v3_small": (lambda: models.mobilenet_v3_small(weights=None), (1, 3, 224, 224)),
        "resnet18": (lambda: models.resnet18(weights=None), (1, 3, 224, 224)),
        "efficientnet_b0": (lambda: models.efficientnet_b0(weights=None), (1, 3, 224, 224)),
        "shufflenet_v2_x1_0": (lambda: models.shufflenet_v2_x1_0(weights=None), (1, 3, 224, 224)),
        "vit_b_16": (lambda: models.vit_b_16(weights=None), (1, 3, 224, 224)),
    }

    exported: list[str] = []
    for name in names:
        if name not in factories:
            continue
        fn, shape = factories[name]
        model = fn().eval()
        dummy = torch.randn(*shape)
        out_path = out_dir / f"{name}.onnx"
        torch.onnx.export(
            model,
            dummy,
            str(out_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
        )
        exported.append(str(out_path))
        print(f"Exported {out_path}")
    return exported


def main() -> int:
    parser = argparse.ArgumentParser(description="Export UAV perception ONNX Model Zoo")
    parser.add_argument("--out-dir", type=Path, default=Path("experiments/models"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "mobilenet_v2",
            "mobilenet_v3_small",
            "resnet18",
            "efficientnet_b0",
            "shufflenet_v2_x1_0",
        ],
    )
    parser.add_argument("--include-yolo", action="store_true", help="Also export yolov8n and yolov8s via ultralytics")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    exported = export_torchvision_models(args.out_dir, args.models)

    if args.include_yolo:
        from ultralytics import YOLO

        for variant in ("yolov8n", "yolov8s"):
            yolo = YOLO(f"{variant}.pt")
            onnx_file = Path(yolo.export(format="onnx", imgsz=640, opset=17, simplify=False))
            dest = args.out_dir / f"{variant}.onnx"
            if onnx_file.resolve() != dest.resolve():
                dest.write_bytes(onnx_file.read_bytes())
            exported.append(str(dest))
            print(f"Exported {dest}")

    print(f"Total exported: {len(exported)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
