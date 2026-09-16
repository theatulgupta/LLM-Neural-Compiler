# YOLOv8n experiment model

Chosen experiment model: **YOLOv8n** (Ultralytics, detect, 640²).

## Export (this host)

CPU-only aarch64 QEMU. Do not install the default PyPI CUDA torch wheel.

```bash
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
python scripts/export_yolov8n.py --out experiments/models/yolov8n.onnx --imgsz 640 --opset 13
```

## Artifact measured here

| Field | Value |
| --- | --- |
| Path | `experiments/models/yolov8n.onnx` (gitignored) |
| Bytes | 12824019 (12.2 MB) |
| SHA-256 | `1d883eedafc9024446074d0b642ef8220a0466e3eb27267888c060c5df7a2e87` |
| Ultralytics | 8.4.153 |
| Torch | 2.14.0+cpu |
| ONNX | 1.22.0 opset 13 |
| Input | `1×3×640×640` |
| Output | `1×84×8400` |
| Params (fused) | 3,151,904 |
| GFLOPs (Ultralytics) | 8.7 |
| ORT CPU compile | 85.0 ms (`ort_extended`, `CPUExecutionProvider`) |
| ORT CPU latency | mean 66.2 ms, p50 61.0 ms, p95 93.0 ms (warmup 5, iters 25, aarch64 QEMU CPU) |
| Throughput | 15.1 IPS |

Compile through ORT CPU:

```bash
python -m compiler compile experiments/models/yolov8n.onnx \
  --backend ort_cpu --strategy ort_extended --kind yolov8n
```
