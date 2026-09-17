# YOLOv8n experiment model

YOLOv8n is one model in the companion zoo (`experiments/zoo.yaml`). The paper
table is native vs allowlisted ORT on this host: `docs/paper_proposal.md` and
`experiments/results/paper_matrix.json`. The numbers below are an earlier
unloaded YOLOv8n-only run and are **not** mixed with cloud x86.

## Export (this host)

CPU-only aarch64 QEMU. Do not install the default PyPI CUDA torch wheel.

```bash
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
python -m compiler export --kind yolov8n
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
| ORT CPU compile (prior ORT-knob run, not `graph_fuse`) | 85.0 ms (`CPUExecutionProvider`) |
| ORT CPU latency | mean 66.2 ms, p50 61.0 ms, p95 93.0 ms (warmup 5, iters 25, aarch64 QEMU CPU) |
| Throughput | 15.1 IPS |

ROS 2 `inference_node` also loaded this same SHA-256 while PX4 SITL was running:
latency 78–84 ms, output `1×84×8400`, `fps_claimed=false`
(`experiments/results/inference_ros_yolov8n.json`). That is a loaded-host
measurement, not a replacement for the unloaded baseline above.

Compile through ORT CPU:

```bash
python -m compiler compile experiments/models/yolov8n.onnx \
  --backend ort_cpu --strategy graph_fuse --kind yolov8n
```
