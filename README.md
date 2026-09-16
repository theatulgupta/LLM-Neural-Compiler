# LLM-Neural-Compiler

LLM-guided neural compilation for real-time UAV edge inference. The thesis
contribution lives in `compiler/`: ONNX graph analysis, an allowlisted strategy
recommender, and compile/benchmark history. Runtime backends live in
`src/nnc/backends/` (ONNX Runtime CPU and TensorRT).

This tree does **not** add UAV-agent stubs. The existing ROS 2 telemetry node
and empty planning/control modules stay as they are.

## Models

| Role | Model | Notes |
| --- | --- | --- |
| Fixture (tests + CPU baseline) | Tiny CNN, input `1×1×8×8` | Flatten width **64**; Gemm `K` must be 64 (`transB=1`, W is `[8, 64]`). A `K=16` graph is the known-broken regression. |
| Experiment | YOLOv8n | Export with `scripts/export_yolov8n.py`. ONNX is gitignored. |

## Setup

Python 3.11+ (3.12 tested on this aarch64 host). Pixi is optional.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

If Pixi is installed:

```bash
pixi install
pixi run test
```

## Commands

```bash
# Correct tiny fixture (Flatten 64, Gemm K=64)
python -m compiler emit-fixture --out fixtures/tiny_cnn.onnx

# Analyze / recommend (allowlist only)
python -m compiler analyze fixtures/tiny_cnn.onnx
python -m compiler recommend fixtures/tiny_cnn.onnx

# ORT CPU compile + benchmark (writes experiments/results/)
python -m compiler compile fixtures/tiny_cnn.onnx --backend ort_cpu --strategy baseline

# Full baseline JSON: ORT CPU always; TensorRT only if an NVIDIA GPU exists
python -m compiler baseline --warmup 10 --iters 50

# Host / ROS / PX4 / Gazebo / NVIDIA facts (no fake success)
python -m compiler probe

# Tests (disable ROS launch_testing plugin autoload)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
# or: bash scripts/run_tests.sh

# YOLOv8n ONNX (optional extra deps)
pip install '.[yolo]'
python scripts/export_yolov8n.py
```

## Backends

- `ort_cpu` — ONNX Runtime `CPUExecutionProvider`. Graph opt level comes from the allowlisted strategy (`baseline` disables fusions).
- `tensorrt` — skipped unless `nvidia-smi` or `/dev/nvidia0` exists **and** the `tensorrt` package imports. This QEMU aarch64 host has a Virtio GPU only.

## Layout

```
compiler/               thesis contribution (parse, analyze, recommend, optimize, history)
src/nnc/backends/       ORT CPU + TensorRT runtime
tests/                  fixture contract, allowlist, ORT compile, history
experiments/results/    JSON run records + history.jsonl
schemas/                run-result.schema.json
ros2_ws/                existing telemetry; not part of the compiler contribution
```
