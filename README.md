# LLM-Neural-Compiler

LLM-guided neural compilation for real-time UAV edge inference. The thesis
contribution lives in `compiler/`: ONNX graph analysis, a schema-bound LLM
that names an allowlisted **pass set**, a transformation engine that rewrites
the DAG, and measured compile/latency history. Runtime backends live in
`src/nnc/backends/` (ONNX Runtime CPU; TensorRT skip-with-reason).

This tree does **not** add UAV-agent stubs. The existing ROS 2 telemetry node
and empty planning/control modules stay as they are. `inference_node` only
loads an ORT artifact and publishes measured latency.

## Models

The UAV companion zoo lives in `experiments/zoo.yaml` (loaded by `compiler.catalog`).
The compiler pipeline does not switch on YOLOv8. Each model is a YAML record plus
an export script; analyze / recommend / compile / matrix use the same CLI.

| Kind | Task | Why a real companion computer runs it |
| --- | --- | --- |
| Fixture: tiny CNN `1×1×8×8` | tests | Flatten width **64**; Gemm `K` must be 64. `K=16` is the known-broken regression. |
| Fixture: tiny depth `1×3×8×8` | tests | RGB-like map to 1-channel depth. No download. |
| `yolov8n` | detect | People/vehicle detect on Jetson and Pi-class PX4 companions. |
| `yolo11n` | detect | Newer Ultralytics nano detector (not a second copy of YOLOv8). |
| `yolov8n-pose` | pose | Person keypoints for search-and-rescue / follow-me. |
| `yolov8n-seg` | segment | Masks for landing-zone / trail. |
| `ssdlite_mobilenetv3` | detect-lite | Classic 320² SSD-MobileNet stack on Raspberry Pi / older Jetson. |
| `mobilenetv3_small` | classify | Landing-pad / gate / sign ID. Smaller than ResNet18. |
| `midas_small` | depth | Monocular depth cue for sense-and-avoid when stereo is not on the companion. |

ONNX weights are gitignored. Reproduce:

```bash
pip install -e '.[yolo]'   # ultralytics + torch
pip install timm           # only for MiDaS
python -m compiler zoo
python -m compiler export              # or --kind yolov8n
python -m compiler matrix --warmup 3 --iters 8
```

Native = unrewritten ONNX (`baseline`, ORT graph opt off). Optimized = schema-bound
advisor **graph pass set**, compiled and measured on the **same** aarch64 host.
`docs/paper_proposal.md` is the proposal write-up. Numbers come only from
`experiments/results/paper_matrix.json`. Do not compare this QEMU box to cloud x86.

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

# Analyze / recommend (allowlist + schemas/llm-proposal.schema.json)
python -m compiler analyze fixtures/tiny_cnn.onnx
python -m compiler recommend fixtures/tiny_cnn.onnx

# Load the ORT artifact and measure real latency (no invented FPS)
python -m compiler infer fixtures/tiny_cnn.onnx --graph-opt disable

# ORT CPU compile + benchmark (writes experiments/results/)
python -m compiler compile fixtures/tiny_cnn.onnx --backend ort_cpu --strategy baseline

# Full baseline JSON: fixture + every zoo ONNX that exists; TensorRT skip-with-reason
python -m compiler baseline --warmup 10 --iters 50

# Host / ROS / PX4 / Gazebo / NVIDIA facts (no fake success)
python -m compiler probe

# Tests (disable ROS launch_testing plugin autoload)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests
# or: bash scripts/run_tests.sh

# UAV zoo ONNX (optional extra deps). Failures write skip JSON, not invented latency.
pip install -e '.[yolo]'
python -m compiler export --kind yolov8n
python -m compiler matrix --kind yolov8n --warmup 3 --iters 8
```

## SITL (this machine)

Scripts use **real paths**: `~/PX4-Autopilot`, `~/px4_ros_uxrce_dds_ws`,
`~/ros2_px4_ws`. They do not use `gnome-terminal` or gitignored `third_party/`.

```bash
bash scripts/build_ros.sh
HEADLESS=1 bash scripts/start_all.sh
python scripts/verify_sitl.py
```

`verify_sitl.py` exits 0 only if x,y,z change on the live local-position topic.

## Backends

- `ort_cpu` — ONNX Runtime `CPUExecutionProvider`. Graph opt level comes from the allowlisted strategy (`baseline` disables fusions).
- `tensorrt` — skipped unless `nvidia-smi` or `/dev/nvidia0` exists **and** the `tensorrt` package imports. This QEMU aarch64 host has a Virtio GPU only.

## Layout

```
compiler/               thesis contribution (parse, analyze, recommend, optimize, history)
src/nnc/backends/       ORT CPU + TensorRT runtime
src/nnc/artifact.py     load ORT ONNX and time a real inference
tests/                  fixture contract, allowlist, ORT compile, history, LLM schema
experiments/results/    JSON run records + history.jsonl
schemas/                run-result.schema.json, llm-proposal.schema.json
ros2_ws/                telemetry + inference_node; empty agent modules stay empty
```
