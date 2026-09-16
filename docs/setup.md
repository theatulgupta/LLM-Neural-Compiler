# Setup

## Python compiler

```bash
cd /home/theatulgupta/LLM-Neural-Compiler   # or your clone
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m compiler emit-fixture --out fixtures/tiny_cnn.onnx
python -m compiler baseline
```

Optional Pixi (reinstall if `~/.pixi/bin/pixi` is an empty file):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
pixi install
pixi run test
```

## YOLOv8n

```bash
pip install '.[yolo]'
python scripts/export_yolov8n.py --out experiments/models/yolov8n.onnx
```

Weights and ONNX are gitignored (`*.pt`, `*.onnx`). The export script prints
the command and SHA-256 when it succeeds.

## ROS 2 / PX4 / Gazebo

This machine may already have ROS 2 Jazzy and Gazebo Harmonic (`gz sim`).
PX4 SITL is **not** assumed built. `scripts/start_px4.sh` points at
`~/LLM-Neural-Compiler/third_party/PX4-Autopilot`, while the checkout on this
host is `~/PX4-Autopilot` and has no `build/` directory until you compile SITL.

Do not treat missing SITL as a passing telemetry run. Use:

```bash
python scripts/verify_sitl.py
python -m compiler probe
```
