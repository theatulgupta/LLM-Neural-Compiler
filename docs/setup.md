# Setup

## Python compiler

```bash
cd /home/theatulgupta/LLM-Neural-Compiler   # or your clone
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
python -m compiler emit-fixture --out fixtures/tiny_cnn.onnx
python -m compiler baseline
python -m compiler infer fixtures/tiny_cnn.onnx --graph-opt disable
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
python -m compiler infer experiments/models/yolov8n.onnx --graph-opt extended --warmup 0 --iters 1
```

Weights and ONNX are gitignored (`*.pt`, `*.onnx`). The export script prints
the command and SHA-256 when it succeeds.

## ROS 2 / PX4 / Gazebo (this Ubuntu 24 aarch64 host)

Real paths (do **not** use the gitignored `third_party/` trees):

| Piece | Path |
| --- | --- |
| PX4 | `~/PX4-Autopilot` (`make px4_sitl_default`, then `HEADLESS=1 make px4_sitl gz_x500`) |
| Micro XRCE-DDS Agent | `~/px4_ros_uxrce_dds_ws/install/microxrcedds_agent/bin/MicroXRCEAgent` |
| px4_msgs | `~/ros2_px4_ws` (source `install/setup.bash` before `colcon` / `ros2`) |
| llm_uav_core | `ros2_ws/` via `scripts/build_ros.sh` |

SSH-safe stack (no `gnome-terminal`):

```bash
bash scripts/build_ros.sh
bash scripts/start_all.sh          # agent + PX4 gz_x500 + telemetry, logs in experiments/results/sitl_logs/
python scripts/verify_sitl.py      # success only if x,y,z actually change
python -m compiler probe
```

Do not treat a built `bin/px4` as a passing telemetry run. `verify_sitl.py`
must see changing `x,y,z` on `/fmu/out/vehicle_local_position_v1` (or the
live `vehicle_local_position*` topic).
