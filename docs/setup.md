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

## UAV model zoo

```bash
pip install -e '.[yolo]'
pip install timm          # MiDaS only
python -m compiler zoo
python -m compiler export --kind yolov8n
python -m compiler export --kind yolo11n
python -m compiler export --kind yolov8n-pose
python -m compiler export --kind yolov8n-seg
python scripts/export_ssdlite.py
python scripts/export_mobilenetv3_small.py
python scripts/export_midas_small.py
python -m compiler matrix --warmup 3 --iters 8
```

Weights and ONNX are gitignored (`*.pt`, `*.onnx`). Export prints SHA-256 on
success, or writes `experiments/results/skip_<kind>.json` on failure. Do not
invent latency. See `docs/paper_proposal.md`.

## ROS 2 / PX4 / Gazebo (this Ubuntu 24 aarch64 host)

Real paths (do **not** use the gitignored `third_party/` trees):

| Piece | Path |
| --- | --- |
| PX4 | `~/PX4-Autopilot` (built `px4_sitl_default` binary; `scripts/start_px4.sh` runs it, does not `make`) |
| Micro XRCE-DDS Agent | `~/px4_ros_uxrce_dds_ws/install/microxrcedds_agent/bin/MicroXRCEAgent` |
| px4_msgs | `~/ros2_px4_ws` (source `install/setup.bash` before `colcon` / `ros2`) |
| llm_uav_core | `ros2_ws/` via `scripts/build_ros.sh` |

SSH-safe stack (no `gnome-terminal`):

```bash
bash scripts/build_ros.sh
bash scripts/start_all.sh          # agent + gz server + PX4 nnc_x500_cam + camera bridge + telemetry
/usr/bin/python3 scripts/record_frames.py --n 64
python scripts/verify_sitl.py      # success only if x,y,z actually change
NNC_START_INFERENCE=1 bash scripts/start_all.sh
/usr/bin/python3 scripts/verify_sim_inference.py --seconds 60
bash scripts/stop_all.sh
python -m compiler probe
```

Do not treat a built `bin/px4` as a passing telemetry run. `verify_sitl.py`
must see changing `x,y,z` on `/fmu/out/vehicle_local_position_v1` (or the
live `vehicle_local_position*` topic).

## Another machine

```bash
git clone -b llm-advisor git@github.com:theatulgupta/LLM-Neural-Compiler.git
cd LLM-Neural-Compiler
bash scripts/bootstrap.sh
mkdir -p ~/.config/nnc
scp <this-host>:~/.config/nnc/llm.env ~/.config/nnc/llm.env
```

PX4, ROS 2 Jazzy, and Micro XRCE-DDS are OS installs (`~/PX4-Autopilot`,
`~/ros2_px4_ws`, `~/px4_ros_uxrce_dds_ws`). The repo `sim/` tree only needs
`GZ_SIM_RESOURCE_PATH` to also include `$PX4_DIR/Tools/simulation/gz/models`
so `model://x500` resolves. Camera loop:

```bash
bash scripts/gz_probe_camera.sh
bash scripts/start_all.sh
/usr/bin/python3 scripts/record_frames.py --n 64 --world nnc_yard --airframe nnc_x500_cam
```

See `docs/deploy.md` for packing the same ONNX onto a later companion.
HLD: `docs/architecture.md`. LLD: `docs/design.md`.

## LLM keys (never git)

```bash
mkdir -p ~/.config/nnc
cp configs/llm.env.example ~/.config/nnc/llm.env
# set NNC_LLM and the matching key; Groq is optional
# NNC_LLM=groq | openai | openrouter | ollama | custom | …
```

`python -m compiler formats` prints `llm_providers`. Tests pin `NNC_LLM=heuristic`.

