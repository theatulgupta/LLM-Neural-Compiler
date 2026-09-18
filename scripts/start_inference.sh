#!/usr/bin/env bash
# Run inference_node under system Python (ROS 2).
# NNC_MODEL_KIND selects the zoo row. NNC_ARTIFACT=native|chosen picks the ONNX.
set -eo pipefail
set +u
source /opt/ros/jazzy/setup.bash
if [[ -f "$HOME/ros2_px4_ws/install/setup.bash" ]]; then
  source "$HOME/ros2_px4_ws/install/setup.bash"
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ ! -f "$ROOT/ros2_ws/install/setup.bash" ]]; then
  echo "llm_uav_core is not built. Run scripts/build_ros.sh" >&2
  exit 1
fi
source "$ROOT/ros2_ws/install/setup.bash"
set -u
KIND="${NNC_MODEL_KIND:-yolov8n}"
ARTIFACT="${NNC_ARTIFACT:-native}"
PYTHON="${NNC_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="${PYTHON:-python3}"
fi

read_spec() {
  "$PYTHON" - "$KIND" "$ARTIFACT" "$ROOT" <<'PY'
import json, sys
from pathlib import Path
kind, artifact, root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, str(root))
from compiler.catalog import get_model
spec = get_model(kind)
task = spec.task
imgsz = spec.imgsz
native = spec.onnx_path(root)
model = native
if artifact == "chosen":
    opt = root / "experiments" / "results" / f"optimize_{kind.replace('/', '_')}.json"
    if not opt.is_file():
        print(f"chosen artifact missing: {opt}", file=sys.stderr)
        sys.exit(2)
    payload = json.loads(opt.read_text(encoding="utf-8"))
    chosen = payload.get("chosen") or {}
    path = ((chosen.get("artifact") or {}).get("path"))
    if not path:
        print(f"chosen.artifact.path missing in {opt}", file=sys.stderr)
        sys.exit(2)
    model = Path(path)
if not Path(model).is_file():
    print(f"model file missing: {model}", file=sys.stderr)
    sys.exit(2)
print(f"{model}\t{task}\t{imgsz}")
PY
}

if [[ -n "${NNC_MODEL_PATH:-}" ]]; then
  MODEL="$NNC_MODEL_PATH"
  TASK="${NNC_TASK:-detect}"
  IMGSZ="${NNC_IMGSZ:-640}"
else
  LINE="$(read_spec)"
  MODEL="$(printf '%s' "$LINE" | cut -f1)"
  TASK="$(printf '%s' "$LINE" | cut -f2)"
  IMGSZ="$(printf '%s' "$LINE" | cut -f3)"
fi
SOURCE="${NNC_SOURCE:-camera}"
exec ros2 run llm_uav_core inference_node --ros-args \
  -p "model_path:=${MODEL}" \
  -p "model_kind:=${KIND}" \
  -p "task:=${TASK}" \
  -p "imgsz:=${IMGSZ}" \
  -p "source:=${SOURCE}"
