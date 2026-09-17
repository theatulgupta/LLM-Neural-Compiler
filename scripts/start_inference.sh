#!/usr/bin/env bash
# Run inference_node under system Python (ROS 2) with optional artifact path.
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
MODEL="${NNC_MODEL_PATH:-$ROOT/experiments/models/yolov8n.onnx}"
KIND="${NNC_MODEL_KIND:-yolov8n}"
TASK="${NNC_TASK:-detect}"
SOURCE="${NNC_SOURCE:-camera}"
exec ros2 run llm_uav_core inference_node --ros-args \
  -p "model_path:=${MODEL}" \
  -p "model_kind:=${KIND}" \
  -p "task:=${TASK}" \
  -p "source:=${SOURCE}"
