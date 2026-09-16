#!/usr/bin/env bash
source /opt/ros/jazzy/setup.bash
if [[ -f "$HOME/ros2_px4_ws/install/setup.bash" ]]; then
  source "$HOME/ros2_px4_ws/install/setup.bash"
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -f "$ROOT/ros2_ws/install/setup.bash" ]]; then
  source "$ROOT/ros2_ws/install/setup.bash"
fi
if [[ "${1:-}" == "inference" ]]; then
  exec ros2 run llm_uav_core inference_node
fi
exec ros2 run llm_uav_core telemetry_node
