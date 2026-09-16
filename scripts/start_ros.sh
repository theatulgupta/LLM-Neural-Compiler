#!/usr/bin/env bash
# Telemetry against real px4_msgs + this workspace's llm_uav_core.
set -euo pipefail
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
exec ros2 run llm_uav_core telemetry_node
