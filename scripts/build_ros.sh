#!/usr/bin/env bash
# Build llm_uav_core against the real px4_msgs overlay (not a stub).
set -eo pipefail
set +u
source /opt/ros/jazzy/setup.bash
if [[ -f "$HOME/ros2_px4_ws/install/setup.bash" ]]; then
  source "$HOME/ros2_px4_ws/install/setup.bash"
else
  echo "px4_msgs overlay missing at $HOME/ros2_px4_ws/install/setup.bash" >&2
  exit 1
fi
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/ros2_ws"
set -u
colcon build --symlink-install --packages-select llm_uav_core
set +u
source install/setup.bash
echo "built llm_uav_core"
