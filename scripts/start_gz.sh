#!/usr/bin/env bash
# Headless Gazebo Sim (server only) with PX4 model/world paths.
set -euo pipefail
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
WORLD="${PX4_GZ_WORLD:-default}"
RENDER="${NNC_GZ_RENDER:-ogre2}"
export GZ_SIM_RESOURCE_PATH="${PX4_DIR}/Tools/simulation/gz/models:${PX4_DIR}/Tools/simulation/gz/worlds${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
export GZ_SIM_SERVER_CONFIG_PATH="${GZ_SIM_SERVER_CONFIG_PATH:-${PX4_DIR}/Tools/simulation/gz/server.config}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
SDF="${PX4_DIR}/Tools/simulation/gz/worlds/${WORLD}.sdf"
if [[ ! -f "$SDF" ]]; then
  echo "Gazebo world missing: $SDF" >&2
  exit 1
fi
set +u
source /opt/ros/jazzy/setup.bash
set -u
exec gz sim -r -s --headless-rendering --render-engine "$RENDER" "$SDF"
