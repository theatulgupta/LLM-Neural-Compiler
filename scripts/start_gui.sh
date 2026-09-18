#!/usr/bin/env bash
# Attach a Gazebo GUI client to a running `scripts/start_gz.sh` server.
# Not used by start_all.sh. Run this from a desktop terminal that has DISPLAY.
set -euo pipefail
if [[ -z "${DISPLAY:-}" ]]; then
  echo "start_gui.sh needs DISPLAY (use a desktop terminal, not SSH)." >&2
  exit 2
fi
RENDER="${NNC_GZ_GUI_RENDER:-ogre}"
set +u
source /opt/ros/jazzy/setup.bash
set -u
echo "gz sim -g --render-engine-gui $RENDER"
exec gz sim -g --render-engine-gui "$RENDER"
