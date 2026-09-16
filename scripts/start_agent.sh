#!/usr/bin/env bash
# Micro XRCE-DDS Agent from the workspace that actually exists on this machine.
set -euo pipefail
AGENT="${MICRO_XRCE_AGENT:-$HOME/px4_ros_uxrce_dds_ws/install/microxrcedds_agent/bin/MicroXRCEAgent}"
if [[ ! -x "$AGENT" ]]; then
  echo "MicroXRCEAgent missing at $AGENT" >&2
  exit 1
fi
exec "$AGENT" udp4 -p "${PX4_UXRCE_DDS_PORT:-8888}"
