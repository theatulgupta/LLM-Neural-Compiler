#!/usr/bin/env bash
# Run a PX4 SITL client command, e.g. scripts/px4_cmd.sh commander takeoff
set -euo pipefail
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
BIN_DIR="$PX4_DIR/build/px4_sitl_default/bin"
if [[ $# -lt 1 ]]; then
  echo "usage: $0 <px4-module> [args...]" >&2
  echo "example: $0 commander takeoff" >&2
  exit 2
fi
MODULE="$1"
shift
CLIENT="$BIN_DIR/px4-${MODULE}"
if [[ ! -x "$CLIENT" ]]; then
  echo "PX4 client missing: $CLIENT" >&2
  exit 1
fi
exec "$CLIENT" "$@"
