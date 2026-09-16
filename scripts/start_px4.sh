#!/usr/bin/env bash
# Headless PX4 SITL (gz_x500). Uses ~/PX4-Autopilot, not the gitignored third_party/ path.
set -euo pipefail
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
BIN="$PX4_DIR/build/px4_sitl_default/bin/px4"
if [[ ! -x "$BIN" ]]; then
  echo "PX4 SITL binary missing at $BIN" >&2
  echo "Build with: cd $PX4_DIR && make -j3 px4_sitl_default" >&2
  exit 1
fi
cd "$PX4_DIR"
export HEADLESS="${HEADLESS:-1}"
export PX4_GZ_WORLD="${PX4_GZ_WORLD:-default}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
exec make px4_sitl gz_x500
