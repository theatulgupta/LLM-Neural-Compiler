#!/usr/bin/env bash
# Headless PX4 SITL. Runs the already-built posix binary; does not invoke make
# (make px4_sitl can wipe the build tree if cmake reconfigure fails).
set -euo pipefail
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
BIN="$PX4_DIR/build/px4_sitl_default/bin/px4"
if [[ ! -x "$BIN" ]]; then
  echo "PX4 SITL binary missing at $BIN" >&2
  echo "Build with: cd $PX4_DIR && make -j3 px4_sitl_default" >&2
  exit 1
fi
if pgrep -f 'build/px4_sitl_default/bin/px4' >/dev/null 2>&1; then
  echo "PX4 SITL already running; bash scripts/stop_all.sh first" >&2
  exit 1
fi
export HEADLESS="${HEADLESS:-1}"
export PX4_GZ_STANDALONE="${PX4_GZ_STANDALONE:-1}"
export PX4_GZ_WORLD="${PX4_GZ_WORLD:-nnc_yard}"
export PX4_GZ_MODELS="${PX4_GZ_MODELS:-$PX4_DIR/Tools/simulation/gz/models}"
export PX4_GZ_SIM_RENDER_ENGINE="${PX4_GZ_SIM_RENDER_ENGINE:-${NNC_GZ_RENDER:-ogre2}}"
export PX4_SIM_MODEL="${PX4_SIM_MODEL:-nnc_x500_cam}"
export PX4_SYS_AUTOSTART="${PX4_SYS_AUTOSTART:-4010}"
cd "$PX4_DIR/build/px4_sitl_default"
# Working dir must contain ROMFS relative paths used by posix SITL.
exec "$BIN" -d .
