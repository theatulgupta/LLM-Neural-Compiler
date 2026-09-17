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
export HEADLESS="${HEADLESS:-1}"
export PX4_GZ_STANDALONE="${PX4_GZ_STANDALONE:-1}"
export PX4_GZ_WORLD="${PX4_GZ_WORLD:-default}"
export PX4_GZ_SIM_RENDER_ENGINE="${PX4_GZ_SIM_RENDER_ENGINE:-${NNC_GZ_RENDER:-ogre2}}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export PX4_SIM_MODEL="${PX4_SIM_MODEL:-x500_mono_cam}"
export PX4_SYS_AUTOSTART="${PX4_SYS_AUTOSTART:-4010}"
cd "$PX4_DIR/build/px4_sitl_default"
# Working dir must contain ROMFS relative paths used by posix SITL.
exec "$BIN" -d .
