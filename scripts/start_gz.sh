#!/usr/bin/env bash
# Gazebo Sim server for the NNC yard world. Display modes:
#   auto  — use $DISPLAY if set, else start Xvfb :99 (software GL, no EGL)
#   xvfb  — start/use Xvfb, LIBGL_ALWAYS_SOFTWARE=1, no --headless-rendering
#   x11   — existing $DISPLAY, software GL, no --headless-rendering
#   egl   — --headless-rendering, no LIBGL_ALWAYS_SOFTWARE (real GPU hosts)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/PX4-Autopilot}"
WORLD="${PX4_GZ_WORLD:-nnc_yard}"
RENDER="${NNC_GZ_RENDER:-ogre}"
MODE="${NNC_GZ_DISPLAY:-auto}"
LOGDIR="${NNC_SITL_LOGDIR:-$ROOT/experiments/results/sitl_logs}"
mkdir -p "$LOGDIR"

export GZ_SIM_RESOURCE_PATH="${ROOT}/sim/models:${ROOT}/sim/worlds:${PX4_DIR}/Tools/simulation/gz/models:${PX4_DIR}/Tools/simulation/gz/worlds${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="${PX4_DIR}/build/px4_sitl_default/src/modules/simulation/gz_plugins${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"

TEMPLATE="$ROOT/sim/server.config"
RUNTIME_CFG="$LOGDIR/server.config"
sed "s/@RENDER_ENGINE@/${RENDER}/g" "$TEMPLATE" > "$RUNTIME_CFG"
export GZ_SIM_SERVER_CONFIG_PATH="$RUNTIME_CFG"

if [[ "$MODE" == "auto" ]]; then
  sock=""
  if [[ -n "${DISPLAY:-}" ]]; then
    sock="/tmp/.X11-unix/X${DISPLAY#:}"
  fi
  if [[ -n "${DISPLAY:-}" ]] && { pgrep -f "Xvfb ${DISPLAY}" >/dev/null 2>&1 || [[ -S "$sock" ]]; }; then
    if pgrep -f "Xvfb ${DISPLAY}" >/dev/null 2>&1; then
      MODE="xvfb"
    else
      MODE="x11"
    fi
  else
    MODE="xvfb"
    unset DISPLAY || true
  fi
fi

HEADLESS_FLAG=()
case "$MODE" in
  xvfb)
    export DISPLAY="${DISPLAY:-:99}"
    if ! pgrep -f "Xvfb ${DISPLAY}" >/dev/null 2>&1; then
      Xvfb "$DISPLAY" -screen 0 1280x1024x24 -nolisten tcp >"$LOGDIR/xvfb.log" 2>&1 &
      echo $! > "$LOGDIR/xvfb.pid"
      sleep 0.4
    fi
    export LIBGL_ALWAYS_SOFTWARE=1
    ;;
  x11)
    if [[ -z "${DISPLAY:-}" ]]; then
      echo "NNC_GZ_DISPLAY=x11 requires DISPLAY" >&2
      exit 1
    fi
    export LIBGL_ALWAYS_SOFTWARE=1
    ;;
  egl)
    unset LIBGL_ALWAYS_SOFTWARE || true
    HEADLESS_FLAG=(--headless-rendering)
    ;;
  *)
    echo "unknown NNC_GZ_DISPLAY=$MODE (auto|xvfb|x11|egl)" >&2
    exit 1
    ;;
esac

SDF="${ROOT}/sim/worlds/${WORLD}.sdf"
if [[ ! -f "$SDF" ]]; then
  SDF="${PX4_DIR}/Tools/simulation/gz/worlds/${WORLD}.sdf"
fi
if [[ ! -f "$SDF" ]]; then
  echo "Gazebo world missing: $WORLD" >&2
  exit 1
fi

echo "gz display_mode=$MODE render=$RENDER world=$SDF" | tee -a "$LOGDIR/stack.log"
set +u
source /opt/ros/jazzy/setup.bash
set -u
exec gz sim -r -s -v 3 --render-engine "$RENDER" "${HEADLESS_FLAG[@]}" "$SDF"
