#!/usr/bin/env bash
# SSH-safe SITL stack: XRCE agent, headless Gazebo, PX4 camera airframe, ROS bridge.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="${NNC_SITL_LOGDIR:-$ROOT/experiments/results/sitl_logs}"
mkdir -p "$LOGDIR"

stop_children() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
}
PIDS=()
trap stop_children EXIT

echo "starting MicroXRCEAgent" | tee "$LOGDIR/stack.log"
"$ROOT/scripts/start_agent.sh" >"$LOGDIR/agent.log" 2>&1 &
PIDS+=($!)
sleep 2

echo "starting Gazebo headless (${NNC_GZ_RENDER:-ogre2})" | tee -a "$LOGDIR/stack.log"
"$ROOT/scripts/start_gz.sh" >"$LOGDIR/gz.log" 2>&1 &
PIDS+=($!)
sleep 3

echo "starting PX4 SITL ${NNC_PX4_TARGET:-gz_x500_mono_cam} (PX4_GZ_STANDALONE=1)" | tee -a "$LOGDIR/stack.log"
HEADLESS=1 PX4_GZ_STANDALONE=1 "$ROOT/scripts/start_px4.sh" >"$LOGDIR/px4.log" 2>&1 &
PIDS+=($!)

echo "waiting for PX4 (up to 120s)" | tee -a "$LOGDIR/stack.log"
for _ in $(seq 1 120); do
  if grep -q "Ready for takeoff\|INFO  \[init\] Gazebo world is ready\|uxrce_dds_client" "$LOGDIR/px4.log" 2>/dev/null; then
    break
  fi
  sleep 1
done

echo "starting camera bridge" | tee -a "$LOGDIR/stack.log"
"$ROOT/scripts/start_camera_bridge.sh" >"$LOGDIR/camera_bridge.log" 2>&1 &
PIDS+=($!)

echo "starting telemetry_node" | tee -a "$LOGDIR/stack.log"
"$ROOT/scripts/start_ros.sh" >"$LOGDIR/telemetry.log" 2>&1 &
PIDS+=($!)

if [[ -x "$ROOT/scripts/start_inference.sh" ]] && [[ "${NNC_START_INFERENCE:-0}" == "1" ]]; then
  echo "starting inference_node" | tee -a "$LOGDIR/stack.log"
  "$ROOT/scripts/start_inference.sh" >"$LOGDIR/inference.log" 2>&1 &
  PIDS+=($!)
fi

echo "stack pids=${PIDS[*]} logs=$LOGDIR" | tee -a "$LOGDIR/stack.log"
if [[ "${NNC_SITL_WAIT:-0}" == "1" ]]; then
  wait
else
  trap - EXIT
  echo "${PIDS[@]}" >"$LOGDIR/pids.txt"
fi
