#!/usr/bin/env bash
# SSH-safe SITL stack. No gnome-terminal. Logs to experiments/results/sitl_logs/.
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

echo "starting PX4 SITL gz_x500 (HEADLESS=1)" | tee -a "$LOGDIR/stack.log"
HEADLESS=1 "$ROOT/scripts/start_px4.sh" >"$LOGDIR/px4.log" 2>&1 &
PIDS+=($!)

echo "waiting for PX4 / gz (up to 90s)" | tee -a "$LOGDIR/stack.log"
for _ in $(seq 1 90); do
  if grep -q "Ready for takeoff\|INFO  \[init\] Gazebo world is ready\|uxrce_dds_client" "$LOGDIR/px4.log" 2>/dev/null; then
    break
  fi
  sleep 1
done

echo "starting telemetry_node" | tee -a "$LOGDIR/stack.log"
"$ROOT/scripts/start_ros.sh" >"$LOGDIR/telemetry.log" 2>&1 &
PIDS+=($!)

echo "stack pids=${PIDS[*]} logs=$LOGDIR" | tee -a "$LOGDIR/stack.log"
if [[ "${NNC_SITL_WAIT:-0}" == "1" ]]; then
  wait
else
  trap - EXIT
  echo "${PIDS[@]}" >"$LOGDIR/pids.txt"
fi
