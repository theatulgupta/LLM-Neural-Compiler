#!/usr/bin/env bash
# Stop the SITL stack started by start_all.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="${NNC_SITL_LOGDIR:-$ROOT/experiments/results/sitl_logs}"
PIDS_FILE="$LOGDIR/pids.txt"
if [[ -f "$PIDS_FILE" ]]; then
  for pid in $(cat "$PIDS_FILE"); do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
fi
if [[ -f "$LOGDIR/xvfb.pid" ]]; then
  kill "$(cat "$LOGDIR/xvfb.pid")" 2>/dev/null || true
  rm -f "$LOGDIR/xvfb.pid"
fi
pkill -f 'gz sim' 2>/dev/null || true
pkill -f 'build/px4_sitl_default/bin/px4' 2>/dev/null || true
pkill -f MicroXRCEAgent 2>/dev/null || true
pkill -f parameter_bridge 2>/dev/null || true
pkill -f 'llm_uav_core' 2>/dev/null || true
pkill -f 'Xvfb :99' 2>/dev/null || true
sleep 1
echo "stopped sitl stack"
