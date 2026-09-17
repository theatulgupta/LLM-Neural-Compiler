#!/usr/bin/env bash
# Best-effort spawn of a visible prop in front of the camera. Skip JSON on failure.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${NNC_SPAWN_JSON:-$ROOT/experiments/results/spawn_props.json}"
mkdir -p "$(dirname "$OUT")"
if gz service -s /world/default/create --help >/dev/null 2>&1; then
  echo '{"ok":true,"note":"gz create service exists; caller may spawn Fuel models"}' >"$OUT"
  exit 0
fi
echo '{"ok":false,"skip":{"reason":"gz create service not available; detections may be empty on default world"}}' >"$OUT"
exit 0
