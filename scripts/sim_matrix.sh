#!/usr/bin/env bash
# Camera-source inference for every zoo kind, native then chosen. One model at a time.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PYTHON="${NNC_PYTHON:-$ROOT/.venv/bin/python}"
RESULTS="$ROOT/experiments/results"
set +u
source /opt/ros/jazzy/setup.bash
if [[ -f "$HOME/ros2_px4_ws/install/setup.bash" ]]; then
  source "$HOME/ros2_px4_ws/install/setup.bash"
fi
source "$ROOT/ros2_ws/install/setup.bash"
set -u

kinds="$("$PYTHON" - <<'PY'
from compiler.catalog import zoo_kinds
print(" ".join(zoo_kinds()))
PY
)"

note="llvmpipe and ORT share 6 vCPUs; one model at a time. fps_claimed=false."
for kind in $kinds; do
  for variant in native chosen; do
    out="$RESULTS/sim_inference_${kind}_${variant}.json"
    echo "sim_matrix $kind $variant"
    NNC_MODEL_KIND="$kind" NNC_ARTIFACT="$variant" NNC_SOURCE=camera \
      "$ROOT/scripts/start_inference.sh" >"$RESULTS/sitl_logs/inference_${kind}_${variant}.log" 2>&1 &
    node_pid=$!
    sleep 3
    if ! kill -0 "$node_pid" 2>/dev/null; then
      python3 - "$out" "$kind" "$variant" "$note" <<'PY'
import json, sys
from pathlib import Path
out, kind, variant, note = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
payload = {
    "ok": False,
    "kind": kind,
    "variant": variant,
    "frames_rx": 0,
    "inferences": 0,
    "error": "inference_node exited before probe",
    "note": note,
    "fps_claimed": False,
}
Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
      sleep 3
      continue
    fi
    /usr/bin/python3 "$ROOT/scripts/verify_sim_inference.py" --seconds 40 --out "$out" || true
    python3 - "$out" "$kind" "$variant" "$note" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
payload["kind"] = sys.argv[2]
payload["variant"] = sys.argv[3]
payload["note"] = sys.argv[4]
payload["fps_claimed"] = False
if int(payload.get("frames_rx") or 0) <= 0:
    payload["ok"] = False
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({k: payload.get(k) for k in ("ok", "kind", "variant", "frames_rx", "inferences")}, sort_keys=True))
PY
    kill "$node_pid" 2>/dev/null || true
    pkill -f "inference_node" 2>/dev/null || true
    sleep 3
  done
done
echo "sim_matrix done"
