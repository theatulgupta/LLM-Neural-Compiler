#!/usr/bin/env bash
# Probe Gazebo camera image without PX4. Writes experiments/results/sim_camera_probe.json.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="${NNC_SITL_LOGDIR:-$ROOT/experiments/results/sitl_logs}"
OUT="${NNC_CAMERA_PROBE:-$ROOT/experiments/results/sim_camera_probe.json}"
mkdir -p "$LOGDIR" "$(dirname "$OUT")"
set +u
source /opt/ros/jazzy/setup.bash
set -u

write_json() {
  python3 - "$OUT" <<'PY'
import json, sys
path = sys.argv[1]
payload = json.loads(sys.stdin.read())
path_obj = __import__("pathlib").Path(path)
path_obj.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

stop_gz() {
  "$ROOT/scripts/stop_all.sh" >/dev/null 2>&1 || true
  pkill -f 'gz sim' 2>/dev/null || true
  sleep 1
}

try_one() {
  local render="$1"
  local mode="$2"
  local label="${render}+${mode}"
  stop_gz
  echo "probe try $label" >&2
  NNC_GZ_RENDER="$render" NNC_GZ_DISPLAY="$mode" PX4_GZ_WORLD=nnc_yard \
    "$ROOT/scripts/start_gz.sh" >"$LOGDIR/gz_probe_${render}_${mode}.log" 2>&1 &
  local gz_pid=$!
  local topic=""
  local i
  for i in $(seq 1 45); do
    if ! kill -0 "$gz_pid" 2>/dev/null; then
      echo "gz exited early for $label" >&2
      break
    fi
    if gz topic -l 2>/dev/null | grep -q '/clock$'; then
      break
    fi
    sleep 1
  done
  local sdf="$ROOT/sim/models/nnc_x500_cam/model.sdf"
  gz service -s /world/nnc_yard/create --reqtype gz.msgs.EntityFactory \
    --reptype gz.msgs.Boolean --timeout 8000 \
    --req "name: \"nnc_x500_cam_0\", allow_renaming: false, sdf: '<sdf version=\"1.6\"> <include> <uri>file://${sdf}</uri> </include> </sdf>'" \
    >/dev/null 2>&1 || true
  for i in $(seq 1 60); do
    topic="$(gz topic -l 2>/dev/null | grep -m1 '/sensor/camera/image$' || true)"
    if [[ -n "$topic" ]]; then
      break
    fi
    sleep 1
  done
  local gl=""
  if command -v glxinfo >/dev/null 2>&1 && [[ -n "${DISPLAY:-}" || "$mode" != "egl" ]]; then
    gl="$(DISPLAY="${DISPLAY:-:99}" LIBGL_ALWAYS_SOFTWARE=1 glxinfo -B 2>/dev/null | grep 'OpenGL renderer' | head -1 || true)"
  fi
  if [[ -n "$topic" ]]; then
    local echo_out
    echo_out="$(timeout 8 gz topic -e -t "$topic" -n 1 2>/dev/null || true)"
    local w h
    w="$(printf '%s\n' "$echo_out" | awk '/^width:/{print $2; exit}')"
    h="$(printf '%s\n' "$echo_out" | awk '/^height:/{print $2; exit}')"
    echo "${label}|${topic}|${w:-0}|${h:-0}|${gl}|ok"
    return 0
  fi
  local err
  err="$(tail -n 8 "$LOGDIR/gz_probe_${render}_${mode}.log" 2>/dev/null | tr '\n' ' ' | cut -c1-400)"
  echo "${label}||0|0|${gl}|fail:${err}"
  return 1
}

tried=()
ok_line=""
# Order from the master plan: ogre2+xvfb, ogre+xvfb, ogre2+egl (expected fail).
for pair in "ogre2 xvfb" "ogre xvfb" "ogre2 egl"; do
  set -- $pair
  result="$(try_one "$1" "$2" || true)"
  tried+=("$result")
  if [[ "$result" == *"|ok" ]]; then
    ok_line="$result"
    break
  fi
done
stop_gz

python3 - "$OUT" "${tried[@]}" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
tried_raw = sys.argv[2:]
tried = []
ok = False
topic = None
width = None
height = None
display_mode = None
render_engine = None
gl_renderer = None
for raw in tried_raw:
    parts = raw.split("|")
    while len(parts) < 6:
        parts.append("")
    label, tpc, w, h, gl, status = parts[0], parts[1], parts[2], parts[3], parts[4], "|".join(parts[5:])
    render, mode = (label.split("+", 1) + [""])[:2]
    entry = {
        "label": label,
        "render_engine": render,
        "display_mode": mode,
        "topic": tpc or None,
        "width": int(w) if w.isdigit() else None,
        "height": int(h) if h.isdigit() else None,
        "gl_renderer": gl or None,
        "status": status,
    }
    tried.append(entry)
    if status == "ok" and not ok:
        ok = True
        topic = tpc
        width = entry["width"]
        height = entry["height"]
        display_mode = mode
        render_engine = render
        gl_renderer = gl or None
payload = {
    "ok": bool(ok and width == 640 and height == 480),
    "display_mode": display_mode,
    "render_engine": render_engine,
    "topic": topic,
    "width": width,
    "height": height,
    "tried": tried,
    "gl_renderer": gl_renderer,
    "world": "nnc_yard",
    "airframe": "nnc_x500_cam",
    "fps_claimed": False,
}
if ok and not payload["ok"]:
    payload["reason"] = f"image topic appeared but size was {width}x{height}, expected 640x480"
elif not ok:
    payload["reason"] = "Gazebo camera image topic never appeared"
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
sys.exit(0 if payload["ok"] else 3)
PY
