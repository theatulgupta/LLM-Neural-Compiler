#!/usr/bin/env bash
# Pack chosen ONNX artifacts + the ROS inference path for a later companion.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
DEST="$ROOT/dist"
STAGE="$DEST/nnc_bundle_$SHA"
rm -rf "$STAGE"
mkdir -p "$STAGE/experiments/artifacts" "$STAGE/experiments/results" "$STAGE/src" "$STAGE/ros2_ws/src" "$STAGE/scripts"

python3 - "$ROOT" "$STAGE" <<'PY'
import json, shutil, sys
from pathlib import Path
root = Path(sys.argv[1])
stage = Path(sys.argv[2])
results = root / "experiments" / "results"
copied = []
for opt in sorted(results.glob("optimize_*.json")):
    payload = json.loads(opt.read_text(encoding="utf-8"))
    chosen = payload.get("chosen") or {}
    art = chosen.get("artifact") or {}
    path = art.get("path")
    if not path:
        continue
    src = Path(path)
    if not src.is_file():
        continue
    kind = payload.get("kind") or src.parent.name
    dest_dir = stage / "experiments" / "artifacts" / kind
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    shutil.copy2(src, dest)
    sidecar = Path(str(src) + ".json")
    if sidecar.is_file():
        shutil.copy2(sidecar, dest_dir / sidecar.name)
    shutil.copy2(opt, stage / "experiments" / "results" / opt.name)
    copied.append(str(dest.relative_to(stage)))
(stage / "MANIFEST.txt").write_text("\n".join(copied) + "\n", encoding="utf-8")
print(f"copied {len(copied)} artifacts")
PY

cp "$ROOT/experiments/zoo.yaml" "$STAGE/experiments/zoo.yaml"
cp -a "$ROOT/src/nnc" "$STAGE/src/nnc"
cp -a "$ROOT/ros2_ws/src/llm_uav_core" "$STAGE/ros2_ws/src/llm_uav_core"
cp "$ROOT/scripts/start_inference.sh" "$STAGE/scripts/start_inference.sh"
cp "$ROOT/scripts/start_camera_bridge.sh" "$STAGE/scripts/start_camera_bridge.sh"
mkdir -p "$STAGE"
cp "$ROOT/docs/deploy.md" "$STAGE/docs_deploy.md"

(
  cd "$STAGE"
  find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256
)
mkdir -p "$DEST"
tar -C "$DEST" -czf "$DEST/nnc_bundle_${SHA}.tar.gz" "nnc_bundle_$SHA"
echo "wrote $DEST/nnc_bundle_${SHA}.tar.gz"
