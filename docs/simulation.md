# Simulation (PX4 + Gazebo + camera)

The compiler does not need Gazebo. This tree uses SITL so the **same**
compiled ONNX can later move to a real companion camera topic.

## What works on this host

Gazebo Harmonic 8.x, PX4 SITL `nnc_x500_cam`, `ros_gz_bridge` Image +
CameraInfo onto `/nnc/camera/image_raw`. World `sim/worlds/nnc_yard.sdf`
has primitive geometry plus Fuel includes (pickup, hatchback, standing
person, casual female, cone). Missing Fuel models are skipped; the world
still loads.

This QEMU aarch64 box has a Virtio GPU. The working camera path is:

- `gz sim -s` **without** `--headless-rendering`
- render engine **ogre** (Ogre 1.x), not ogre2
- `Xvfb :99` + `LIBGL_ALWAYS_SOFTWARE=1` (llvmpipe)

ogre2 + EGL on this GPU segfaults. Do not “fix” that by inventing frames.

## Scripts

| Script | Job |
| --- | --- |
| `scripts/start_all.sh` | XRCE agent, Gazebo server, PX4, camera bridge, telemetry |
| `scripts/start_gui.sh` | optional `gz sim -g` (needs a real `DISPLAY`; never started by `start_all`) |
| `scripts/gz_probe_camera.sh` | one-shot Image on `/nnc/camera/image_raw` |
| `scripts/record_frames.py` | npz calib (+ `--jpeg` for a still) |
| `scripts/sim_matrix.sh` | native then chosen artifact per zoo kind on camera frames |
| `scripts/stop_all.sh` | tear down |

```bash
bash scripts/build_ros.sh
HEADLESS=1 bash scripts/start_all.sh
bash scripts/gz_probe_camera.sh
/usr/bin/python3 scripts/record_frames.py --n 64 --world nnc_yard \
  --airframe nnc_x500_cam --jpeg experiments/results/yard_frame.jpg
NNC_START_INFERENCE=1  # or scripts/sim_matrix.sh after start_all
bash scripts/stop_all.sh
```

`verify_sitl.py` is pose motion, not camera. Camera success is a non-empty
Image and `record_frames.py` writing `experiments/calib/gz_frames.npz`.

## Interactive GUI

On an Ubuntu desktop with Gazebo Harmonic, `scripts/start_gui.sh` attaches
a client to the already-running server. SSH without X forwarding will exit
2 (`DISPLAY` empty). A one-shot `DISPLAY=:0` attach on this QEMU SSH session
aborted in Qt (`createPlatformIntegration`); use a local desktop terminal
instead.
