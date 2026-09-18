# Deploy the same compiled ONNX to a companion computer

The mentor path is sim first, then the **same** rewritten ONNX on the drone.
`scripts/deploy_bundle.sh` packs every chosen artifact plus the ROS inference
node. TensorRT, board power, and COCO mAP stay **not measured** until that
board exists.

## Targets

- NVIDIA Jetson Orin Nano (TensorRT can leave skip-with-reason)
- Raspberry Pi 5 (ORT CPU only)

## On the board

```bash
sudo apt-get install -y python3-pip python3-venv
# ROS 2 Jazzy + a camera node (v4l2_camera or usb_cam)
python3 -m venv .venv && source .venv/bin/activate
pip install onnxruntime
tar -tzf nnc_bundle_*.tar.gz
# place artifacts, then:
python -m compiler probe
```

Remap the board camera onto the same topic the sim uses:

```bash
ros2 run v4l2_camera v4l2_camera_node --ros-args -r image_raw:=/nnc/camera/image_raw
# or usb_cam: -r image_raw:=/nnc/camera/image_raw
```

`inference_node` is unchanged. Point it at the bundled artifact:

```bash
NNC_MODEL_KIND=yolov8n NNC_ARTIFACT=chosen bash scripts/start_inference.sh
```

On Jetson, `python -m compiler compile experiments/artifacts/yolov8n/<plan>.onnx --backend tensorrt`
stops skipping when `nvidia-smi` and the TensorRT python package exist. Write
board JSON under `experiments/results/<board>/` (`--results`).

## Still not measured here

- TensorRT latency
- Board power
- COCO mAP on flight footage
- GPU utilisation
