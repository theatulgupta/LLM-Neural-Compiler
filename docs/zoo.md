# UAV companion zoo

Seven industry-style onboard nets share one compile loop (`experiments/zoo.yaml`).
Latency numbers live in `experiments/results/report.md` (from `paper_matrix.json`
and `sim_inference_*.json`). Do not copy latencies into this page.

| Kind | Task | Input | UAV role | Export | Decode |
| --- | --- | --- | --- | --- | --- |
| `yolov8n` | detect | 1×3×640×640 | people / vehicles on a PX4 companion | `python -m compiler export --kind yolov8n` | full boxes |
| `yolo11n` | detect | 1×3×640×640 | same job, newer Ultralytics nano graph | `python -m compiler export --kind yolo11n` | full boxes |
| `yolov8n-pose` | pose | 1×3×640×640 | SAR / follow-me keypoints | `python -m compiler export --kind yolov8n-pose` | boxes + 17×3 keypoints |
| `yolov8n-seg` | segment | 1×3×640×640 | landing-zone / trail masks | `python -m compiler export --kind yolov8n-seg` | boxes only (`mask_decoded: false`) |
| `ssdlite_mobilenetv3` | detect-lite | 1×3×320×320 | Pi-class SSD-MobileNet detector | `python -m compiler export --kind ssdlite_mobilenetv3` | SSDLite heads, class-agnostic NMS |
| `mobilenetv3_small` | classify | 1×3×224×224 | pad / gate / sign ID | `python -m compiler export --kind mobilenetv3_small` | top-1 |
| `midas_small` | depth | 1×3×256×256 | monocular sense-and-avoid cue | `python -m compiler export --kind midas_small` | passthrough min/max/mean |

ONNX files are gitignored. The planner does not switch on YOLOv8.

```bash
python -m compiler zoo
python -m compiler export
python -m compiler matrix --candidates all --warmup 5 --iters 30
python -m compiler report
```
