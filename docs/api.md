Allowlisted **plan atoms** (graph passes) plus ORT session options.
The LLM may only name those atoms. Unknown names are rejected. Plan JSON
must match `schemas/llm-plan.schema.json`. Preset names (`baseline`,
`ort_default`, `graph_simplify`, `graph_fuse`, `graph_fuse_ort`,
`int8_dynamic`, `int8_static`) remain valid `--strategy` values.

API surface:

- `python -m compiler emit-fixture` (`--kind cnn|depth`)
- `python -m compiler analyze <model.onnx>`
- `python -m compiler plan <model.onnx>` (verified plan JSON)
- `python -m compiler recommend <model.onnx>` (legacy name → preset)
- `python -m compiler infer <model.onnx>` (ORT CPU load + measured latency)
- `python -m compiler compile <model.onnx> --backend ort_cpu|tensorrt --strategy PRESET`
- `python -m compiler optimize <model.onnx> --kind KIND [--candidates default|all]`
- `python -m compiler zoo`
- `python -m compiler export [--kind KIND]`
- `python -m compiler matrix [--candidates default|all]`
- `python -m compiler report`
- `python -m compiler baseline`
- `python -m compiler probe`
- `python -m compiler formats` (frontends / backends / IR)
- `python -m compiler live` (Groq if `~/.config/nnc/groq.env`; key never printed)

ROS 2 (after `scripts/build_ros.sh`):

- `scripts/start_all.sh` then `scripts/record_frames.py`
- `ros2 run llm_uav_core telemetry_node`
- `NNC_START_INFERENCE=1 scripts/start_all.sh` or `scripts/start_inference.sh`
  (camera frames → `/nnc/detections`, `/nnc/latency_ms`, `/nnc/status`)
- `scripts/verify_sim_inference.py --seconds 60`
- `scripts/stop_all.sh`
