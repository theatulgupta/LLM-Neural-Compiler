Allowlisted strategies: `baseline`, `ort_basic`, `ort_extended`, `ort_all`.
Unknown names are rejected. LLM advisor JSON must match
`schemas/llm-proposal.schema.json`. There is no free-form graph rewrite API.

API surface:

- `python -m compiler emit-fixture`
- `python -m compiler analyze <model.onnx>`
- `python -m compiler recommend <model.onnx>`
- `python -m compiler infer <model.onnx>` (ORT CPU load + measured latency)
- `python -m compiler compile <model.onnx> --backend ort_cpu|tensorrt`
- `python -m compiler baseline`
- `python -m compiler probe`

ROS 2 (after `scripts/build_ros.sh`):

- `ros2 run llm_uav_core telemetry_node`
- `ros2 run llm_uav_core inference_node` (loads the ORT artifact; publishes `/nnc/status`)
