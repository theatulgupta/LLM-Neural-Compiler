Allowlisted strategies: `baseline`, `graph_simplify`, `graph_fuse`,
`graph_fuse_ort`. Each name is a **graph pass set** plus an ORT session
graph-opt level. Unknown names are rejected. LLM JSON must match
`schemas/llm-proposal.schema.json`. There is no free-form rewrite API.

API surface:

- `python -m compiler emit-fixture` (`--kind cnn|depth`)
- `python -m compiler analyze <model.onnx>`
- `python -m compiler recommend <model.onnx>`
- `python -m compiler infer <model.onnx>` (ORT CPU load + measured latency)
- `python -m compiler compile <model.onnx> --backend ort_cpu|tensorrt`
- `python -m compiler zoo`
- `python -m compiler export [--kind KIND]`
- `python -m compiler matrix` (native `baseline` vs allowlisted pass set)
- `python -m compiler baseline`
- `python -m compiler probe`
- `python -m compiler live` (Groq if `~/.config/nnc/groq.env`; key never printed)

ROS 2 (after `scripts/build_ros.sh`):

- `ros2 run llm_uav_core telemetry_node`
- `ros2 run llm_uav_core inference_node` (loads the ORT artifact; publishes `/nnc/status`)
