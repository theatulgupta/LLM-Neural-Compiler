# Architecture

## Contribution boundary

`compiler/` is the thesis contribution: ONNX load, graph summary, allowlisted
LLM recommendations, pass application, and JSON run history.

`src/nnc/backends/` is runtime infrastructure. It must not import `compiler`.
`compiler` may call backends.

## Data flow

1. `compiler.parsers` loads ONNX (tiny fixture or YOLOv8n).
2. `compiler.graph` infers shapes, counts ops, estimates coarse FLOPs.
3. `compiler.llm.recommendation_engine` maps the summary onto
   `compiler.strategies.ALLOWED_STRATEGIES`. Unknown names raise
   `UnknownStrategyError`.
4. `compiler.optimization` runs only allowlisted passes (today: ONNX shape
   inference). ORT graph-opt level is a strategy field, not a free-form rewrite.
5. A backend compiles and benchmarks. Results are appended to
   `experiments/results/history.jsonl` and `experiments/results/runs/<id>.json`.

## Tiny fixture contract

Input `1×1×8×8` → Conv 1→4 (3×3, pad 1) → ReLU → MaxPool 2×2 → Flatten → Gemm.

Flatten width is `4×4×4 = 64`. Gemm uses `transB=1` so `B` is `[N, K] = [8, 64]`.
Setting `K=16` (channel count or `4×4` spatial) is the compile failure this
tree treats as a regression.

## UAV stack

ROS 2 Jazzy telemetry and an ORT inference node live in `llm_uav_core`.
PX4 SITL / Gazebo are probed, not stubbed. Empty planning/control modules stay
empty. Scripts point at `~/PX4-Autopilot` and `~/px4_ros_uxrce_dds_ws`, not
gitignored `third_party/` checkouts.
