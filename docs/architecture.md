# Architecture

Thesis: **LLM-Guided Neural Network Compilation for Real-time UAV Edge Inference**.

A network is a **computational graph** (DAG of operators and tensors). That
DAG is what we optimize. ROS 2 / PX4 is only where the compiled model runs.

## Atul’s 10 phases vs this tree

| Phase | Notes | This repo |
| --- | --- | --- |
| 0 Deploy | Ubuntu, ROS 2, PX4, SITL | Scripts in `scripts/`. Camera airframe `gz_x500_mono_cam`. |
| 1–3 Foundations / ONNX | CNN, YOLO/MobileNet, ONNX IR | Analyzer v2: FLOPs, patterns, memory. |
| 4 Compilers | TensorRT, TVM, ORT | ORT CPU backend + `BackendOptions`. TensorRT skip-with-reason. |
| 5 Graph opts | Fusion, fold, DCE, INT8 | `compiler/optimization` (FusedConv is a real ORT CPU op). |
| 6 LLM-guided | Summary → plan | Schema-bound Groq / heuristic / mock; verifier. |
| 7 System | Parser → LLM → planner → transform → engine | Modules below. |
| 8 Sim | ROS inference + PX4 + Gazebo camera | `inference_node` on `/nnc/camera/image_raw`. |
| 9 Eval | Latency, RSS, CPU, numerics | `compile_verify_profile`, `paper_matrix.json`. No invented FPS. |
| 10 Paper | Proposal | `docs/paper_proposal.md`. |

## Phase 7 modules

```
PyTorch weights
  → Exporter          scripts/export_*.py + experiments/zoo.yaml
  → ONNX parser       compiler/parsers, compiler/graph/graph_loader.py
  → Graph analyzer    compiler/graph (FLOPs, patterns, memory)
  → Context builder   compiler/llm/context_builder.py
  → LLM interface     compiler/llm (heuristic / groq / mock)
  → Candidates        compiler/planner/candidates.py
  → Verifier          compiler/planner/verifier.py
  → Transformation    compiler/optimization (mutate ModelProto)
  → Backend           src/nnc/backends (ORT now; TensorRT later)
  → Profiler          compiler/profiling
  → Numerics          compiler/verification
  → Logs / report     experiments/results + compiler/report
```

`src/nnc/backends/` must not import `compiler`. `compiler` may call backends.

## Data flow

1. Load ONNX. `compiler.pipeline` does not switch on YOLOv8.
2. Summarize the DAG (op counts, shape-inferred FLOPs, patterns, liveness).
3. Build context (graph + hardware + constraints + last history rows).
4. LLM proposes a **plan** (`schemas/llm-plan.schema.json`): ordered atoms + options.
5. Verifier drops inapplicable atoms (e.g. FP16 on ORT CPU, BN fuse when no `conv_bn`).
6. Transformation engine applies the pass list. Conv-ReLU becomes
   `com.microsoft.FusedConv` which **runs on ORT CPU**.
7. Artifact + sidecar JSON saved under `experiments/artifacts/`.
8. Profiler measures latency / RSS / CPU. Numerics compare against the native DAG
   on calibration frames when present, else random tensors (`inputs_source` recorded).
9. History feeds the next prompt. `fps_claimed` is false.

## Tiny fixture contract

Input `1×1×8×8` → Conv → ReLU → MaxPool → Flatten → Gemm.
Flatten width `4×4×4 = 64`. Gemm `K` must be 64 (`transB=1`).
`graph_fuse` must drop the Relu node (`tests/test_transformation.py`) and the
runtime graph still contains `FusedConv`.

## UAV stack

ROS 2 Jazzy telemetry and `inference_node` live in `llm_uav_core`.
`scripts/start_all.sh` starts Micro XRCE-DDS, headless Gazebo, PX4
`gz_x500_mono_cam` (standalone), `ros_gz_bridge` onto `/nnc/camera/image_raw`,
and telemetry. Set `NNC_START_INFERENCE=1` to also start `inference_node`.
Empty planning/control stay empty. Inference loads artifacts through
`nnc.artifact`, not Groq.

## Extension seams

- New **pass**: function + allowlist in `compiler/optimization/passes.py` and `compiler/planner/atoms.py`.
- New **preset**: `compiler/planner/plan.py` PRESETS + proposal schema enum.
- New **backend**: `register_backend`.
- New **model**: YAML row in `experiments/zoo.yaml`.
