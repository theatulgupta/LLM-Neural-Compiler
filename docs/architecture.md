# Architecture

Thesis: **LLM-Guided Neural Network Compilation for Real-time UAV Edge Inference**.

A network is a **computational graph** (DAG of operators and tensors). That
DAG is what we optimize. ROS 2 / PX4 is only where the compiled model runs.

## Atul’s 10 phases vs this tree

| Phase | Notes | This repo |
| --- | --- | --- |
| 0 Deploy | Ubuntu, ROS 2, PX4, SITL | Done. Empty planning/control stay empty. |
| 1–3 Foundations / ONNX | CNN, YOLO/MobileNet, ONNX IR | Analyzer + zoo exporters. |
| 4 Compilers | TensorRT, TVM, ORT | ORT CPU backend. TensorRT skip-with-reason (no NVIDIA). |
| 5 Graph opts | Fusion, fold, DCE | **Transformation engine** in `compiler/optimization`. |
| 6 LLM-guided | Summary → strategy | Schema-bound Groq / heuristic / mock. |
| 7 System | Parser → LLM → planner → transform → engine | Modules below. |
| 8 Sim | ROS inference + PX4 | SITL measured. Isaac Sim later. |
| 9 Eval | Latency from logs | `paper_matrix.json`, `graph_changed`. No invented FPS. |
| 10 Paper | Proposal | `docs/paper_proposal.md`. |

## Phase 7 modules

```
PyTorch weights
  → Exporter          scripts/export_*.py + experiments/zoo.yaml
  → ONNX parser       compiler/parsers, compiler/graph/graph_loader.py
  → Graph analyzer    compiler/graph
  → LLM interface     compiler/llm
  → Planner           compiler/strategies.py (named pass sets)
  → Transformation    compiler/optimization (mutate ModelProto)
  → Backend           src/nnc/backends (ORT now; TensorRT later)
  → Logs              experiments/results/runs + history.jsonl
```

`src/nnc/backends/` must not import `compiler`. `compiler` may call backends.

## Data flow

1. Load ONNX. `compiler.pipeline` does not switch on YOLOv8.
2. Summarize the DAG (op counts, FLOPs estimate, shapes).
3. LLM may only name `baseline` | `graph_simplify` | `graph_fuse` | `graph_fuse_ort`.
4. Transformation engine applies the pass list (shape infer, Identity DCE,
   constant fold, Conv-BN fuse, Conv-ReLU fuse). Unknown passes raise.
5. Conv-ReLU becomes compiler IR `FusedConv`. ORT CPU may not run that op, so
   `prepare_for_ort` expands it back to Conv+Relu **for the session only**.
   Run JSON still logs the fused IR (`graph_after`) vs the runtime DAG
   (`graph_runtime`).
6. Backend compiles and the profiler **measures**. `fps_claimed` is false.

## Tiny fixture contract

Input `1×1×8×8` → Conv → ReLU → MaxPool → Flatten → Gemm.
Flatten width `4×4×4 = 64`. Gemm `K` must be 64 (`transB=1`).
`graph_fuse` must drop the Relu node in compiler IR (`tests/test_transformation.py`).

## UAV stack

ROS 2 Jazzy telemetry and `inference_node` live in `llm_uav_core`.
PX4 SITL / Gazebo are probed. Empty planning/control stay empty.
Inference loads artifacts through `nnc.artifact`, not Groq.

## Extension seams

- New **pass**: function + allowlist in `compiler/optimization/passes.py`.
- New **strategy**: named pass set in `compiler/strategies.py` + schema enum.
- New **backend**: `register_backend`.
- New **model**: YAML row in `experiments/zoo.yaml`.
