# Thesis notes

Working title: LLM-Guided Neural Network Compilation for Real-time UAV Edge Inference.

ML engineers see layers. This thesis sees a **computational graph**. The LLM
is a **planner**: it names allowlisted atoms. The verifier drops what the
graph/hardware cannot run. The transformation engine rewrites the ONNX DAG
(fusion including `com.microsoft.FusedConv` on ORT CPU, folding, DCE, INT8).
The profiler measures compile time, p50, RSS, CPU. UAV (ROS 2 / PX4 SITL /
Gazebo camera) is the deployment box, not the contribution. Empty
`llm_uav_core` planning/control modules stay empty.

Notebook mapping:

| Notebook box | Module |
| --- | --- |
| Graph analyzer | `compiler/graph/` |
| Context builder | `compiler/llm/context_builder.py` |
| LLM | `compiler/llm/` |
| Strategy generator | `compiler/planner/candidates.py` |
| Verification | `compiler/planner/verifier.py` + `compiler/verification/` |
| Compiler | `compiler/optimization/` + `src/nnc/backends/` |
| Profiler | `compiler/profiling/` |
| Report / history | `compiler/report/`, `compiler/history.py` |

Application domain remains PX4 / ROS 2 / Gazebo edge inference.

Zoo models share one compile loop (`experiments/zoo.yaml`). See
`docs/architecture.md` and `docs/paper_proposal.md`.
