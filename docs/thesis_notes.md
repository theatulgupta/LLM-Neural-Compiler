# Thesis notes

Working title: LLM-Guided Neural Network Compilation for Real-time UAV Edge Inference.

ML engineers see layers. This thesis sees a **computational graph**. The LLM
may only name an allowlisted pass set. The transformation engine rewrites the
ONNX DAG (fusion, folding, dead-node elim). The profiler measures compile and
inference time. UAV (ROS 2 / PX4 SITL) is the deployment box, not the
contribution. Empty `llm_uav_core` planning/control modules stay empty.

Application domain remains PX4 / ROS 2 / Gazebo edge inference. On this host,
PX4 SITL `gz_x500` published changing x,y,z
(`experiments/results/sitl_probe.json`).

Zoo models share one compile loop (`experiments/zoo.yaml`). See
`docs/architecture.md` and `docs/paper_proposal.md`.
