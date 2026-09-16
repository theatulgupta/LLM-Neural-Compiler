# Thesis notes

Working title: LLM-Guided Neural Network Compilation for Real-time UAV Edge Inference.

Contribution is graph analysis + allowlisted strategy selection (schema-bound
advisor, mock or heuristic) + measured compile and latency history, not a new
UAV agent. Empty `llm_uav_core` planning/control modules stay empty.

Application domain remains PX4 / ROS 2 / Gazebo edge inference. SITL firmware
can be built on this host; do not cite telemetry numbers unless
`experiments/results/sitl_probe.json` shows `xyz_changed: true`.
