# Thesis notes

Working title: LLM-Guided Neural Network Compilation for Real-time UAV Edge Inference.

Contribution is graph analysis + allowlisted strategy selection (schema-bound
advisor, mock or heuristic) + measured compile and latency history, not a new
UAV agent. Empty `llm_uav_core` planning/control modules stay empty.

Application domain remains PX4 / ROS 2 / Gazebo edge inference. On this host,
PX4 SITL `gz_x500` published `/fmu/out/vehicle_local_position_v1` with changing
x,y,z (`experiments/results/sitl_probe.json`, `xyz_changed: true`).
