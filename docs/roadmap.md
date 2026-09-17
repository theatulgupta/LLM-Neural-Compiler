# Roadmap

1. Tiny CNN fixture with Flatten 64 / Gemm K=64 (done; K=16 is a regression test).
2. ORT CPU baseline JSON + TensorRT skip-with-reason on non-NVIDIA hosts (done).
3. UAV companion zoo (detect, pose, seg, SSD-lite, classify, depth) through the
   same compiler loop on this aarch64 QEMU host (see `docs/paper_proposal.md`).
4. LLM advisor mock + `schemas/llm-proposal.schema.json` (done; heuristic is the default client).
5. ROS inference node that loads the ORT artifact (done; synthetic input, measured latency only).
6. PX4 SITL `gz_x500` on `~/PX4-Autopilot` + Micro XRCE-DDS + telemetry: **measured**.
   `/fmu/out/vehicle_local_position_v1` showed changing x,y,z (`experiments/results/sitl_probe.json`).
7. Optional networked LLM client still filtered by the same allowlist.
