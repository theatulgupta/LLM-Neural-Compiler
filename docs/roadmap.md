# Roadmap

1. Tiny CNN fixture with Flatten 64 / Gemm K=64 (done; K=16 is a regression test).
2. ORT CPU baseline JSON + TensorRT skip-with-reason on non-NVIDIA hosts (done).
3. YOLOv8n export + ORT CPU compile/infer on this aarch64 QEMU host (done; see `docs/yolov8n.md`).
4. LLM advisor mock + `schemas/llm-proposal.schema.json` (done; heuristic is the default client).
5. ROS inference node that loads the ORT artifact (done; synthetic input, measured latency only).
6. PX4 SITL firmware `px4_sitl_default` / `gz_x500` on `~/PX4-Autopilot` (firmware built this session).
   Runtime telemetry is **not** success until `scripts/verify_sitl.py` records changing x,y,z.
7. Optional networked LLM client still filtered by the same allowlist.
