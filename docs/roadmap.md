# Roadmap

Mapped to the 10-phase compiler thesis (UAV deploy is Phase 0, not the science).

1. Tiny CNN fixture with Flatten 64 / Gemm K=64 (done; K=16 is a regression test).
2. ORT CPU baseline JSON + TensorRT skip-with-reason on non-NVIDIA hosts (done).
3. UAV companion zoo through the same compile loop (done; see `docs/paper_proposal.md`).
4. LLM advisor mock + plan schema `schemas/llm-plan.schema.json` (done).
5. **Transformation engine** on the ONNX DAG: fusion (real `FusedConv` on ORT CPU),
   constant folding, Identity DCE, MatMul+Add→Gemm, dynamic INT8 (done).
6. Context builder, verifier, candidates, history feedback, report (done).
7. ROS inference node on Gazebo camera frames (wired; SITL binary must be present).
8. PX4 SITL `gz_x500_mono_cam` + Micro XRCE-DDS + `ros_gz_bridge` scripts (done).
   Camera-frame JSON is written when the stack is up (`record_frames.py`).
9. Optional Groq client, still schema-bound (done; key stays outside git).
10. Later: Jetson TensorRT, INT8 static with a large calib set, Isaac Sim, camera mAP
    — not this QEMU host.

**Do not measure:** TensorRT, GPU utilisation, board power, COCO mAP.
**Do measure:** p50/p95 latency, throughput_ips = 1000/mean_ms, RSS, CPU%,
node delta, numerical gates vs native FP32.
