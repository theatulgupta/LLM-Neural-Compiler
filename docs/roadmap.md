# Roadmap

Mapped to the 10-phase compiler thesis (UAV deploy is Phase 0, not the science).

1. Tiny CNN fixture with Flatten 64 / Gemm K=64 (done; K=16 is a regression test).
2. ORT CPU baseline JSON + TensorRT skip-with-reason on non-NVIDIA hosts (done).
3. UAV companion zoo through the same compile loop (done; see `docs/paper_proposal.md`).
4. LLM advisor mock + `schemas/llm-proposal.schema.json` (done).
5. **Transformation engine** on the ONNX DAG: fusion, constant folding, Identity DCE
   (done; `compiler/optimization`). LLM names a pass set; it does not edit nodes.
6. ROS inference node loads the ORT artifact (done; synthetic input, measured only).
7. PX4 SITL `gz_x500` + Micro XRCE-DDS + telemetry: **measured**
   (`experiments/results/sitl_probe.json`).
8. Optional Groq client, still schema-bound (done; key stays outside git).
9. Later: Jetson TensorRT, INT8, Isaac Sim, camera mAP — not this QEMU host.
