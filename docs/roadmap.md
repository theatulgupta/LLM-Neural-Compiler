# Roadmap

Mapped to the 10-phase compiler thesis (UAV deploy is Phase 0, not the science).

1. Tiny CNN fixture with Flatten 64 / Gemm K=64 (done; K=16 is a regression test).
2. ORT CPU baseline JSON + TensorRT skip-with-reason on non-NVIDIA hosts (done).
3. UAV companion zoo through the same compile loop (done; see `docs/paper_proposal.md`).
4. LLM advisor mock + plan schema `schemas/llm-plan.schema.json` (done).
5. **Transformation engine** on the ONNX DAG: fusion (real `FusedConv` on ORT CPU),
   constant folding, Identity DCE, MatMul+Add→Gemm, dynamic INT8 (done).
6. Context builder, verifier (thread cap / sequential on `ort_cpu`), candidates,
   history feedback, report (done).
7. ROS inference node on Gazebo camera frames (done; `sim_matrix` measured).
8. PX4 SITL `nnc_x500_cam` + Micro XRCE-DDS + `ros_gz_bridge` (done).
   Camera JSON: `record_frames.py`. Optional overlay: `/nnc/annotated`.
9. HTTP LLM client (`HttpLlmClient` + provider registry), schema-bound
   (done; key stays outside git). Groq is one preset. After a measured HTTP
   plan, **one** extra try: `llm_revised` on gate/compile fail, `llm_improved`
   if the first plan passed but is strictly slower. LLM may still lose.
10. **Frozen until a board exists:** Jetson TensorRT *builder*, board power,
    labelled camera mAP, Isaac Sim. There is no fused CPU SiLU kernel; do not
    add a fake `fuse_conv_silu` pass. INT8 + threads are the CPU levers.

**Do not measure on this QEMU host:** TensorRT, GPU utilisation, board power, COCO mAP.
**Do measure:** p50/p95 latency, throughput_ips = 1000/mean_ms, RSS, CPU%,
node delta, numerical gates vs native FP32, llm_rank vs heuristic vs presets.
`fps_claimed` is false.
