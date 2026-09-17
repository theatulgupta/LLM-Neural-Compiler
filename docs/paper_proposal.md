# Paper proposal: LLM-guided compile for UAV companion models

Working title: **Schema-bound LLM advice for ONNX compilation on a PX4 companion computer.**

This note is for a thesis / conference proposal. It is written in plain English.
Every latency number below was measured on this aarch64 QEMU host and written
to `experiments/results/paper_matrix.json`. Nothing here is TensorRT. Nothing
here is a cloud x86 GPU box.

## Problem

Small drones run cameras on a Linux **companion computer** (Jetson, Raspberry
Pi class). The flight controller (Pixhawk / PX4) only flies. Mission Planner is
ground software for a different stack. The neural net has to run on the
companion, in ONNX Runtime (or TensorRT when an NVIDIA GPU exists).

Today people pick a compile recipe by hand: “turn on ORT fusions”, “try
TensorRT”, “use TVM”. An LLM can suggest those recipes, but a free-form LLM
will invent passes, invent FPS, and sometimes try to fly the drone. That is
not safe and it is not a measurement.

We need a loop that a paper can defend:

1. The LLM may only emit a **schema-valid strategy name** from a small
   allowlist.
2. The compiler **checks** the name, loads the ONNX graph, and compiles.
3. The profiler **measures** real compile time and inference p50/mean.
4. The decision is taken from those logs, not from the LLM’s adjectives.

## Novelty (what is actually true in this repo)

- **Allowlist, not “optimize the net”.** Strategies today are `baseline`,
  `ort_basic`, `ort_extended`, `ort_all`. Unknown names are rejected
  (`schemas/llm-proposal.schema.json`). The LLM cannot add quantization, custom
  ops, or TensorRT tactics in prose.
- **Verify → compile → measure.** `compiler.pipeline.compile_and_benchmark`
  writes `experiments/results/runs/<id>.json`. `fps_claimed` is false.
  `throughput_ips` is `1000 / mean_ms` from samples on this process.
- **Same loop for several real UAV workloads**, not a YOLOv8-only demo.
  Models are data in `experiments/zoo.yaml`. `compiler.pipeline` does not
  switch on YOLOv8. Export scripts are per family; compile/benchmark is shared.
- **Native vs advised on the same machine.** Native = allowlisted `baseline`
  (ORT graph optimizations disabled). Advised = heuristic/Groq strategy
  (here: `ort_extended`) compiled and timed on the **same** aarch64 QEMU CPU.
- **Honest TensorRT.** This host has a Virtio GPU only. TensorRT is skipped
  with a reason. We do not claim Jetson TensorRT speedups.

This is **not** a new autopilot, not a Mission Planner plugin, and not an LLM
that sends offboard setpoints. Planning / control / offboard modules stay empty.

## Related work (contrast, not a full survey)

| Line of work | What it does | How this proposal differs |
| --- | --- | --- |
| TVM / AutoTVM / Ansor | Search schedules; strong compilers | No schema-bound LLM in the loop; we do not replace TVM. We pick an ORT graph-opt **level** that is already allowlisted. |
| ONNX Runtime graph opts | The engine we actually run | We do not invent a new runtime. We **choose** disable vs extended (etc.) through the allowlist and then **measure**. |
| TensorRT | NVIDIA engine, often the Jetson winner | Skipped here: no NVIDIA device. A later Jetson chapter can add it as another backend that still cannot escape the allowlist. |
| LLM agents for flying / code | Free-form plans, sometimes control | Our LLM never flies and never rewrites the graph. It only names a strategy. |
| YOLO papers on drones | Accuracy / FPS of one detector | We treat several **tasks** (detect, pose, seg, SSD-lite, classify, depth) as compile subjects, not as a new detector architecture. |

A full submission still needs a deeper related-work section (citations, AutoTVM
vs ORT, PX4 companion vision stacks). That is listed under gaps.

## Experimental matrix

Host: `uname -m = aarch64`, `machine_id = e996b2604c4c49ada2a0d0e9e55b47d5`,
Python 3.12.3, ONNX Runtime CPU, NVIDIA absent. Warmup 3, iters 8, synthetic
input (not a camera). Backend `ort_cpu` only.

Why these models (companion computer, not the STM32):

| Kind | Task | UAV job |
| --- | --- | --- |
| YOLOv8n | detect | People / vehicles on PX4 companions. Already in this tree. |
| YOLO11n | detect | 2024 Ultralytics nano default (C3k2/C2PSA). Not a second copy of YOLOv8. Chosen over YOLOv8s so this VM does not OOM. |
| YOLOv8n-pose | pose | Keypoints for search-and-rescue and follow-me. A box is not a person. |
| YOLOv8n-seg | segment | Masks for landing-zone / trail. |
| SSDLite320 MobileNetV3 | detect-lite | Classic Pi / older Jetson detector at 320². Different architecture (SSD, not YOLO). NMS stays off-graph. |
| MobileNetV3-small | classify | Landing-pad / gate / sign ID. Smaller than ResNet18. |
| MiDaS small | depth | Monocular depth cue for sense-and-avoid. Not PX4 stereo firmware. |

Fixtures for tests: tiny CNN (Flatten 64 / Gemm K=64) and tiny depth. They are
not claimed as industry models.

## Measured results (from `paper_matrix.json` only)

p50 and mean are milliseconds per inference. Compile is milliseconds once per
run. Speedup is native p50 / optimized p50. Values **below 1** mean the
allowlisted fusions were **not** faster. That is still a result.

| Model | Task | Native p50 (baseline) | Advised p50 (`ort_extended`) | p50 ratio | Native mean | Advised mean | Compile ms (native) | Compile ms (advised) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| yolov8n | detect | 51.09 | 51.11 | 1.00 | 51.34 | 51.41 | 96.2 | 70.7 |
| yolo11n | detect | 45.45 | 43.75 | 1.04 | 45.37 | 43.90 | 82.9 | 83.2 |
| yolov8n-pose | pose | 54.31 | 55.20 | 0.98 | 54.41 | 55.00 | 89.5 | 74.1 |
| yolov8n-seg | segment | 67.93 | 66.76 | 1.02 | 67.93 | 67.06 | 89.8 | 96.9 |
| ssdlite_mobilenetv3 | detect-lite | 11.50 | 11.39 | 1.01 | 11.48 | 11.37 | 36.7 | 37.3 |
| mobilenetv3_small | classify | 2.31 | 2.20 | 1.05 | 2.32 | 2.16 | 10.5 | 10.8 |
| midas_small | depth | 53.92 | 52.02 | 1.04 | 54.12 | 51.49 | 96.9 | 85.1 |

SHA-256 of each ONNX is in `experiments/results/paper_matrix.json`.
`fps_claimed` is false. Throughput in the JSON is `1000/mean_ms` from these
samples, not a camera FPS.

**How to read this.** On this QEMU CPU, extended ORT fusions do **not** give a
large inference win. The biggest p50 gain is MobileNetV3-small (about 5%).
YOLOv8n-pose is slightly slower with fusions. We keep those numbers. A paper
that claimed “the LLM makes YOLO 3× faster on the companion” from this table
would be false.

**Compile vs inference.** Compile is a one-time cost. It is on the order of
one to two inference frames for the YOLO-class nets (70–97 ms compile vs
~50–68 ms p50). It is **not** true that “compiling is cheaper than native
inference” as a general slogan: MobileNet compile is ~11 ms and one inference
is ~2 ms. Compile is paid once; inference is paid every frame.

**TensorRT.** Skipped. Probe reason: no `nvidia-smi`, no `/dev/nvidia0`, Virtio
GPU. Do not compare these p50s to Jetson TensorRT blogs.

**Not compared.** Earlier Groq-live or ROS-under-SITL YOLOv8n logs on this
same VM are **loaded-host** or **different iter counts**. They are not mixed
into the table above. Cloud x86 35 ms numbers are a different computer.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,yolo]'
pip install timm   # MiDaS
python -m compiler zoo
python -m compiler export          # skip JSON if a download fails
python -m compiler matrix --warmup 3 --iters 8
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests
```

LLM path (optional, key stays in `~/.config/nnc/groq.env`, never in git):

```bash
python -m compiler live experiments/models/yolov8n.onnx --kind yolov8n --no-fixture
```

ROS `inference_node` still loads any ORT path; default is YOLOv8n. It does not
become a zoo of if-branches.

## What a full paper still needs

- More iters / repeats (8 is a first table, not a stats chapter).
- A real camera, not synthetic tensors; and a Jetson-class board, not only QEMU.
- TensorRT chapter **on NVIDIA hardware**, still behind the same allowlist.
- Deeper related work (TVM, ORT, TensorRT, PX4 avoidance, companion YOLO
  papers) with citations.
- Accuracy / mAP is out of scope until we add a labelled set; this proposal
  only claims compile + latency logs.
- The LLM advisor on Groq is schema-bound; a submission should show live vs
  heuristic agreement on the zoo, without putting API keys in the paper.

Until those exist, this tree is a **proposal with a measured companion-CPU
table**, not a finished conference paper.
