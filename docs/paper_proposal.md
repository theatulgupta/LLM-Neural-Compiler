# Paper proposal: LLM-guided compile for UAV companion models

Working title: **Schema-bound LLM advice for ONNX graph compilation on a PX4 companion computer.**

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
   allowlist (`baseline`, `graph_simplify`, `graph_fuse`, `graph_fuse_ort`).
2. The **transformation engine** applies that named pass set to the ONNX DAG
   (fusion, constant folding, dead-node elimination). It does not invent ops.
3. The profiler **measures** real compile time and inference p50/mean.
4. The decision is taken from those logs (`graph_changed`, node counts, p50),
   not from the LLM’s adjectives.

## Novelty (what is actually true in this repo)

- **Allowlist, not “optimize the net”.** Strategies are named **graph pass
  sets**: `baseline`, `graph_simplify`, `graph_fuse`, `graph_fuse_ort`
  (`schemas/llm-proposal.schema.json`). The LLM cannot add quantization,
  custom ops, or TensorRT tactics in prose.
- **Transformation engine owns the DAG.** `compiler/optimization` copies the
  `ModelProto` and applies shape infer, Identity DCE, constant folding,
  Conv-BN fuse, Conv-ReLU fuse. Conv-ReLU is logged as `FusedConv`; ORT CPU
  may not run that op, so `prepare_for_ort` expands it for the session.
  `graph_after` is the compiler IR; `graph_runtime` is what ORT ran.
- **Verify → rewrite → measure.** `compile_and_benchmark` writes
  `experiments/results/runs/<id>.json` with `graph_changed`, node counts,
  `fps_claimed: false`. Throughput is `1000 / mean_ms` from samples.
- **Same loop for several real UAV workloads**, not a YOLOv8-only demo.
  Models are data in `experiments/zoo.yaml`.
- **Native vs advised on the same machine.** Native = unrewritten ONNX
  (`baseline`). Advised = heuristic `graph_fuse` on the **same** aarch64 QEMU
  CPU. YOLO graphs here use SiLU, not Relu, so Conv-ReLU fuse is a **no-op**
  (`graph_changed: false`). SSDLite / MobileNetV3 / MiDaS **do** shrink.
- **Honest TensorRT.** Virtio GPU only. TensorRT skipped with a reason.

This is **not** a new autopilot, not a Mission Planner plugin, and not an LLM
that sends offboard setpoints. Planning / control / offboard modules stay empty.

## Related work (contrast, not a full survey)

| Line of work | What it does | How this proposal differs |
| --- | --- | --- |
| TVM / AutoTVM / Ansor | Search schedules; strong compilers | We do not replace TVM. We apply a **small allowlisted ONNX pass set** the LLM named, then measure. |
| ONNX Runtime graph opts | The engine we actually run | Ablation `graph_fuse_ort` can add ORT extended **after** our DAG passes. Native is our unrewritten graph, not an ORT knob. |
| TensorRT | NVIDIA engine, often the Jetson winner | Skipped here: no NVIDIA device. A later Jetson chapter can add it as another backend that still cannot escape the allowlist. |
| LLM agents for flying / code | Free-form plans, sometimes control | Our LLM never flies. It only names a pass set. The engine rewrites the graph. |
| YOLO papers on drones | Accuracy / FPS of one detector | We treat several **tasks** as compile subjects, not a new detector. |

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
allowlisted rewrites were **not** faster. `graph_changed` is compiler IR
(node_count / op_counts), not a marketing flag.

| Model | Task | Native p50 | Advised p50 (`graph_fuse`) | p50 ratio | DAG changed | Nodes before→after | Native mean | Advised mean | Compile ms (native) | Compile ms (advised) |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| yolov8n | detect | 50.29 | 51.78 | 0.97 | no | 263→263 | 50.15 | 51.96 | 68.0 | 62.3 |
| yolo11n | detect | 43.97 | 44.06 | 1.00 | no | 355→355 | 44.15 | 44.18 | 60.7 | 58.4 |
| yolov8n-pose | pose | 53.76 | 55.32 | 0.97 | no | 313→313 | 53.78 | 55.54 | 69.7 | 72.0 |
| yolov8n-seg | segment | 67.91 | 67.42 | 1.01 | no | 301→301 | 68.89 | 67.50 | 81.8 | 82.5 |
| ssdlite_mobilenetv3 | detect-lite | 11.36 | 11.22 | 1.01 | yes | 325→306 | 11.60 | 12.99 | 21.8 | 22.5 |
| mobilenetv3_small | classify | 2.47 | 2.35 | 1.05 | yes | 141→127 | 2.47 | 2.43 | 12.4 | 12.0 |
| midas_small | depth | 52.22 | 53.75 | 0.97 | yes | 645→636 | 52.49 | 53.20 | 96.5 | 91.4 |

SHA-256 of each ONNX is in `experiments/results/paper_matrix.json`.
`fps_claimed` is false.

**How to read this.** Ultralytics YOLO ONNX on this export uses SiLU, not Relu,
so `fuse_conv_relu` does not rewrite those graphs (nodes stay). MobileNet-class
and MiDaS graphs **do** shrink. Latency on this QEMU CPU is still a small wash
(about 0.97–1.05×). Tiny CNN / tiny depth **do** change in unit tests. A paper
that claimed “the LLM makes YOLO 3× faster on the companion” from this table
would be false.

**Compile vs inference.** Compile is paid once. YOLO-class compile is ~60–80 ms
vs ~44–68 ms p50. MobileNet compile is ~12 ms vs ~2 ms p50.

**TensorRT.** Skipped. Probe reason: no `nvidia-smi`, no `/dev/nvidia0`, Virtio
GPU.

**Not compared.** Earlier Groq-live or ROS-under-SITL YOLOv8n logs, or the old
ORT-knob-only matrix, are different experiments. Cloud x86 numbers are a
different computer.

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

- A `fuse_conv_silu` pass for Ultralytics graphs (SiLU, not Relu).
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
