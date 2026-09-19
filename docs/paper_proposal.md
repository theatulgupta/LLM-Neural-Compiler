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

1. The LLM may only emit a **schema-valid plan** (`schemas/llm-plan.schema.json`):
   ordered allowlisted atoms plus ORT options, with rationale and confidence.
2. The **verifier** drops atoms the graph or hardware cannot run (e.g. FP16 on
   ORT CPU, Conv-BN fuse when there is no BN).
3. The **transformation engine** applies the remaining atoms to the ONNX DAG.
   `com.microsoft.FusedConv` **runs** on ORT CPU; it is not expanded away.
4. The profiler **measures** compile time, p50/p95, RSS, CPU. Numerics compare
   against the native DAG.
5. History (last five `{plan_id, p50_ms, passed, origin}` for that kind) plus
   **at most one** extra HTTP round: a compact measured table is sent back as
   `measured_failure` (compile/gates) or `measured_not_best` (passed but
   strictly slower). Never both. Never a while-loop.

## Novelty (what is actually true in this repo)

- **Allowlist of atoms, not free-form rewrite.** Plans are ordered pass lists
  (`fuse_conv_relu`, `quantize_dynamic_int8`, …) plus `ort_graph_opt` /
  `intra_op_threads`. The LLM cannot add custom ops in prose.
- **Transformation engine owns the DAG.** `compiler/optimization` copies the
  `ModelProto`. Conv-ReLU becomes `com.microsoft.FusedConv`, which ORT CPU
  executes. `graph_after` equals `graph_runtime`.
- **Verify → rewrite → measure → one bounded revision.** After the first HTTP
  plan is compiled, `optimize_model` may ask once more with the measured table
  (`origin=llm_revised` or `llm_improved`). Offline heuristic/mock skip that
  round. A heuristic fallback from a failed follow-up is recorded as
  `{skipped: fallback: …}` and is **not** compiled under an LLM origin.
- **Same loop for several real UAV workloads**, not a YOLOv8-only demo.
- **Native vs advised on the same machine.** Native = unrewritten ONNX
  (`baseline`). YOLO graphs here use SiLU, so Conv-ReLU fuse is a **no-op**.
  INT8 and thread options are the CPU levers. SSDLite / MobileNetV3 / MiDaS
  can shrink under fuse.
- **Honest TensorRT.** Virtio GPU only. TensorRT skipped with a reason.
- **Provider-agnostic HTTP.** Groq is one `ProviderSpec`. The same CLI is
  `NNC_LLM=openai` / `custom`. There is no Groq branch in `optimize_model`.

This is **not** a new autopilot, not a Mission Planner plugin, and not an LLM
that sends offboard setpoints. Planning / control / offboard modules stay empty.

## Related work (contrast, not a full survey)

Citations below are real venues / product docs. No invented DOI. No invented
speedups from those systems.

| Line of work | What it does | How this proposal differs |
| --- | --- | --- |
| TVM ([Chen et al., OSDI 2018](https://www.usenix.org/conference/osdi18/presentation/chen)) / Ansor ([Zheng et al., OSDI 2020](https://www.usenix.org/conference/osdi20/presentation/zheng)) | Search tensor schedules and generate kernels | We do not replace TVM. We apply a **small allowlisted ONNX pass set** the LLM named, then measure on ORT CPU. |
| TASO ([Jia et al., SOSP 2019](https://doi.org/10.1145/3341301.3359630)) | Auto-generated, formally verified graph substitutions | Closest graph-level cousin. Our substitutions are a closed enum; the LLM names them; a drop/reject verifier is the last word, not a theorem prover. |
| LLMs for compiler opt ([Cummins et al., arXiv:2309.07062](https://arxiv.org/abs/2309.07062); [Meta LLM Compiler, arXiv:2407.02524](https://arxiv.org/abs/2407.02524)) | Fine-tuned models that propose LLVM pass lists / IR | They target LLVM assembly and code size. We target ONNX graphs for UAV companion inference, schema-bound, with measured p50 and numerics gates, and **no** fine-tune. |
| ONNX Runtime ([onnxruntime.ai](https://onnxruntime.ai)) | The engine we actually run | Ablation `graph_fuse_ort` can add ORT extended **after** our DAG passes. Native is our unrewritten graph, not an ORT knob. |
| NVIDIA TensorRT (developer guide) | NVIDIA engine, often the Jetson winner | Skipped here: no NVIDIA device. A later Jetson chapter can add it as another backend that still cannot escape the allowlist. |
| PX4 ([Meier, Honegger, Pollefeys, ICRA 2015](https://doi.org/10.1109/ICRA.2015.7140074); [companion-computer docs](https://docs.px4.io/main/en/companion_computer/)) | FMU flies; companion runs vision | ROS / PX4 SITL is the deploy adapter. The LLM never sends setpoints. |
| Workload papers, not compile papers | YOLOv8/11 (Ultralytics software); MobileNetV3 ([Howard et al., ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html)); SSD ([Liu et al., ECCV 2016](https://doi.org/10.1007/978-3-319-46448-0_2)); MiDaS ([Ranftl et al., TPAMI 2022](https://arxiv.org/abs/1907.01341)) | We treat these as **compile subjects**, not new detectors. |

## Experimental matrix

Host: `uname -m = aarch64`, `machine_id = e996b2604c4c49ada2a0d0e9e55b47d5`,
Python 3.12.3, ONNX Runtime CPU, NVIDIA absent. This table is
`NNC_LLM=groq python -m compiler matrix --candidates all --warmup 5 --iters 30`
then `python -m compiler report`. Numerics use recorded calib tensors
(`inputs_source=calib`), not the Gazebo camera. Backend `ort_cpu` only.

Absolute p50 on this QEMU box moves with load. Gaps of a few percent are
noise, not a ranking claim. Do not mix this table with an older warmup-3 /
iters-8 run, a prior Groq matrix, or a Jetson.

Prompts for this run included the last five `history.jsonl` rows **for that
kind** (including the earlier Groq matrix). That is the designed advisor loop,
not a clean-room ablation. History was not wiped.

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

## Measured results (from `paper_matrix.json` / `report.md` only)

p50 is milliseconds per inference on this host. Speedup is baseline p50 /
chosen p50. Three LLM columns, not one:

- **First plan** is `origin=llm` (`llm_rank` / `llm.p50` / passed). Empty
  `llm_rank` means the first plan failed compile or numerics.
- **Follow-up** is at most one extra HTTP+compile: `llm_revised` or
  `llm_improved`, or `{skipped: …}`.
- **Chosen** is the fastest **passed** row after that follow-up. It may be a
  preset, the heuristic, `llm_revised`, or `llm_improved`. It is **not**
  implied by `llm_rank`.

`fps_claimed` is false. Full candidate tables are in
`experiments/results/report.md`.

| Model | Task | Baseline p50 | First LLM (source / plan / rank / passed) | Follow-up | Chosen (origin / plan) | Chosen p50 | Speedup |
| --- | --- | ---: | --- | --- | --- | ---: | ---: |
| yolov8n | detect | 90.58 | groq / `uav_dynamic_quant_fuse` / 9 / yes | improved `uav_dynamic_quant_extended` p50 68.02 rank 3 (passed) | preset / `int8_dynamic` | 61.84 | 1.465 |
| yolo11n | detect | 78.49 | groq / `uav_dynamic_int8_ort_extended` / — / no | revised `graph_fuse_ort_all` p50 76.36 **rank 1** (passed) | **llm_revised** / `graph_fuse_ort_all` | 76.36 | 1.028 |
| yolov8n-pose | pose | 96.61 | heuristic fallback (HTTP 429) / `graph_fuse` / 2 / yes | none (offline source) | heuristic / `graph_fuse` | 63.59 | 1.519 |
| yolov8n-seg | segment | 120.72 | groq / `uav_companion_opt_graph_simplify` / 6 / yes | improved `uav_companion_opt_dynamic_extended` p50 89.72 **failed gates** | preset / `graph_fuse_ort` | 120.02 | 1.006 |
| ssdlite_mobilenetv3 | detect-lite | 20.72 | groq / `uav_graph_fuse_ort_dynamic` / — / no | revised `graph_fuse_ort_threads` p50 20.60 rank 2 (passed) | preset / `graph_fuse_ort` | 20.56 | 1.007 |
| mobilenetv3_small | classify | 4.10 | groq / `uav_companion_opt_v2` / — / no | revised skipped `fallback: LLM HTTP 429 (rate_limit_exceeded)` | preset / `ort_default` | 2.80 | 1.464 |
| midas_small | depth | 95.37 | groq / `uav_companion_opt` / 8 / yes | improved `graph_fuse_ort_parallel` p50 199.07 rank 9 (passed, slower) | preset / `graph_fuse_ort` | 89.40 | 1.067 |

SHA-256 of each ONNX is in `experiments/results/paper_matrix.json`.

**How to read this.** The LLM is not claimed to win.

- On **YOLO11n** the first Groq INT8-style plan failed numerics; the one
  `measured_failure` retry produced `graph_fuse_ort_all`, which **is** the
  chosen row (rank 1 among passed plans, 1.028× vs baseline). That is the
  recovery path working, not a first-plan win.
- On **YOLOv8n** the first plan passed but was slower than INT8 (rank 9). One
  `llm_improved` try reached rank 3 (68 ms) and still lost to preset
  `int8_dynamic` (62 ms, 1.465× vs baseline).
- On **YOLOv8n-seg** the improved plan was **faster** (90 ms vs 120 ms) and
  **failed gates**, so it is not chosen. Latency without numerics is not a win.
- On **SSDLite** the revised plan passed and sat 0.2% behind `graph_fuse_ort`.
  Treat that gap as QEMU noise.
- **Pose** hit Groq HTTP 429 on the first call; the row is heuristic, not a
  silent skip. **MobileNet** hit 429 on the follow-up; that compile was skipped
  (`fallback: LLM HTTP 429`) instead of stamping a heuristic plan as
  `llm_revised`.
- **MiDaS** improved with `execution_mode=parallel`; the verifier’s CPU path
  still compiled it, and it was slower. The extra try is allowed to lose.

Ultralytics YOLO ONNX here uses SiLU, not Relu, so `fuse_conv_relu` does not
rewrite those DAGs (263→263, 355→355, 313→313, 301→301). There is **no fused
CPU SiLU kernel** in this tree; do not invent `fuse_conv_silu`. INT8 grows the
graph (YOLOv8n 263→640) and cuts weight bytes (~12.8 MiB → ~3.5 MiB).
MobileNetV3 and MiDaS **do** shrink under Conv-BN / Conv-ReLU fuse (141→127,
645→636). SSDLite fuse is 325→306. Tiny CNN / tiny depth change in unit tests.

**TensorRT.** Skipped. Probe reason: no `nvidia-smi`, no `/dev/nvidia0`, Virtio
GPU.

**Camera.** Gazebo `nnc_x500_cam` inference JSON lives under
`sim_inference_*.json` in the same results dir. That is a different experiment
(live frames, earlier artifacts). Do not paste those p50s into this table.

**Not compared.** Older warmup-3 / iters-8 tables, the previous Groq matrix,
cloud x86 boxes, or a Jetson. Those are different computers or different runs.

## Reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,yolo]'
pip install timm   # MiDaS
python -m compiler zoo
python -m compiler export          # skip JSON if a download fails
NNC_LLM=heuristic python -m compiler matrix --candidates all --warmup 5 --iters 30
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests
```

LLM path (optional; key stays in `~/.config/nnc/<provider>.env` or `llm.env`,
never in git):

```bash
# Groq is one preset. Same CLI for openai / openrouter / custom URL.
export NNC_LLM=groq   # or openai, ollama, custom, …
python -m compiler plan experiments/models/yolov8n.onnx
python -m compiler matrix --candidates all --warmup 5 --iters 30
python -m compiler report
```

ROS `inference_node` still loads any ORT path; default is YOLOv8n. It does not
become a zoo of if-branches.

## What a full paper still needs

- Repeats / more iters for a stats chapter (30 iters is one QEMU snapshot).
- A Jetson-class board, not only QEMU; TensorRT chapter **on NVIDIA hardware**,
  still behind the same allowlist.
- Accuracy / mAP is out of scope until a labelled set exists; this proposal
  claims compile + latency + numerics vs native FP32.
- A submission should quote `report.md`, not invent a win rate, and must not
  put API keys in the paper.

Until a board chapter exists, this tree is a **proposal with a measured
companion-CPU table**, not a finished conference paper.
