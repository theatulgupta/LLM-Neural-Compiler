# Design (HLD + LLD)

Thesis compiler: a graph goes in, an allowlisted plan is chosen, the DAG is
rewritten, ORT (or later TensorRT) runs it, numbers are measured. UAV ROS/PX4
is the deploy box, not the science.

## High level

```
source model (.onnx, later .pt via export)
        |  frontend
        v
     GraphIR (ONNX today)
        |  analyzer
        v
   graph summary + hardware + history
        |  prompt builder  (compiler/llm/prompting.py)
        v
   LLM / heuristic plan  ->  verifier (drop illegal atoms)
        |  pass engine
        v
   rewritten ONNX  ->  backend (ort_cpu | tensorrt skip)
        v
   latency / RSS / numerics JSON
        v
   ROS inference_node on /nnc/camera/image_raw
```

Same image topic on a real drone later; only the publisher changes.

ONNX is the **IR**, not “UAVs only run ONNX”. A Jetson will want TensorRT
`.engine`. That is a backend. PyTorch `.pt` is a frontend (export to ONNX).

## Packages

| Package | Job |
| --- | --- |
| `compiler/frontends` | suffix → GraphIR or skip-with-reason |
| `compiler/graph` | load, FLOPs, patterns, summary |
| `compiler/llm` | context, prompt, Groq/heuristic |
| `compiler/planner` | atoms, presets, verifier, candidates |
| `compiler/optimization` | mutate the DAG |
| `compiler/pipeline` | compile, optimize, matrix |
| `compiler/schema` | JSON schemas |
| `src/nnc/backends` | run the artifact (`compiler` may import this; not the reverse) |

## SOLID (what we actually did)

- One module, one job (prompt builder does not HTTP; Groq does not write prompts).
- New pass / frontend / backend registers; pipeline does not grow if-else on YOLO.
- Planner depends on the `LlmClient` protocol, not Groq.
- `LoadedGraph` is the handle tests and the pipeline share.

## Not on this QEMU box

Gazebo camera frames, TensorRT, Jetson power, COCO mAP. Do not invent them.
