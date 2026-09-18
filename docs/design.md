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
   LlmClient.propose  ->  verifier (drop illegal atoms)
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
| `compiler/llm` | context, prompt, `LlmClient`, provider registry |
| `compiler/planner` | atoms, presets, verifier, candidates |
| `compiler/optimization` | mutate the DAG |
| `compiler/pipeline` | compile, optimize, matrix |
| `compiler/schema` | JSON schemas |
| `src/nnc/backends` | run the artifact (`compiler` may import this; not the reverse) |

## LLM client (LLD)

```
                    <<protocol>>
                      LlmClient
                     propose()
                          ^
          ----------------+----------------
          |               |               |
 HeuristicLlmClient  MockLlmClient  HttpLlmClient
                                          ^
                                    GroqLlmClient
                                    (compat alias)
```

`HttpLlmClient` holds a `ProviderSpec` (name, chat URL, default model, key
env, optional extra headers / auth header). `build_client()` reads `NNC_LLM`.
`chat_completions` POSTs OpenAI-style JSON and retries 408/429/5xx and
transport errors (1s / 2s / 4s). `propose_plan` is the correction loop
(schema + verifier, max 3, then heuristic).

Adding a vendor:

1. If it speaks `/v1/chat/completions`, `register_provider(ProviderSpec(...))`
   or set `NNC_LLM=custom` with `NNC_LLM_BASE_URL` and `NNC_LLM_MODEL`.
2. If the auth header is not `Authorization: Bearer`, set
   `NNC_LLM_AUTH_HEADER` / `NNC_LLM_AUTH_PREFIX` (Azure `api-key` is empty prefix).
3. If `response_format=json_object` is unsupported, `NNC_LLM_JSON_OBJECT=0`.

Do not add a second HTTP client class for a new OpenAI-compatible host.

## SOLID (what we actually did)

- One module, one job (prompt builder does not HTTP; the HTTP client does not write prompts).
- New pass / frontend / backend / LLM host registers; pipeline does not grow if-else on YOLO or Groq.
- Planner depends on the `LlmClient` protocol, not a vendor.
- `LoadedGraph` is the handle tests and the pipeline share.

## Not measured on this QEMU box

TensorRT, Jetson power, COCO mAP. Do not invent them. Gazebo camera Image
on `/nnc/camera/image_raw` is measured when the SITL stack is up (see
`docs/simulation.md`).
