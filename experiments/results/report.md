# LLM-guided compiler report

Numbers below are measured on this host. `fps_claimed` is false.

## Hardware

```json
{
  "machine": "aarch64",
  "machine_id": "e996b2604c4c49ada2a0d0e9e55b47d5",
  "nvidia": {
    "detail": "no NVIDIA device: nvidia-smi not on PATH and /dev/nvidia0 missing (this host is aarch64 QEMU with a Virtio GPU)",
    "present": false
  },
  "python": "3.12.3",
  "uname_m": "aarch64"
}
```

## Models

### mobilenetv3_small
- chosen origin `heuristic` plan `graph_fuse` p50_ms `2.1933000007265946` passed `True`
- llm_rank `None` oracle_gap_pct `None`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 8.968125999672338 | True | 141->141 | 10185813 |
| preset:ort_default | ort_default | 4.3195979997108225 | True | 141->141 | 10185813 |
| preset:graph_fuse | graph_fuse | 4.466312000658945 | True | 141->127 | 10196209 |
| heuristic | graph_fuse | 2.1933000007265946 | True | 141->127 | 10196209 |

## Not measured

- TensorRT / NVIDIA GPU
- Board power
- COCO mAP (agreement vs FP32 reference is reported instead)

