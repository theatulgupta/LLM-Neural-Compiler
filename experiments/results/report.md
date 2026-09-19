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

## Summary

| kind | task | baseline p50 | chosen origin | chosen plan | chosen p50 | speedup | passed | llm_source | llm_rank | llm_gap_pct | llm_followup | inputs_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolov8n | detect | 90.58196100158966 | preset:int8_dynamic | int8_dynamic | 61.83848400178249 | 1.465 | True | groq | 9 | 104.78323982875281 | improved:uav_dynamic_quant_extended p50=68.01968699801364 rank=3 | calib |
| yolo11n | detect | 78.48745699811843 | llm_revised | graph_fuse_ort_all | 76.3605739994091 | 1.028 | True | groq | None | 66.42196534845574 | revised:graph_fuse_ort_all p50=76.3605739994091 rank=1 | calib |
| yolov8n-pose | pose | 96.6111939997063 | heuristic | graph_fuse | 63.59092800266808 | 1.519 | True | heuristic | 2 | 0.0 |  | calib |
| yolov8n-seg | segment | 120.71993399877101 | preset:graph_fuse_ort | graph_fuse_ort | 120.01700900145806 | 1.006 | True | groq | 6 | 22.865553995813748 | improved:uav_companion_opt_dynamic_extended p50=89.71881899924483 rank=None | calib |
| ssdlite_mobilenetv3 | detect-lite | 20.715902999654645 | preset:graph_fuse_ort | graph_fuse_ort | 20.563694997690618 | 1.007 | True | groq | None | 268.51752570803365 | revised:graph_fuse_ort_threads p50=20.59831800215761 rank=2 | calib |
| mobilenetv3_small | classify | 4.1013049994944595 | preset:ort_default | ort_default | 2.8013139999529812 | 1.464 | True | groq | None | 196.4050085553112 | revised:skipped:fallback: LLM HTTP 429 (rate_limit_exceeded) | calib |
| midas_small | depth | 95.37040699797217 | preset:graph_fuse_ort | graph_fuse_ort | 89.39827300127945 | 1.067 | True | groq | 8 | 38.443188940492966 | improved:graph_fuse_ort_parallel p50=199.07179599977098 rank=9 | calib |

## Models

### yolov8n
- chosen origin `preset:int8_dynamic` plan `int8_dynamic` p50_ms `61.83848400178249` passed `True`
- llm source `groq` plan `uav_dynamic_quant_fuse` rank `9` oracle_gap_pct `104.78323982875281` fallback `None`
- llm.revised `None`
- llm.improved `{'p50_ms': 68.01968699801364, 'passed': True, 'plan_id': 'uav_dynamic_quant_extended', 'rank': 3}`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 90.58196100158966 | True | 263->263 | 12824019 |
| preset:ort_default | ort_default | 92.47553100067307 | True | 263->263 | 12824019 |
| preset:graph_simplify | graph_simplify | 91.00566000051913 | True | 263->263 | 12839897 |
| preset:graph_fuse | graph_fuse | 92.58902299916372 | True | 263->263 | 12839897 |
| preset:graph_fuse_ort | graph_fuse_ort | 92.88735199879739 | True | 263->263 | 12839897 |
| preset:int8_dynamic | int8_dynamic | 61.83848400178249 | True | 263->640 | 3492861 |
| preset:int8_static | int8_static | 124.04958399929455 | False | 263->878 | 3677978 |
| heuristic | graph_fuse | 63.884049999614945 | True | 263->640 | 3492861 |
| llm | uav_dynamic_quant_fuse | 126.63485099983518 | True | 263->640 | 3492861 |
| llm_improved | uav_dynamic_quant_extended | 68.01968699801364 | True | 263->640 | 3492861 |

### yolo11n
- chosen origin `llm_revised` plan `graph_fuse_ort_all` p50_ms `76.3605739994091` passed `True`
- llm source `groq` plan `uav_dynamic_int8_ort_extended` rank `None` oracle_gap_pct `66.42196534845574` fallback `None`
- llm.revised `{'p50_ms': 76.3605739994091, 'passed': True, 'plan_id': 'graph_fuse_ort_all', 'rank': 1}`
- llm.improved `None`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 78.48745699811843 | True | 355->355 | 10701774 |
| preset:ort_default | ort_default | 77.7842919997056 | True | 355->355 | 10701774 |
| preset:graph_simplify | graph_simplify | 80.7534380001016 | True | 355->355 | 10724317 |
| preset:graph_fuse | graph_fuse | 80.14244000150939 | True | 355->355 | 10724317 |
| preset:graph_fuse_ort | graph_fuse_ort | 76.48516900007962 | True | 355->355 | 10724317 |
| preset:int8_dynamic | int8_dynamic | 66.73322499773349 | False | 355->873 | 3032924 |
| preset:int8_static | int8_static | 117.4641249999695 | False | 355->1200 | 3299336 |
| heuristic | graph_fuse | 61.866498999734176 | False | 355->873 | 3032924 |
| llm | uav_dynamic_int8_ort_extended | 127.08076800117851 | False | 355->873 | 3032924 |
| llm_revised | graph_fuse_ort_all | 76.3605739994091 | True | 355->355 | 10724317 |

### yolov8n-pose
- chosen origin `heuristic` plan `graph_fuse` p50_ms `63.59092800266808` passed `True`
- llm source `heuristic` plan `graph_fuse` rank `2` oracle_gap_pct `0.0` fallback `LLM HTTP 429 (rate_limit_exceeded)`
- llm.revised `None`
- llm.improved `None`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 96.6111939997063 | True | 313->313 | 13484436 |
| preset:ort_default | ort_default | 94.67570400011027 | True | 313->313 | 13484436 |
| preset:graph_simplify | graph_simplify | 98.23830299865222 | True | 313->313 | 13503215 |
| preset:graph_fuse | graph_fuse | 99.1104210006597 | True | 313->313 | 13503215 |
| preset:graph_fuse_ort | graph_fuse_ort | 94.97519400247256 | True | 313->313 | 13503215 |
| preset:int8_dynamic | int8_dynamic | 67.2296999982791 | True | 313->741 | 3756078 |
| preset:int8_static | int8_static | 131.86724700062769 | False | 313->1020 | 3960552 |
| heuristic | graph_fuse | 63.59092800266808 | True | 313->741 | 3756078 |
| llm | graph_fuse | 63.59092800266808 | True | 313->741 | 3756078 |

### yolov8n-seg
- chosen origin `preset:graph_fuse_ort` plan `graph_fuse_ort` p50_ms `120.01700900145806` passed `True`
- llm source `groq` plan `uav_companion_opt_graph_simplify` rank `6` oracle_gap_pct `22.865553995813748` fallback `None`
- llm.revised `None`
- llm.improved `{'p50_ms': 89.71881899924483, 'passed': False, 'plan_id': 'uav_companion_opt_dynamic_extended', 'rank': None}`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 120.71993399877101 | True | 301->301 | 13842019 |
| preset:ort_default | ort_default | 122.26829599967459 | True | 301->301 | 13842019 |
| preset:graph_simplify | graph_simplify | 121.78404600126669 | True | 301->301 | 13860318 |
| preset:graph_fuse | graph_fuse | 126.06434799818089 | True | 301->301 | 13860318 |
| preset:graph_fuse_ort | graph_fuse_ort | 120.01700900145806 | True | 301->301 | 13860318 |
| preset:int8_dynamic | int8_dynamic | 81.66192399949068 | False | 301->746 | 3824939 |
| preset:int8_static | int8_static | 158.87716200086288 | False | 301->1012 | 3984242 |
| heuristic | graph_fuse | 79.5557240016933 | False | 301->746 | 3824939 |
| llm | uav_companion_opt_graph_simplify | 147.4595629988471 | True | 301->301 | 13860318 |
| llm_improved | uav_companion_opt_dynamic_extended | 89.71881899924483 | False | 301->746 | 3824939 |

### ssdlite_mobilenetv3
- chosen origin `preset:graph_fuse_ort` plan `graph_fuse_ort` p50_ms `20.563694997690618` passed `True`
- llm source `groq` plan `uav_graph_fuse_ort_dynamic` rank `None` oracle_gap_pct `268.51752570803365` fallback `None`
- llm.revised `{'p50_ms': 20.59831800215761, 'passed': True, 'plan_id': 'graph_fuse_ort_threads', 'rank': 2}`
- llm.improved `None`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 20.715902999654645 | True | 325->325 | 13788595 |
| preset:ort_default | ort_default | 21.500230999663472 | True | 325->325 | 13788595 |
| preset:graph_simplify | graph_simplify | 20.94002699959674 | True | 325->325 | 13817915 |
| preset:graph_fuse | graph_fuse | 20.99610900040716 | True | 325->306 | 13815029 |
| preset:graph_fuse_ort | graph_fuse_ort | 20.563694997690618 | True | 325->306 | 13815029 |
| preset:int8_dynamic | int8_dynamic | 45.261111998115666 | False | 325->902 | 3834704 |
| preset:int8_static | int8_static | 31.997909001802327 | False | 325->900 | 4200207 |
| heuristic | graph_fuse | 32.235073998890584 | False | 325->769 | 5082100 |
| llm | uav_graph_fuse_ort_dynamic | 75.78081999963615 | False | 325->769 | 5082100 |
| llm_revised | graph_fuse_ort_threads | 20.59831800215761 | True | 325->306 | 13815029 |

### mobilenetv3_small
- chosen origin `preset:ort_default` plan `ort_default` p50_ms `2.8013139999529812` passed `True`
- llm source `groq` plan `uav_companion_opt_v2` rank `None` oracle_gap_pct `196.4050085553112` fallback `None`
- llm.revised `{'skipped': 'fallback: LLM HTTP 429 (rate_limit_exceeded)'}`
- llm.improved `None`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 4.1013049994944595 | True | 141->141 | 10185813 |
| preset:ort_default | ort_default | 2.8013139999529812 | True | 141->141 | 10185813 |
| preset:graph_simplify | graph_simplify | 2.8853139992861543 | True | 141->141 | 10197728 |
| preset:graph_fuse | graph_fuse | 2.911063998908503 | True | 141->127 | 10196209 |
| preset:graph_fuse_ort | graph_fuse_ort | 2.821522000886034 | True | 141->127 | 10196209 |
| preset:int8_dynamic | int8_dynamic | 8.69039899771451 | False | 141->463 | 2733411 |
| preset:int8_static | int8_static | 8.184901998902205 | False | 141->491 | 2915414 |
| heuristic | graph_fuse | 3.2476859996677376 | True | 141->127 | 10196209 |
| llm | uav_companion_opt_v2 | 8.303235001221765 | False | 141->365 | 3406521 |

### midas_small
- chosen origin `preset:graph_fuse_ort` plan `graph_fuse_ort` p50_ms `89.39827300127945` passed `True`
- llm source `groq` plan `uav_companion_opt` rank `8` oracle_gap_pct `38.443188940492966` fallback `None`
- llm.revised `None`
- llm.improved `{'p50_ms': 199.07179599977098, 'passed': True, 'plan_id': 'graph_fuse_ort_parallel', 'rank': 9}`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 95.37040699797217 | True | 645->645 | 66384550 |
| preset:ort_default | ort_default | 90.46116200261167 | True | 645->645 | 66384550 |
| preset:graph_simplify | graph_simplify | 94.39144100178964 | True | 645->645 | 66431778 |
| preset:graph_fuse | graph_fuse | 93.2386200001929 | True | 645->636 | 66431112 |
| preset:graph_fuse_ort | graph_fuse_ort | 89.39827300127945 | True | 645->636 | 66431112 |
| preset:int8_dynamic | int8_dynamic | 131.04470100006438 | False | 645->1216 | 17026816 |
| preset:int8_static | int8_static | 109.41294199801632 | True | 645->1074 | 17556222 |
| heuristic | graph_fuse | 91.36891000161995 | True | 645->636 | 66431112 |
| llm | uav_companion_opt | 123.76582000069902 | True | 645->1050 | 29278480 |
| llm_improved | graph_fuse_ort_parallel | 199.07179599977098 | True | 645->636 | 66431112 |

## Gazebo camera loop

| file | kind | variant | frames_rx | inferences | e2e p50 | e2e p95 | infer p50 | infer p95 | detections_total | ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sim_inference_midas_small_chosen.json | midas_small | chosen | 61 | 52 | 157.3960250007076 | 271.425315999295 | 154.17748799973197 | 264.25253599973075 | 0 | True |
| sim_inference_midas_small_native.json | midas_small | native | 61 | 53 | 135.8996279996063 | 280.6825480001862 | 130.4983859999993 | 274.27343300041684 | 0 | True |
| sim_inference_mobilenetv3_small_chosen.json | mobilenetv3_small | chosen | 47 | 41 | 7.067614999868965 | 8.072030999755953 | 4.3122849992869305 | 5.187077000300633 | 0 | True |
| sim_inference_mobilenetv3_small_native.json | mobilenetv3_small | native | 53 | 43 | 9.447780000300554 | 15.119650000087859 | 6.772658999580017 | 12.35994499984372 | 0 | True |
| sim_inference_ssdlite_mobilenetv3_chosen.json | ssdlite_mobilenetv3 | chosen | 47 | 55 | 26.951558999826375 | 38.58992500045133 | 20.74435499980609 | 29.83438999945065 | 55 | True |
| sim_inference_ssdlite_mobilenetv3_native.json | ssdlite_mobilenetv3 | native | 51 | 49 | 26.951066000037827 | 43.95097300039197 | 21.300321000126132 | 36.064977000023646 | 49 | True |
| sim_inference_yolo11n_chosen.json | yolo11n | chosen | 62 | 46 | 56.817229999978736 | 82.53475800029264 | 51.58587799996894 | 73.73591499981558 | 138 | True |
| sim_inference_yolo11n_native.json | yolo11n | native | 54 | 61 | 120.46312599977682 | 212.9903770000965 | 112.70539900033327 | 204.2113210000025 | 244 | True |
| sim_inference_yolov8n-pose_chosen.json | yolov8n-pose | chosen | 57 | 55 | 57.89817999993829 | 75.91796900032932 | 53.86205600007088 | 71.44405200051551 | 110 | True |
| sim_inference_yolov8n-pose_native.json | yolov8n-pose | native | 61 | 57 | 171.34817499936617 | 251.95113100016897 | 164.56612999991194 | 245.8541680007329 | 114 | True |
| sim_inference_yolov8n-seg_chosen.json | yolov8n-seg | chosen | 59 | 57 | 230.92708500007575 | 292.98190899953624 | 221.47409799981688 | 284.2243289996986 | 228 | True |
| sim_inference_yolov8n-seg_native.json | yolov8n-seg | native | 67 | 64 | 260.528293999414 | 329.4929209996553 | 252.697351000279 | 317.4055449999287 | 256 | True |
| sim_inference_yolov8n_chosen.json | yolov8n | chosen | 43 | 45 | 55.386526000802405 | 85.21933700012596 | 50.48789299962664 | 78.27769800042006 | 88 | True |
| sim_inference_yolov8n_native.json | yolov8n | native | 67 | 68 | 183.47336499937228 | 279.2707190001238 | 175.72247200041602 | 267.18727099978423 | 136 | True |

## Not measured

- TensorRT / NVIDIA GPU
- Board power
- COCO mAP (agreement vs FP32 reference is reported instead)

