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

| kind | task | baseline p50 | chosen plan | chosen p50 | speedup | passed | llm_source | llm_rank | llm_gap_pct | inputs_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolov8n | detect | 50.777794999703474 | graph_fuse | 23.52122699994652 | 2.159 | True | groq | 9 | 183.2854425483645 | calib |
| yolo11n | detect | 44.586764999621664 | graph_fuse | 26.48174700061645 | 1.684 | True | groq | 8 | 163.17461230385047 | calib |
| yolov8n-pose | pose | 57.239376999859815 | graph_fuse | 27.7956970003288 | 2.059 | True | groq | 9 | 179.8861348902459 | calib |
| yolov8n-seg | segment | 69.64457499998389 | ort_default | 69.30274499973166 | 1.005 | True | groq | None | 148.7980324597315 | calib |
| ssdlite_mobilenetv3 | detect-lite | 21.483624999746098 | ort_default | 20.917545999509457 | 1.027 | True | groq | None | 274.32280536854125 | calib |
| mobilenetv3_small | classify | 4.092217999641434 | graph_fuse | 2.906144000007771 | 1.408 | True | heuristic | 6 | 14.010455078082604 | calib |
| midas_small | depth | 94.13615399989794 | graph_fuse_ort | 91.08464000019012 | 1.034 | True | groq | 8 | 51.047700248592754 | calib |

## Models

### yolov8n
- chosen origin `heuristic` plan `graph_fuse` p50_ms `23.52122699994652` passed `True`
- llm source `groq` plan `dynamic_quantization` rank `9` oracle_gap_pct `183.2854425483645` fallback `None`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 50.777794999703474 | True | 263->263 | 12824019 |
| preset:ort_default | ort_default | 52.08990899973287 | True | 263->263 | 12824019 |
| preset:graph_simplify | graph_simplify | 50.985876999220636 | True | 263->263 | 12839897 |
| preset:graph_fuse | graph_fuse | 50.91154500041739 | True | 263->263 | 12839897 |
| preset:graph_fuse_ort | graph_fuse_ort | 50.649838999561325 | True | 263->263 | 12839897 |
| preset:int8_dynamic | int8_dynamic | 32.654610000463435 | True | 263->640 | 3492861 |
| preset:int8_static | int8_static | 65.92367499979446 | True | 263->878 | 3677987 |
| heuristic | graph_fuse | 23.52122699994652 | True | 263->640 | 3492861 |
| llm | dynamic_quantization | 66.6322119996039 | True | 263->640 | 3492861 |

### yolo11n
- chosen origin `heuristic` plan `graph_fuse` p50_ms `26.48174700061645` passed `True`
- llm source `groq` plan `dynamic_int8_opt` rank `8` oracle_gap_pct `163.17461230385047` fallback `None`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 44.586764999621664 | True | 355->355 | 10701774 |
| preset:ort_default | ort_default | 43.28069300026982 | True | 355->355 | 10701774 |
| preset:graph_simplify | graph_simplify | 45.09005399995658 | True | 355->355 | 10724317 |
| preset:graph_fuse | graph_fuse | 45.40246800024761 | True | 355->355 | 10724317 |
| preset:graph_fuse_ort | graph_fuse_ort | 44.650767000348424 | True | 355->355 | 10724317 |
| preset:int8_dynamic | int8_dynamic | 36.285958999542345 | True | 355->873 | 3032924 |
| preset:int8_static | int8_static | 63.62007400002767 | False | 355->1200 | 3299300 |
| heuristic | graph_fuse | 26.48174700061645 | True | 355->873 | 3032924 |
| llm | dynamic_int8_opt | 69.6932350001589 | True | 355->873 | 3032924 |

### yolov8n-pose
- chosen origin `heuristic` plan `graph_fuse` p50_ms `27.7956970003288` passed `True`
- llm source `groq` plan `dynamic_int8_opt` rank `9` oracle_gap_pct `179.8861348902459` fallback `None`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 57.239376999859815 | True | 313->313 | 13484436 |
| preset:ort_default | ort_default | 55.6245569996463 | True | 313->313 | 13484436 |
| preset:graph_simplify | graph_simplify | 57.17550300050789 | True | 313->313 | 13503215 |
| preset:graph_fuse | graph_fuse | 57.766373999584175 | True | 313->313 | 13503215 |
| preset:graph_fuse_ort | graph_fuse_ort | 58.34837000020343 | True | 313->313 | 13503215 |
| preset:int8_dynamic | int8_dynamic | 37.74282500035042 | True | 313->741 | 3756078 |
| preset:int8_static | int8_static | 73.47925100020802 | True | 313->1020 | 3960606 |
| heuristic | graph_fuse | 27.7956970003288 | True | 313->741 | 3756078 |
| llm | dynamic_int8_opt | 77.7963020000243 | True | 313->741 | 3756078 |

### yolov8n-seg
- chosen origin `preset:ort_default` plan `ort_default` p50_ms `69.30274499973166` passed `True`
- llm source `groq` plan `uav_companion_opt` rank `None` oracle_gap_pct `148.7980324597315` fallback `None`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 69.64457499998389 | True | 301->301 | 13842019 |
| preset:ort_default | ort_default | 69.30274499973166 | True | 301->301 | 13842019 |
| preset:graph_simplify | graph_simplify | 69.33520399979898 | True | 301->301 | 13860318 |
| preset:graph_fuse | graph_fuse | 69.65841100009129 | True | 301->301 | 13860318 |
| preset:graph_fuse_ort | graph_fuse_ort | 75.25866000014503 | True | 301->301 | 13860318 |
| preset:int8_dynamic | int8_dynamic | 49.61719400034781 | False | 301->746 | 3824939 |
| preset:int8_static | int8_static | 167.93893599970033 | False | 301->1012 | 3984215 |
| heuristic | graph_fuse | 85.45808199960447 | False | 301->746 | 3824939 |
| llm | uav_companion_opt | 172.42386599991733 | False | 301->746 | 3824939 |

### ssdlite_mobilenetv3
- chosen origin `preset:ort_default` plan `ort_default` p50_ms `20.917545999509457` passed `True`
- llm source `groq` plan `graph_fuse_ort_quant_dyn` rank `None` oracle_gap_pct `274.32280536854125` fallback `None`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 21.483624999746098 | True | 325->325 | 13788595 |
| preset:ort_default | ort_default | 20.917545999509457 | True | 325->325 | 13788595 |
| preset:graph_simplify | graph_simplify | 21.012920000430313 | True | 325->325 | 13817915 |
| preset:graph_fuse | graph_fuse | 21.164419000342605 | True | 325->306 | 13815029 |
| preset:graph_fuse_ort | graph_fuse_ort | 21.002337000027183 | True | 325->306 | 13815029 |
| preset:int8_dynamic | int8_dynamic | 45.41343999972014 | False | 325->902 | 3834704 |
| preset:int8_static | int8_static | 34.10332000021299 | False | 325->900 | 4200216 |
| heuristic | graph_fuse | 34.06490399993345 | False | 325->769 | 5082100 |
| llm | graph_fuse_ort_quant_dyn | 78.29914499961887 | False | 325->769 | 5082100 |
| llm_revised | graph_fuse_ort_v2 | 57.22310100009054 | True | 325->306 | 13815029 |

### mobilenetv3_small
- chosen origin `preset:graph_fuse` plan `graph_fuse` p50_ms `2.906144000007771` passed `True`
- llm source `heuristic` plan `graph_fuse` rank `6` oracle_gap_pct `14.010455078082604` fallback `LLM HTTP 429 (rate_limit_exceeded)`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 4.092217999641434 | True | 141->141 | 10185813 |
| preset:ort_default | ort_default | 2.9079770001771976 | True | 141->141 | 10185813 |
| preset:graph_simplify | graph_simplify | 2.9080610001983587 | True | 141->141 | 10197728 |
| preset:graph_fuse | graph_fuse | 2.906144000007771 | True | 141->127 | 10196209 |
| preset:graph_fuse_ort | graph_fuse_ort | 3.0152270001053694 | True | 141->127 | 10196209 |
| preset:int8_dynamic | int8_dynamic | 8.805391000350937 | False | 141->463 | 2733411 |
| preset:int8_static | int8_static | 8.951639000770228 | False | 141->491 | 2915432 |
| heuristic | graph_fuse | 3.3133079996332526 | True | 141->127 | 10196209 |
| llm | graph_fuse | 3.3133079996332526 | True | 141->127 | 10196209 |

### midas_small
- chosen origin `preset:graph_fuse_ort` plan `graph_fuse_ort` p50_ms `91.08464000019012` passed `True`
- llm source `groq` plan `graph_fuse_ort_static` rank `8` oracle_gap_pct `51.047700248592754` fallback `None`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 94.13615399989794 | True | 645->645 | 66384550 |
| preset:ort_default | ort_default | 97.57766999973683 | True | 645->645 | 66384550 |
| preset:graph_simplify | graph_simplify | 96.01747599936061 | True | 645->645 | 66431778 |
| preset:graph_fuse | graph_fuse | 93.59132800000225 | True | 645->636 | 66431112 |
| preset:graph_fuse_ort | graph_fuse_ort | 91.08464000019012 | True | 645->636 | 66431112 |
| preset:int8_dynamic | int8_dynamic | 133.93156599977374 | False | 645->1216 | 17026816 |
| preset:int8_static | int8_static | 116.56899199988402 | True | 645->1074 | 17556186 |
| heuristic | graph_fuse | 119.3248899999162 | True | 645->636 | 66431112 |
| llm | graph_fuse_ort_static | 137.581253999997 | True | 645->1050 | 29278444 |

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

