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

| kind | task | baseline p50 | chosen plan | chosen p50 | speedup | passed | llm_rank | llm_gap_pct | inputs_source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolov8n | detect | 53.89802299987423 | baseline | 53.89802299987423 | 1.000 | True | None | None | calib |
| yolo11n | detect | 80.78582899997855 | graph_fuse | 57.65770700008943 | 1.401 | True | None | None | calib |
| yolov8n-pose | pose | 98.90571300002193 | graph_fuse | 60.21018699993874 | 1.643 | True | None | None | calib |
| yolov8n-seg | segment | 119.17291400004615 | graph_fuse_ort | 114.93083300001672 | 1.037 | True | None | None | calib |
| ssdlite_mobilenetv3 | detect-lite | 19.20123400009288 | ort_default | 18.34838300010233 | 1.046 | True | None | None | calib |
| mobilenetv3_small | classify | 3.3675540000785986 | graph_fuse | 2.661255000020901 | 1.265 | True | None | None | calib |
| midas_small | depth | 91.91707300010421 | graph_fuse | 84.45504399992387 | 1.088 | True | None | None | calib |

## Models

### yolov8n
- chosen origin `preset:baseline` plan `baseline` p50_ms `53.89802299987423` passed `True`
- llm_rank `None` oracle_gap_pct `None`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 53.89802299987423 | True | 263->263 | 12824019 |
| preset:ort_default | ort_default | 101.29531799998404 | True | 263->263 | 12824019 |
| preset:graph_simplify | graph_simplify | 152.14219899985437 | True | 263->263 | 12839897 |
| preset:graph_fuse | graph_fuse | 91.41238799998064 | True | 263->263 | 12839897 |
| preset:graph_fuse_ort | graph_fuse_ort | 89.62552399998458 | True | 263->263 | 12839897 |
| preset:int8_dynamic | int8_dynamic | 63.96301500012669 | True | 263->640 | 3492861 |
| preset:int8_static | int8_static | 119.65375700015102 | True | 263->878 | 3677987 |
| heuristic | graph_fuse | 55.244436000066344 | True | 263->640 | 3492861 |

### yolo11n
- chosen origin `heuristic` plan `graph_fuse` p50_ms `57.65770700008943` passed `True`
- llm_rank `None` oracle_gap_pct `None`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 80.78582899997855 | True | 355->355 | 10701774 |
| preset:ort_default | ort_default | 78.27947999999196 | True | 355->355 | 10701774 |
| preset:graph_simplify | graph_simplify | 81.37166300002718 | True | 355->355 | 10724317 |
| preset:graph_fuse | graph_fuse | 79.57110000006651 | True | 355->355 | 10724317 |
| preset:graph_fuse_ort | graph_fuse_ort | 79.99970199989548 | True | 355->355 | 10724317 |
| preset:int8_dynamic | int8_dynamic | 68.80988700004309 | True | 355->873 | 3032924 |
| preset:int8_static | int8_static | 121.97116200013625 | False | 355->1200 | 3299300 |
| heuristic | graph_fuse | 57.65770700008943 | True | 355->873 | 3032924 |

### yolov8n-pose
- chosen origin `heuristic` plan `graph_fuse` p50_ms `60.21018699993874` passed `True`
- llm_rank `None` oracle_gap_pct `None`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 98.90571300002193 | True | 313->313 | 13484436 |
| preset:ort_default | ort_default | 97.81851399998231 | True | 313->313 | 13484436 |
| preset:graph_simplify | graph_simplify | 109.91006599988395 | True | 313->313 | 13503215 |
| preset:graph_fuse | graph_fuse | 101.09316100010801 | True | 313->313 | 13503215 |
| preset:graph_fuse_ort | graph_fuse_ort | 98.49037100002533 | True | 313->313 | 13503215 |
| preset:int8_dynamic | int8_dynamic | 74.32005000009667 | True | 313->741 | 3756078 |
| preset:int8_static | int8_static | 136.43084599993927 | True | 313->1020 | 3960606 |
| heuristic | graph_fuse | 60.21018699993874 | True | 313->741 | 3756078 |

### yolov8n-seg
- chosen origin `preset:graph_fuse_ort` plan `graph_fuse_ort` p50_ms `114.93083300001672` passed `True`
- llm_rank `None` oracle_gap_pct `None`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 119.17291400004615 | True | 301->301 | 13842019 |
| preset:ort_default | ort_default | 122.30308300013348 | True | 301->301 | 13842019 |
| preset:graph_simplify | graph_simplify | 117.26457899999332 | True | 301->301 | 13860318 |
| preset:graph_fuse | graph_fuse | 118.78605099991546 | True | 301->301 | 13860318 |
| preset:graph_fuse_ort | graph_fuse_ort | 114.93083300001672 | True | 301->301 | 13860318 |
| preset:int8_dynamic | int8_dynamic | 91.2776010000016 | False | 301->746 | 3824939 |
| preset:int8_static | int8_static | 153.2372750000377 | False | 301->1012 | 3984215 |
| heuristic | graph_fuse | 71.79628200015031 | False | 301->746 | 3824939 |

### ssdlite_mobilenetv3
- chosen origin `preset:ort_default` plan `ort_default` p50_ms `18.34838300010233` passed `True`
- llm_rank `None` oracle_gap_pct `None`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 19.20123400009288 | True | 325->325 | 13788595 |
| preset:ort_default | ort_default | 18.34838300010233 | True | 325->325 | 13788595 |
| preset:graph_simplify | graph_simplify | 19.636926999965 | True | 325->325 | 13817915 |
| preset:graph_fuse | graph_fuse | 19.663750000063374 | True | 325->306 | 13815029 |
| preset:graph_fuse_ort | graph_fuse_ort | 18.91002699994715 | True | 325->306 | 13815029 |
| preset:int8_dynamic | int8_dynamic | 45.83977800007233 | False | 325->902 | 3834704 |
| preset:int8_static | int8_static | 33.69581300012214 | False | 325->900 | 4200216 |
| heuristic | graph_fuse | 27.523399000074278 | False | 325->769 | 5082100 |

### mobilenetv3_small
- chosen origin `heuristic` plan `graph_fuse` p50_ms `2.661255000020901` passed `True`
- llm_rank `None` oracle_gap_pct `None`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 3.3675540000785986 | True | 141->141 | 10185813 |
| preset:ort_default | ort_default | 3.1737609999709093 | True | 141->141 | 10185813 |
| preset:graph_simplify | graph_simplify | 3.3795529998315033 | True | 141->141 | 10197728 |
| preset:graph_fuse | graph_fuse | 3.2482189999427646 | True | 141->127 | 10196209 |
| preset:graph_fuse_ort | graph_fuse_ort | 3.5934269999415847 | True | 141->127 | 10196209 |
| preset:int8_dynamic | int8_dynamic | 12.059323999892513 | False | 141->463 | 2733411 |
| preset:int8_static | int8_static | 6.219512999905419 | False | 141->491 | 2915432 |
| heuristic | graph_fuse | 2.661255000020901 | True | 141->127 | 10196209 |

### midas_small
- chosen origin `heuristic` plan `graph_fuse` p50_ms `84.45504399992387` passed `True`
- llm_rank `None` oracle_gap_pct `None`

| origin | plan | p50_ms | passed | nodes | bytes |
| --- | --- | --- | --- | --- | --- |
| preset:baseline | baseline | 91.91707300010421 | True | 645->645 | 66384550 |
| preset:ort_default | ort_default | 85.36795499981054 | True | 645->645 | 66384550 |
| preset:graph_simplify | graph_simplify | 90.04562800009808 | True | 645->645 | 66431778 |
| preset:graph_fuse | graph_fuse | 92.30430000002343 | True | 645->636 | 66431112 |
| preset:graph_fuse_ort | graph_fuse_ort | 89.02613100008239 | True | 645->636 | 66431112 |
| preset:int8_dynamic | int8_dynamic | 129.75099899995257 | False | 645->1216 | 17026816 |
| preset:int8_static | int8_static | 109.67133399981321 | True | 645->1074 | 17556186 |
| heuristic | graph_fuse | 84.45504399992387 | True | 645->636 | 66431112 |

## Gazebo camera loop

| file | kind | variant | frames_rx | inferences | e2e p50 | e2e p95 | infer p50 | infer p95 | detections_total | ok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sim_inference_midas_small_chosen.json | midas_small | chosen | 58 | 44 | 57.599399000082485 | 62.116332999721635 | 55.58787800009668 | 60.204496999631374 | 0 | True |
| sim_inference_midas_small_native.json | midas_small | native | 57 | 56 | 78.04595800007519 | 155.79986700004156 | 75.10958900002151 | 153.27445799994166 | 0 | True |
| sim_inference_mobilenetv3_small_chosen.json | mobilenetv3_small | chosen | 62 | 55 | 4.623987999821111 | 5.172205000235408 | 2.787209999951301 | 3.3608439998715767 | 0 | True |
| sim_inference_mobilenetv3_small_native.json | mobilenetv3_small | native | 70 | 54 | 9.433422000256542 | 16.7321419999098 | 7.258965000346507 | 14.241012000184128 | 0 | True |
| sim_inference_ssdlite_mobilenetv3_chosen.json | ssdlite_mobilenetv3 | chosen | 51 | 54 | 27.053807000129382 | 57.756031999815605 | 21.363953999980367 | 44.640750000098706 | 54 | True |
| sim_inference_ssdlite_mobilenetv3_native.json | ssdlite_mobilenetv3 | native | 57 | 46 | 24.703286999965712 | 47.331963000033284 | 19.5098889998917 | 42.084855999974025 | 46 | True |
| sim_inference_yolo11n_chosen.json | yolo11n | chosen | 57 | 55 | 63.21069999989959 | 69.82966299983673 | 57.603239999934885 | 63.69919500002652 | 55 | True |
| sim_inference_yolo11n_native.json | yolo11n | native | 51 | 45 | 129.83046699991974 | 237.15792899997723 | 122.60382199997366 | 230.13115699995979 | 45 | True |
| sim_inference_yolov8n-pose_chosen.json | yolov8n-pose | chosen | 43 | 38 | 63.32261800002925 | 79.50092500004757 | 58.54629999998906 | 75.33505699984744 | 0 | True |
| sim_inference_yolov8n-pose_native.json | yolov8n-pose | native | 50 | 51 | 151.36404099985157 | 260.0554939999711 | 146.96329099979266 | 238.6998469999071 | 0 | True |
| sim_inference_yolov8n-seg_chosen.json | yolov8n-seg | chosen | 62 | 49 | 164.74806900009753 | 250.2326839999114 | 158.79041900006996 | 242.97288300022046 | 49 | True |
| sim_inference_yolov8n-seg_native.json | yolov8n-seg | native | 49 | 52 | 170.82349299994348 | 275.24745599998823 | 164.54190600006768 | 268.1048050001209 | 52 | True |
| sim_inference_yolov8n_baseline.json | yolov8n | baseline | 0 | 3 | None | None | None | None | 0 | True |
| sim_inference_yolov8n_chosen.json | yolov8n | chosen | 57 | 46 | 146.26438999994207 | 198.30511899999692 | 140.7117940000262 | 179.53863799993997 | 0 | True |
| sim_inference_yolov8n_native.json | yolov8n | native | 39 | 41 | 80.3854240000419 | 143.87749299999086 | 76.56775300006302 | 139.43020500005332 | 0 | True |

## Not measured

- TensorRT / NVIDIA GPU
- Board power
- COCO mAP (agreement vs FP32 reference is reported instead)

