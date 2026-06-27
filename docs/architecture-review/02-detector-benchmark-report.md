# Detector Modernization & Benchmark Report — EO/IR Mission Console

**Status:** Design + benchmark *framework* deliverable. No detector code changed yet;
no benchmarks run yet (requires the GPU box + datasets).
**Date:** 2026-06-23

> **Important:** YOLOv11 is **not** removed. The goal is a *detector abstraction* with
> YOLOv11 as the baseline and RF-DETR as a first-class, **config-selectable** alternative,
> plus a benchmark harness so the choice is made on measured data, not assumptions.

> **Context note:** A prior decision (2026-06-22) declined RF-DETR over the 8 GB OOM risk
> when it would have been a *third concurrent* model. That concern does **not** apply here:
> RF-DETR is a **swappable replacement** for YOLO (one detector loaded at a time), so peak
> VRAM is one detector + the VLM, same as today. The benchmark must still confirm RF-DETR's
> single-model VRAM fits alongside Qwen-3B in 4-bit.

---

## 1. Current state

`src/detection/detector.py` (79 LOC) is a thin `YOLODetector` wrapping `ultralytics.YOLO`:
- `__init__(config, device, half)` loads the model once.
- `detect(image) -> (List[Detection], sv.Detections)` returns framework-agnostic
  `Detection` objects **plus** a `supervision.Detections` container.
- `warmup()` does a dummy inference.

The rest of the pipeline already depends only on the **`Detection` dataclass** (`utils/types.py`)
and the `sv.Detections` handed to ByteTrack. That is the seam to formalize.

---

## 2. Detector abstraction interface

Introduce a backend-agnostic protocol; `YOLODetector` and `RFDetrDetector` both implement it.

```python
# src/detection/base.py  (proposed)
class Detector(Protocol):
    def detect(self, image: np.ndarray) -> tuple[list[Detection], "sv.Detections"]: ...
    def warmup(self, size: int = 640) -> None: ...
    @property
    def name(self) -> str: ...        # "yolo11m", "rf-detr-base", ...
    @property
    def classes(self) -> list[int] | None: ...
```

**Key design points:**
- **Return contract is fixed:** every backend returns the same `(List[Detection],
  sv.Detections)` pair, so tracker/ReID/fusion/events are untouched. The adapter converts
  the backend's native output into `Detection` + `sv.Detections`.
- **Class mapping:** both backends must emit a consistent class id/name space (the pipeline
  filters `config.detection.classes` as COCO ids). RF-DETR adapters map their label space to
  the same COCO ids; document any class that doesn't map 1:1.
- **Factory + config:** a `build_detector(config)` reads `detection.backend` and returns the
  right implementation. No code change to switch — only `config.yaml`.

### Config (additive, backward-compatible)
```yaml
detection:
  backend: "yolo11"          # "yolo11" | "rf_detr"      <-- NEW selector
  model_path: "yolo11m.pt"   # used when backend == yolo11
  conf_threshold: 0.30
  iou_threshold: 0.50
  img_size: 640
  classes: [0,1,2,3,4,5,6,7,8]
  rf_detr:                    # NEW, only read when backend == rf_detr
    variant: "base"          # base | large | nano (per RF-DETR release)
    weights: ""              # path or hub id
    img_size: 560            # RF-DETR native resolution
```

Default `backend: yolo11` keeps current behavior identical. `console.py`/`pipeline.py`/
`analyzers.py` change from `YOLODetector(...)` to `build_detector(config)` — a one-line swap
each (they already accept an injectable `detector=`).

---

## 3. RF-DETR integration plan (first-class backend)

1. Add `rf_detr` optional dependency group (isolated so YOLO-only installs stay light;
   aligns with the repo-split goal of lean environments).
2. Implement `RFDetrDetector(Detector)`:
   - Load weights/variant from `config.detection.rf_detr`.
   - Run inference, convert boxes/scores/labels → `Detection` + `sv.Detections`.
   - Honor `conf_threshold`, `classes` filtering, and `device`/`half`.
   - `warmup()` parity.
3. Verify the COCO class-id mapping matches YOLO's so downstream filters and labels are
   consistent across backends.
4. **VRAM check:** confirm RF-DETR (chosen variant) + Qwen2.5-VL-3B (4-bit) fits in 8 GB with
   headroom. If a variant exceeds budget, prefer a smaller RF-DETR variant or FP16.
5. Keep YOLOv11 as the default until benchmarks justify a switch.

---

## 4. Benchmark framework

A reproducible harness (`benchmarks/` or `tools/bench_detectors.py`) that runs each backend
over identical EO and IR workloads and emits a comparative report. **It must run on the
target RTX 5060 box** to be meaningful — these are device-specific numbers.

### 4.1 Workloads (identical inputs per backend)
- **EO set:** a fixed folder/video of EO frames (sparse, medium, dense/crowd).
- **IR set:** a fixed folder/video of IR/thermal frames.
- Same frames, same order, same `conf`/`img_size` policy per backend's native resolution.
- Optional labeled subset for accuracy (see mAP caveat below).

### 4.2 Metrics
| Metric | How measured |
|---|---|
| Inference latency (ms) p50/p95 | `torch.cuda.Event` around the forward pass (true GPU time) + wall-clock |
| Throughput (FPS) | frames / total inference time |
| End-to-end pipeline FPS | full `console`/`pipeline` loop with each backend (uses `01` §6 instrumentation) |
| GPU utilization % | NVML sampled during the run |
| VRAM peak (MB) | `torch.cuda.max_memory_allocated` + NVML |
| CPU % / process RSS | psutil |
| Detections/frame, by class | from `Detection` outputs |
| Latency contribution | detector ms as a fraction of total pipeline ms |
| **mAP / accuracy** | **only on a labeled set** — see caveat |

**mAP caveat:** mAP requires ground-truth labels. Without an annotated EO/IR set we can
report *relative* detection behavior (count, stability across frames, confidence
distributions, agreement/IoU between backends) but **not** absolute mAP. If a labeled
validation set exists or can be made (even small), the harness computes COCO-style mAP via
`supervision`/`pycocotools`. State clearly in any report whether mAP is real or N/A.

### 4.3 Output
- A `detector-benchmark-<date>.md` + CSV with the table above per backend per workload
  (EO sparse/medium/dense, IR), plus a recommendation.
- A short "selection guide": which backend for which scenario given the 8 GB budget and the
  latency targets from `03`.

### 4.4 Harness design (deterministic, fair)
- Warm up each backend before timing (exclude first-inference compilation).
- Same frame count; discard warmup frames; report p50/p95 not just mean.
- Run backends in separate processes (clean VRAM between runs).
- Pin versions; record torch/cuda/ultralytics/rf-detr versions in the report header.
- Optional `--profile` to attach the per-stage timing from `01` §6 so detector swap effect
  on *whole-pipeline* FPS is visible, not just raw inference.

---

## 5. Report template (to be filled after running on the box)

```
# Detector Benchmark — <date>, RTX 5060 8GB, torch <ver>/cu128
Backend      | Workload   | infer p50/p95 ms | FPS | GPU% | VRAM MB | CPU% | det/frame | mAP
yolo11m      | EO sparse  |                  |     |      |         |      |           | N/A
yolo11m      | EO dense   |                  |     |      |         |      |           |
yolo11m      | IR         |                  |     |      |         |      |           |
rf_detr-base | EO sparse  |                  |     |      |         |      |           |
rf_detr-base | EO dense   |                  |     |      |         |      |           |
rf_detr-base | IR         |                  |     |      |         |      |           |
End-to-end pipeline FPS with each backend (fusion off / fusion on): ...
Recommendation: ...
```

---

## 6. Acceptance criteria
- `Detector` protocol + `build_detector(config)` factory; YOLO path behaves identically to today.
- `RFDetrDetector` selectable purely via `config.yaml`, returns the same `(Detection, sv.Detections)` contract.
- Benchmark harness produces the §5 table on the target box for EO + IR workloads.
- VRAM peak for the chosen RF-DETR variant + Qwen-3B(4-bit) confirmed < 8 GB.
- Detector selection documented with measured evidence.

> This is detector *modernization through a clean interface + measured comparison*. It adds
> no AI features and changes no downstream module — the pipeline keeps consuming `Detection`.
