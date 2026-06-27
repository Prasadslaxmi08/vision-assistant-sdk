# 👁️ Vision Assistant SDK

A reusable **Computer-Vision + Multimodal-AI engine** that plugs into any existing
camera or surveillance software. Camera vendors already provide RTSP, PTZ, zoom,
recording, playback, maps and UI — **Vision Assistant is only the AI layer**:
detection, tracking, EO/IR fusion, and vision-language reasoning, exposed through
JSON / callbacks / overlay rendering.

A lightweight **PySide6 desktop app** demonstrates the engine. It is *not* the
product — it just shows the SDK working across the three input modes.

> Long-term mission management (persistent memory, threat scoring, query assistant,
> report generation, RAG, agentic workflows) is **out of scope** for this repo and
> lives in a separate **AI Mission Analyst** project. The code for it is staged in
> [`future-ai-mission-analyst/`](future-ai-mission-analyst/).

---

## Architecture

```
                        VISION ASSISTANT SDK
┌───────────── Input Layer ─────────────────────────────────────────┐
│   Image   │   Video (EO / IR)   │   RTSP (EO + IR, independent)    │
└───────────────────┬───────────────────────────────────────────────┘
                    ▼
┌───────────── AI Core ─────────────────────────────────────────────┐
│   Detector (RF-DETR primary · abstraction)  →  ByteTrack          │
│   EO/IR Fusion (gated: both sensors only)   →  Vision-Language VLM │
└───────────────────┬───────────────────────────────────────────────┘
                    ▼
┌───────────── Intelligence Layer ──────────────────────────────────┐
│   Event generation · scene understanding · rolling summaries      │
└───────────────────┬───────────────────────────────────────────────┘
                    ▼
┌───────────── Output Layer ────────────────────────────────────────┐
│   JSON   ·   callbacks   ·   overlay rendering   ·   scene_json()  │
└───────────────────────────────────────────────────────────────────┘
```

### Real-time contract (highest priority)
The live feed is **never blocked by AI inference.** The render path shows the
freshest raw frame immediately; detection / tracking / fusion / VLM run on a
background inference thread and publish *overlay data* (coordinates, not pixels)
by a single atomic reference swap. Overlays may lag a frame or two; **the video
stays smooth no matter how slow inference is.**

### Input modes & EO/IR fusion
- **Live** — one or two **independent** RTSP streams (EO and/or IR).
- **Video** — EO-only or IR-only file.
- **Image** — single EO/IR image, or an EO+IR pair.

**Fusion is hard-gated on both sensors being present** (`fusion_available`). The
fusion engine is **sensor-agnostic**: it estimates the IR→EO mapping online from
shared targets (RANSAC, no calibration), so any EO/IR camera pair works.

---

## Install

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows  (source .venv/bin/activate on Linux)

# 1. PyTorch FIRST, from the CUDA 12.8 index (required for RTX 50-series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. The SDK + desktop demo
pip install -r requirements.txt

# 3. The PRIMARY detector (RF-DETR). YOLOv11 is already installed as the fallback.
pip install -r requirements-rfdetr.txt
```

> **8 GB VRAM:** the default config runs RF-DETR + Qwen2.5-VL-3B (4-bit). Confirm
> the RF-DETR variant fits alongside the VLM on your box; if VRAM is tight, set
> `detection.backend: "yolo11"` in `config/config.yaml` (instant fallback, no
> reinstall) and/or lower `vlm.max_pixels`.

---

## Run the demo

```bash
python -m desktop          # PySide6 single-window demo (Live / Video / Image)
```

Pick a mode, enter sources, and Start. In **Live** mode, fill the EO and IR RTSP
URLs and tick *Enable EO/IR fusion* to correlate thermal contacts onto the EO view.

### Headless CLI

```bash
# Single EO RTSP stream, 60 s, stream JSON results
python run_cli.py --eo rtsp://user:pass@host:554/eo --duration 60 --json

# Dual EO + IR with fusion
python run_cli.py --eo rtsp://host/eo --ir rtsp://host/ir --fusion

# A local video treated as IR, no preview window
python run_cli.py --eo clip.mp4 --video --modality IR --no-display
```

---

## Embed the SDK

```python
from src.config.settings import get_config
from src.engine import VisionAssistant

va = VisionAssistant(get_config())
va.add_source("EO", "rtsp://cam/eo")
va.add_source("IR", "rtsp://cam/ir")
va.set_fusion(True)                       # only acts when both streams are live
va.set_classes({"person", "car", "boat"}) # report only these COCO classes (None = all)
va.on_result(lambda r: print(r["summary"], r["events"]))   # Output Layer callback
va.start()

frame = va.frame_view().eo_frame          # overlay-composited live frame (BGR)
scene = va.scene_json()                   # JSON-serialisable scene snapshot
va.stop()
```

Offline analysis uses the same AI Core:

```python
from src.engine import ImageAnalyzer, VideoAnalyzer
report = ImageAnalyzer(cfg).analyze_pair(eo_img, ir_img)   # fused EO+IR assessment
out    = VideoAnalyzer(cfg).analyze("clip.mp4", persist=True)
```

**Output Layer surfaces:** `on_result(cb)` (per-event callbacks), `scene_json()`
(full snapshot), `frame_view()` (overlay frames), `overlay_payload()` (compact box
JSON for an external/browser renderer), `export_scene()` (Markdown + JSON to disk).

### Already own your camera? Push frames in.

If your application already decodes frames (its own RTSP/camera SDK/capture card),
use the **push** input instead of letting the SDK open a stream — the host stays in
full control of capture, the SDK is just the AI layer:

```python
va = VisionAssistant(get_config())
va.add_frame_source("EO")                 # a push source (no RTSP opened by the SDK)
va.on_result(lambda r: print(r["summary"]))
va.start()
va.submit_frame("EO", frame)              # feed each decoded BGR frame (your thread)
for d in va.overlay_payload()["eo"]["dets"]:
    draw_box(d["b"], f'{d["c"]}#{d["id"]}')   # draw on your own video widget
va.stop()
```

---

## Integration into an existing GCS / VMS

The SDK is built to embed into existing surveillance software with a few lines.
A complete reference — embedding it into a real PyQt/Tkinter operator station
behind **one button**, with live AI overlays, an intelligence brief, EO/IR fusion
and a one-click PDF mission report — is documented end to end:

- **Guide:** [`docs/integration/INTEGRATION.md`](docs/integration/INTEGRATION.md)
  (architecture diagram, the four-call contract, GPU tuning)
- **Runnable example:** [`examples/tkinter_gcs/`](examples/tkinter_gcs/) — a minimal,
  self-contained mini-GCS that owns its own camera and embeds the SDK.


---

## Project layout

```
src/
├── config/settings.py        # typed (Pydantic) config
├── ingestion/                # Input Layer — frame_buffer, video_reader,
│                             #   rtsp_handler (GStreamer→FFmpeg low-latency capture)
├── detection/                # AI Core — detector abstraction + RF-DETR + YOLOv11
├── tracking/                 # AI Core — ByteTrack + appearance ReID
├── reasoning/                # AI Core — EO/IR fusion + auto registration + spatial
├── vlm/                      # AI Core — Qwen2.5-VL reasoning (async, gated worker)
├── events/                   # Intelligence Layer — event generation
├── intelligence/             # Intelligence Layer — live aggregation + analyzers
├── engine/                   # the VisionAssistant SDK engine (Output Layer)
├── contracts/                # serializable data contracts + VLM gateway
└── utils/                    # types, overlay drawing, logger, profiling
desktop/                      # PySide6 demonstration app (python -m desktop)
run_cli.py                    # headless runner
config/config.yaml            # all runtime configuration
future-ai-mission-analyst/    # staged code for the separate Analyst repo
docs/architecture-review/     # design docs (06 = this refactor)
```

---

## Models

| Stage | Model | Why |
|-------|-------|-----|
| Detection | **RF-DETR** (primary) | Modern transformer detector behind an abstraction; backends swap via one config line. **YOLOv11** stays as the fallback. |
| Tracking | **ByteTrack** (`supervision`) | Strong ID retention through occlusion, near-zero overhead, decoupled from the detector. |
| Reasoning | **Qwen2.5-VL** | Open VLM for grounded scene description, multi-image EO+IR fusion; runs locally in 4-bit on 8 GB. |

**Detection class filter ("class selector").** Choose which COCO classes the
assistant reports via `detection.report_classes` in `config.yaml` (a list of class
names; `null` = all). It is applied *after* detection, so small-object recall is
preserved — only which classes reach the report/overlay is narrowed. The same
filter is exposed everywhere: the demo's searchable class selector edits it,
`run_cli.py --classes person,car,boat` overrides it headlessly, and the SDK reads
it (`VisionAssistant.set_classes(...)`, `ImageAnalyzer/VideoAnalyzer.analyze(classes=...)`).

Everything is driven by `config/config.yaml` (override the path with `EOIR_CONFIG`).
Logs go to `logs/` (Loguru). See
`docs/architecture-review/06-vision-assistant-sdk-refactor.md` for the full refactor
rationale and module-by-module plan.

---

## Notes & limitations
- EO/IR fusion assumes the two streams view a shared scene; the IR→EO mapping is
  estimated online (no calibration), and degrades gracefully to per-frame thermal
  analysis for a lone IR stream.
- RF-DETR's VRAM fit alongside Qwen on an 8 GB RTX 5060 should be confirmed on the
  box; `detection.backend: "yolo11"` is the instant fallback.
- The desktop app is a **demo** of the SDK, deliberately minimal — production hosts
  integrate the engine directly via the Output Layer.
```
