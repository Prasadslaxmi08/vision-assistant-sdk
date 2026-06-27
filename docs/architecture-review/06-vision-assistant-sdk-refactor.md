# Vision Assistant SDK — Refactor Review & Plan

**Status:** Review + **executed** (2026-06-25). Phases 0–5 below were applied in
this repo; Phase 6 (live GPU verification with RF-DETR + Qwen + an RTSP pair) is a
box task. `rfdetr` and `PySide6` are listed in requirements but not yet installed in
`.venv`, so the detector build and the Qt GUI were not exercised here — all other
modules import/construct cleanly and the engine SDK surface is smoke-tested.
**Date:** 2026-06-25
**Supersedes:** the "EO/IR Mission Console" identity in docs `04`/`05`. This document
re-draws the repository boundary per the new direction.

---

## 1. The pivot

This repo has grown into a full **surveillance application** (operator console,
persistent mission memory, threat scoring, query assistant, report generation,
WebRTC feed, Streamlit web UI). The new objective is narrower and cleaner:

> A reusable **Vision Assistant SDK / Engine** — the *AI layer only* — that any
> existing camera/surveillance software can plug into. Camera vendors already
> provide RTSP, PTZ, zoom, recording, playback, maps and UI. We provide the CV +
> multimodal AI: detection, tracking, EO/IR fusion, and vision-language reasoning,
> exposed through JSON / callbacks / APIs / overlay rendering. A small **PySide6**
> desktop app *demonstrates* the SDK — it is not the product.

Everything that is "mission management" (memory, archives, threat, query, reports,
RAG, agentic) leaves this repo for a future **AI Mission Analyst** repository.

### Target architecture

```
                       VISION ASSISTANT SDK
┌───────────── Input Layer ─────────────┐
│  Image  │  Video (EO / IR / pair)  │  RTSP (EO + IR, independent)  │
└───────────────────┬───────────────────────────────────────────────┘
                    ▼
┌───────────── AI Core ─────────────────────────────────────────────┐
│  Detector (RF-DETR primary, abstraction)  →  ByteTrack            │
│  EO/IR Fusion (gated: both sensors only)  →  Vision-Language VLM  │
└───────────────────┬───────────────────────────────────────────────┘
                    ▼
┌───────────── Intelligence Layer ──────────────────────────────────┐
│  Event generation · scene understanding · scene/mission summaries │
└───────────────────┬───────────────────────────────────────────────┘
                    ▼
┌───────────── Output Layer ────────────────────────────────────────┐
│  JSON   ·   callbacks   ·   API   ·   overlay rendering (data)     │
└───────────────────────────────────────────────────────────────────┘
```

**Real-time contract (highest priority):** the live feed is **never** blocked by
AI inference. The render path shows the latest raw frame immediately; detection /
tracking / fusion / VLM run asynchronously and publish *overlay data* (coordinates,
not pixels) that the renderer composites — overlays may lag, the video never does.
**This is already implemented** in `src/pipeline/console.py` (the `OverlayResult` /
`FrameView` decoupling, latency-roadmap "M2"). We preserve it.

---

## 2. What already exists in our favour

The codebase had a "repo-split prep" pass, so much of the seam work is done:

- **Detector abstraction** (`src/detection/base.py` + `build_detector`) — RF-DETR is
  already a config-selectable backend honouring the same contract. Switching the
  primary detector is a one-line config change, exactly as the prompt wants.
- **Decoupled display** (`console.py`) — the live-feed-never-blocks design is built.
- **Dual independent EO + IR streams** with **fusion gated on `fusion_available`**
  (both sensors present) — `console.py` already does this and is **sensor-agnostic**
  (online RANSAC registration, no calibration; sensor profiles optional).
- **Async, gated VLM worker** (`vlm/vlm_worker.py`) behind a `VlmGateway` contract.
- **Shared data contracts** (`src/utils/types.py` re-exported by `src/contracts/`).

The refactor is therefore **mostly subtraction + repackaging**, not a rewrite of the
CV/AI core.

---

## 3. Module-by-module classification

### ✅ PRESERVE (the SDK core — keep, minor edits only)

| Module | Role | Notes |
|---|---|---|
| `src/ingestion/frame_buffer.py` | drop-oldest bounded buffer | latency backbone |
| `src/ingestion/video_reader.py` | video + threaded file source | image/video input |
| `src/detection/base.py`, `detector.py`, `rf_detr_detector.py` | detector abstraction + YOLO + RF-DETR | **RF-DETR becomes primary**; YOLO stays as a swappable fallback |
| `src/tracking/tracker.py`, `reid.py` | ByteTrack + stable-id ReID | AI Core |
| `src/reasoning/fusion.py`, `registration.py` | EO/IR cross-sensor fusion | core capability, sensor-agnostic |
| `src/vlm/qwen_vlm.py`, `vlm_worker.py`, `prompts.py` | Vision-Language reasoning | AI Core (stays — VLM is *our* layer now) |
| `src/utils/types.py`, `image_utils.py`, `visualization.py`, `logger.py`, `system_stats.py`, `profiling.py` | shared infra + overlay compositing | Output Layer uses `compose_overlay` |
| `src/contracts/vlm.py` (`VlmGateway`) | VLM boundary | keep |

### 🔧 REWRITE / REFACTOR

| Module | Change |
|---|---|
| `src/pipeline/console.py` | **Keystone.** Becomes the SDK **Engine** (`VisionAssistant`). Strip `MissionStore` (self.store + all `record_*`/`update_*` calls), strip `ThreatScorer`, strip report export. Keep streams, detection, tracking, fusion, events, VLM, decoupled render. Rename "mission" → engine/scene vocabulary. |
| `src/intelligence/mission_intelligence.py` | Slim to **live aggregation only** (intel feed, timeline, stats, scene summaries). Remove `to_excel_bytes` / `to_pdf_bytes` / `export` (those move out with `exporters.py`). |
| `src/intelligence/analyzers.py` | Rewrite `ImageAnalyzer` / `VideoAnalyzer` on top of the SDK engine; drop `MissionStore` + `ThreatScorer` usage. Image/video input modes route here. |
| `src/events/event_manager.py` | Keep as **event generation** (Intelligence Layer). Drop the threat-only `THREAT_ESCALATION` coupling. |
| `src/config/settings.py` + `config/config.yaml` | Drop `memory`, `query`, `threat`, `webrtc`, most of `mission`/`ui` (Streamlit) config. Make `detection.backend` default `rf_detr`. |
| `src/pipeline/pipeline.py` | Single-stream orchestrator — fold its unique bits into the engine or keep a thin single-stream path; strip store/threat. |
| `src/ingestion/rtsp_handler.py` | **Adopt the RTSP.py streaming layer** — GStreamer-first (`drop-on-latency`, `max-buffers=1`, `latency=0`) with FFmpeg fallback using the low-latency flags (`fflags;nobuffer|flags;low_delay|max_delay`), plus its reconnect/backoff. Keep our clean `FrameBuffer` delivery; discard RTSP.py's Tkinter UI/PTZ/recording. |
| `run_cli.py` | Repoint to the SDK engine; drop `--report`/threat. |

### ❌ REMOVE (web UI + browser-feed infra — not part of an SDK)

- `app.py` (Streamlit entry)
- `ui/` — **all** Streamlit tabs (replaced by the PySide6 demo)
- `static/feed.html`, `.streamlit/`, `mediamtx.yml`, `docs/WEBRTC-FEED.md`, `src/pipeline`
  WebRTC payload path, `webrtc` config — MediaMTX/WebRTC browser feed
- `streamlit`, `reportlab`, `openpyxl` from core `requirements.txt`
- `RTSP.py` (root) — once its streaming layer is lifted into ingestion

### 📦 MOVE → future **AI Mission Analyst** (staged in-repo, not deleted)

Staged under `future-ai-mission-analyst/` (this is **not** a git repo, so nothing is
hard-deleted — the code is preserved verbatim for the new repo).

| Module | Why it leaves |
|---|---|
| `src/memory/` (`mission_store.py`, `schema.py`) | SQLite persistent mission memory / archives |
| `src/query/` (`assistant.py`, `intents.py`, `prompts.py`) | NL query assistant |
| `src/reasoning/threat.py` | threat / anomaly scoring |
| `src/intelligence/mission_report.py`, `exporters.py` | report generation (PDF/Excel) |
| `src/contracts/store.py` (`StoreWriter`/`StoreReader`) | DB seam |
| `ui/timeline_tab.py`, `objects_tab.py`, `thermal_tab.py`, `query_tab.py`, `reports_tab.py` | memory-backed surveillance panels |
| `outputs/*.db`, `outputs/*.xlsx`, `outputs/*.pdf` | mission archives |

`src/reasoning/spatial.py` (object-to-object proximity/following/group) is **borderline**:
it is scene-understanding, but its main consumer was threat. **Decision: keep it** in the
Intelligence Layer as optional scene understanding (it has no threat dependency itself).

### ➕ ADD (new)

- **SDK surface** — a clean engine package (`src/sdk/` or `vision_assistant/`) exposing
  `VisionAssistant`: configure input mode (image/video/live), EO/IR, fusion (auto-gated),
  register callbacks, and read structured results (JSON-serialisable) + overlay frames.
- **PySide6 desktop demo** — `desktop/` lightweight Qt app: source picker (image / video /
  live EO+IR RTSP), live feed widget (renders raw frame + overlay), detections list, scene
  summary panel, fusion lock indicator. Demonstration only.
- `requirements.txt`: add `PySide6`; `requirements-rfdetr.txt` stays the RF-DETR extra.

---

## 4. CV→AI coupling that must be cut (precise seams)

From `console.py` / `pipeline.py` / `analyzers.py`, the lines that reach into
to-be-moved modules:

1. `from src.memory import MissionStore` + `self.store = MissionStore(...)` and every
   `self.store.record_frame/record_events/update_threat/record_interaction/record_thermal/
   update_object_media/open_mission/close_mission/queue_stats` call.
2. `from src.reasoning import ... ThreatScorer` + `self.threat = ThreatScorer(...)` and
   every `self.threat.assess(...)` / `self.threat.mission_threat()` call.
3. `from src.contracts import StoreWriter` (the store seam typing).
4. `mission.export(...)` / `to_pdf_bytes` / `to_excel_bytes` and the `exporters` imports.
5. `src/reasoning/__init__.py` re-exports `ThreatScorer` → drop that export.

All of these are unidirectional (CV/AI never *needs* them to run) — removing them does
not break detection→tracking→fusion→VLM. Threat was only a *consumer* of events/fusion;
the producers stay.

---

## 5. Incremental sequence (keeps the pipeline runnable at every step)

**Phase 0 — Plan + safe prep (non-breaking)**
- This document. Add `PySide6` to requirements. Stage the move targets.

**Phase 1 — Carve the SDK engine (additive first)**
- New `VisionAssistant` engine = `console.py` minus store/threat/reports. Build it
  alongside the old code so nothing breaks; unit-smoke it headless.

**Phase 2 — Ingestion swap**
- Fold the RTSP.py GStreamer/FFmpeg low-latency capture into the ingestion layer behind
  the existing `RTSPStreamHandler` interface (no downstream change).

**Phase 3 — Detector primary = RF-DETR**
- Flip `detection.backend` default; document `pip install -r requirements-rfdetr.txt`;
  YOLO remains the fallback backend.

**Phase 4 — PySide6 demo**
- Three input modes over the SDK; remove Streamlit (`app.py`, `ui/`, `.streamlit/`,
  WebRTC infra) once the Qt demo covers live/video/image.

**Phase 5 — Move the analyst modules + slim config/requirements**
- Relocate memory/query/threat/reporting to `future-ai-mission-analyst/`; trim
  `config.yaml`, `settings.py`, `requirements.txt`; delete the moved UI tabs.

**Phase 6 — Verify** (needs the GPU box: RF-DETR weights, Qwen, a live RTSP pair).

---

## 6. Constraints / things that can't be verified in this session

- **8 GB RTX 5060:** RF-DETR variant VRAM alongside Qwen-3B(4-bit) must be confirmed on
  the box (per `docs/02`). We keep YOLO as the instant fallback.
- `PySide6` and `rfdetr` are **not yet installed** in `.venv`; GUI + RF-DETR detection
  can't be exercised here. Code is written to be correct; live verification is a box task.
- This is **not a git repo** → moved modules are *relocated*, never hard-deleted.
</content>
</invoke>
