# Performance & Latency Report — EO/IR Mission Console

**Status:** Review deliverable. Static analysis from source + an instrumentation plan.
**Date:** 2026-06-23
**Scope:** Real-time path only (live operator console). Offline analyzers excluded.

> **Important caveat — no measured numbers yet.** This system has *not* been run on
> the GPU box with two live feeds (per project notes). The latencies below are
> derived from reading the code (algorithmic cost, threading model, copy/lock points),
> not from a profiler. Section 6 specifies the instrumentation needed to replace these
> estimates with real measurements. Treat ranked findings as **where to measure first**,
> not as final attribution.

---

## 1. Executive summary

The reported symptoms — delay, stutter, buffering, lagging overlays, occasional feed
freezes — are explained by **one architectural root cause** plus several amplifiers:

**Root cause:** *the live video display is coupled to AI inference.* The UI only ever
renders `latest_annotated`, and that field is written **at the end** of a single,
per-frame, single-threaded chain:

```
detect → track → ReID → draw → (EO) events → spatial → threat
        → 4× DB enqueue → VLM submit → [fusion: hot_blobs → RANSAC → fuse → overlay]
```

Until that entire chain finishes for a frame, the operator sees nothing newer — even
the *raw* frame is withheld (`latest_raw` is set only inside `_process_stream`, after
detection). So display FPS = pipeline FPS, and any spike (RANSAC, blob extraction,
clustering, a slow YOLO frame) becomes a visible freeze.

**Amplifiers:**
1. UI fragment capped at **5 Hz** (`@st.fragment(run_every=0.2)`).
2. `snapshot()` called **twice per refresh** (in `_feed()` and `panels()`), each doing
   **two full-frame `.copy()`** under lock **+ an NVML `gpu_stats()` query**.
3. **One inference thread round-robins EO + IR** through a shared YOLO — a slow IR frame
   delays EO and vice-versa.
4. **Inline fusion**: RANSAC registration (worst-case O(n⁴), capped ~5000 iters),
   `cv2.connectedComponentsWithStats` over the full IR frame every fusion cycle, and an
   overlay redraw — all on the same loop.

The good news: the ingestion layer is already correct (drop-oldest `FrameBuffer`,
`CAP_PROP_BUFFERSIZE=1`, real-FPS measurement) and **DB writes are already async**
(non-blocking queue, dropped on overflow). The fix is decoupling render from inference,
not rewriting ingestion. See `03-latency-optimization-roadmap.md`.

---

## 2. The pipeline, stage by stage

Per-frame stages on the hot path, with cost class and where they run. "Cost" is
algorithmic/structural, pending measurement.

| # | Stage | File | Thread | Cost | On display path? |
|---|---|---|---|---|---|
| 1 | RTSP/video capture + decode | `ingestion/rtsp_handler.py`, `video_reader.py` | capture (per stream) | GPU/CPU decode | produces frames |
| 2 | Frame buffering (drop-oldest) | `ingestion/frame_buffer.py` | capture→infer handoff | O(1) | no (correct) |
| 3 | YOLOv11 detect | `detection/detector.py:48` `model.predict` | inference (shared) | **HEAVY (GPU)** | **yes** |
| 4 | ByteTrack update + merge | `tracking/tracker.py` | inference | light (O(n²) IoU, small n) | yes |
| 5 | ReID embed + match | `tracking/reid.py:188` | inference | **MEDIUM** (calcHist×2–4, Sobel/crop) | yes |
| 6 | draw_detections + draw_hud | `utils/visualization.py:36,72` | inference | MEDIUM (**2× full-frame copy**) | yes |
| 7 | Events | `events/event_manager.py` | inference (EO) | light | yes (EO) |
| 8 | Spatial reasoning | `reasoning/spatial.py` | inference (EO) | MEDIUM (O(n²)+union-find) | yes (EO) |
| 9 | Threat scoring | `reasoning/threat.py` | inference (EO) | light | yes (EO) |
| 10 | DB writes (frame/events/threat/interaction) | `memory/mission_store.py` | inference enqueue → writer thread | **async (non-blocking)** | no |
| 11 | VLM submit | `vlm/vlm_worker.py` | inference enqueue → VLM thread | async; **+1 frame copy** on submit | no |
| 12 | Fusion: hot_blobs | `reasoning/fusion.py:217,222` | inference | **MEDIUM–HIGH** (percentile + connected-components, full IR) | yes (when fusion on) |
| 13 | Fusion: registration RANSAC | `reasoning/registration.py:185` | inference | **HIGH spike** (O(n⁴) capped) | yes (when fusion on) |
| 14 | Fusion: fuse_pair + overlay draw | `reasoning/fusion.py`, `console.py:363` | inference | MEDIUM | yes (when fusion on) |
| 15 | snapshot() for UI | `console.py:401` | UI thread (5 Hz) | **2× full-frame copy + NVML**, under lock | **display read** |
| 16 | Streamlit render (st.image) | `ui/console_view.py:177` | UI | encode + rerun | **display** |

Stages 3–14 are **serialized on one thread**; the display (15–16) reads the result of
14. That is the coupling to break.

---

## 3. Ranked bottleneck report (pre-measurement, by expected impact)

Ordered by expected contribution to perceived latency/stutter. "Expected gain" is the
qualitative improvement from fixing each, to be confirmed by profiling.

| Rank | Bottleneck | Where | Why it hurts | Expected gain when fixed |
|---|---|---|---|---|
| **1** | **Display coupled to inference** (no raw fast-path) | `console.py:248–280`, UI reads `latest_annotated` | Display FPS = slowest-stage FPS; every spike = freeze | **Largest.** Live feed becomes smooth & decoupled from AI entirely |
| **2** | **5 Hz UI cap** | `console_view.py:159` `run_every=0.2` | Hard 5 fps ceiling on the feed | High — smoothness; raise once render is cheap |
| **3** | **Double `snapshot()` + 2× frame copy + NVML per refresh** | `console_view.py:164,205`; `console.py:403–407,438` | Per refresh: 2 snapshots × (2 copies + NVML lock) on the UI thread | High — removes redundant work from the render loop |
| **4** | **EO+IR round-robin on one inference thread** | `console.py:218–242` | A slow IR frame stalls EO; serialized GPU use | Medium–High — EO stays responsive under IR load |
| **5** | **RANSAC registration spike** | `registration.py:185–206` | O(n⁴) capped; can burn 10–50 ms on busy frames; runs inline | Medium — removes worst stutter spikes when fusion on |
| **6** | **IR connected-components every fusion cycle** | `fusion.py:217,222` | Full-IR `np.percentile` + `connectedComponentsWithStats`, no downscale | Medium — cheaper, steadier fusion |
| **7** | **Spatial clustering O(n²)+union-find every frame** | `spatial.py:230–249` | Scales with crowd size; inline on EO | Low–Medium — matters on dense scenes |
| **8** | **ReID histograms/Sobel per refresh** | `reid.py:188–226` | ~20% of frames recompute appearance; scales with det count | Low–Medium |
| **9** | **Per-frame double `.copy()` in drawing** | `visualization.py:36,72` | 2 full-frame copies/frame for annotation | Low (but free once decoupled) |

**Note on attribution uncertainty:** ranks 1–4 are structural and confident. Ranks 5–9
are algorithmic costs whose *real* weight depends on scene density and image size — these
are exactly what the profiling pass must quantify before optimizing.

---

## 4. What is already correct (don't "fix" these)

- **Drop-oldest buffering** — `FrameBuffer.get_latest()` discards stale frames; the
  buffer keeps the pipeline on near-live imagery (`config: drop_policy: newest_keeps`).
- **RTSP buffer minimization** — `CAP_PROP_BUFFERSIZE=1`, TCP transport, read-timeout
  watchdog, auto-reconnect with backoff, EMA-measured real FPS.
- **Async DB writes** — `MissionStore` uses a writer thread + bounded queue
  (`queue_maxsize: 10000`, `put_nowait`, shed on overflow). The inference thread never
  blocks on disk.
- **VLM fully async + rate-limited** — `VLMWorker` thread, `min_seconds_between_calls`,
  `max_pending_requests: 2` (drops oldest). Inference never waits on Qwen.
- **Single shared YOLO for 8 GB** — deliberately avoids a second model / concurrent CUDA.

---

## 5. Memory / VRAM notes (8 GB budget)

- YOLOv11m + Qwen2.5-VL-3B (4-bit) ≈ 4.8 GB per project notes — headroom exists but is
  not large. Any RF-DETR work must be *swappable*, not concurrent (see detector report).
- `gpu_stats()` (`system_stats.py`) reads `torch.cuda.mem_get_info()` + NVML; it's the
  right telemetry source. Move it **off** the per-refresh path (cache ~1 Hz) — see rank 3.
- Frame copies (`snapshot`, drawing, VLM submit) are host RAM, not VRAM, but they add up
  on the UI thread; the roadmap removes the redundant ones.

---

## 6. Profiling instrumentation plan (to get real numbers)

The current telemetry shows only stream FPS, processing FPS, and GPU/VRAM. To produce a
*measured* bottleneck report we need per-stage timing and queue health. Proposed, minimal,
and low-overhead (the prompt explicitly wants this dashboard — it is instrumentation, not
a new product feature):

### 6.1 Metrics to capture
- **Per-stage latency** (ms): decode, detect, track, reid, events, spatial, threat,
  fusion.hot_blobs, fusion.ransac, fusion.fuse, draw, snapshot-copy, st.image render.
- **Rates:** capture FPS, decode FPS, inference FPS, *display* FPS (separately!), VLM calls/min.
- **Queues:** `FrameBuffer.size`/`dropped`/`received` per stream; VLM `dropped`/`total_calls`;
  DB write-queue depth.
- **Frame age:** wall-clock between capture timestamp and (a) inference, (b) display.
- **Resources:** CPU%, GPU%, VRAM used/total (already have), process RSS.
- **Registration:** RANSAC iterations actually run, inliers, lock state, time per `update()`.

### 6.2 How to instrument (lightweight)
- A `StageTimer` context manager writing to a ring buffer of EMA + p50/p95 per stage
  (no per-frame logging spam). ~5 lines around each stage in `_process_stream`,
  `_eo_reasoning`, `_fuse`.
- Add a wall-clock `capture_ts` to `Frame` (already has `timestamp`/`source_ts`) and stamp
  it at `read()`; compute frame-age at display in `snapshot()`.
- Expose a `Console.metrics()` returning the aggregates; the writer thread already exists
  for DB — reuse the same non-blocking pattern so timing never blocks inference.

### 6.3 Profiling dashboard (operator-hidden, dev view)
- A separate Streamlit page / sidebar expander (dev-only, behind a config flag) showing:
  per-stage p50/p95 bar chart, capture-vs-display FPS lines, queue depths, frame-age
  histogram, RANSAC time, VRAM. Refreshes at 1 Hz (NOT on the feed fragment).
- For deeper one-off profiling: a `--profile` CLI run using `cProfile`/`yappi` over
  `run_cli.py`, plus `torch.cuda.Event` timing around `model.predict` for true GPU time,
  and `nvidia-smi dmon` / Nsight Systems for a system-level trace.

### 6.4 Acceptance — what "measured" looks like
A bottleneck report table identical to §3 but with **real p50/p95 ms per stage** on the
target box, for three scenes: (a) sparse EO-only, (b) dense EO (crowd), (c) EO+IR fusion
locked. That data confirms or re-orders §3 before any optimization is committed.

---

## 7. Hand-off

- **Optimizations** that follow from this report → `03-latency-optimization-roadmap.md`.
- **Detector cost** (YOLO vs RF-DETR) → `02-detector-benchmark-report.md`.
- **Sequencing** of all of the above → `05-implementation-plan-and-ui.md`.
