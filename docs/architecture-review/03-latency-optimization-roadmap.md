# Latency Optimization Roadmap — EO/IR Mission Console

**Status:** Plan deliverable (design only; no code changed).
**Date:** 2026-06-23
**Prereq reading:** `01-performance-latency-report.md`.

> **Prime directive (from the prompt):** the live EO and IR feeds must *always* stay
> responsive. The operator must always see the most current frame, even if the overlay
> lags. **AI work must never block or slow the live stream.** Every item here serves that.

---

## 1. Target architecture — decouple rendering from inference

Today: one thread does capture→infer→reason→draw and the UI shows the *fully processed*
frame. Target: **three independent rates** with the display owning the fastest one.

```
                ┌─────────────────────────────────────────────────────────┐
   CAPTURE      │  RTSP/Video → FrameBuffer (drop-oldest)   [per stream]   │  ~stream FPS
   (exists)     └───────────────┬──────────────────────────┬──────────────┘
                                 │ newest raw frame         │ newest raw frame
                 ┌───────────────▼───────────┐  ┌───────────▼───────────────┐
   DISPLAY       │  Render loop: take latest  │  │  AI INFERENCE (background) │  AI FPS
   (new,fast)    │  RAW frame, draw cached    │  │  detect→track→reid→reason  │  (slower, OK)
   ~stream FPS   │  overlay on top, show NOW  │  │  →fusion→threat→DB→VLM     │
                 └───────────────▲───────────┘  └───────────┬───────────────┘
                                 │ overlay (boxes, markers, threat) — may be 1–N frames old
                                 └────────── latest AI result (lock-free swap) ◄┘
```

**Contract:**
- The render path **never** calls a model, RANSAC, blob extraction, or the DB.
- The render path reads (a) the newest raw frame from the buffer and (b) the most recent
  AI **overlay result** (boxes/markers/threat), and composites them. If AI is behind, the
  operator still sees live video with a slightly stale overlay — exactly as specified.
- AI publishes results via an **atomic/lock-free swap** of an immutable result object
  (no full-frame copy on the AI→render handoff; pass coordinates, not pixels).

---

## 2. Phased roadmap (ordered: most impact / least risk first)

### Phase A — Decouple display from AI (the core fix)
Highest impact (report rank #1–3). No new features; pure restructuring.

1. **Render the raw frame immediately.**
   - Add a render path that pulls `buffer.get_latest()` (or a dedicated `latest_raw`
     updated by the *capture* thread, not the inference thread) and displays it.
   - Stop displaying `latest_annotated` directly.
2. **Publish AI overlays as data, not pixels.**
   - AI thread produces an `OverlayResult{boxes, labels, track_ids, markers, threat,
     reg_banner}` and swaps it into a single slot (lock-free reference assignment).
   - Render composites overlay over the live raw frame with `draw_detections`-style code
     on the *render* thread (cheap; or precompute a transparent overlay layer).
3. **Kill redundant `snapshot()` work.**
   - One `snapshot()` per refresh (not two). Split into `frame_view()` (cheap, render) and
     `panel_view()` (intel/timeline/threat, called only on full rerun).
   - **Cache `gpu_stats()` at ~1 Hz** — never call NVML on the feed loop.
   - Remove the double `.copy()`; the render path copies at most once (or draws onto a
     reused buffer).
4. **Raise / decouple the UI rate.**
   - Once render is cheap, raise the feed fragment above 5 Hz (e.g. 15–30 Hz target), or
     better, drive display at the stream's measured FPS. Keep panels on a slow rerun.

*Expected outcome:* live feed smoothness becomes independent of detector/fusion cost.
Freezes from AI spikes disappear (feed keeps moving; overlay just lags briefly).

### Phase B — Make AI inference cheaper & spike-resistant
Reduces overlay lag and CPU spikes (report ranks #5–8). Overlay still may lag — that's fine.

5. **Bound RANSAC registration.** Early-exit on confidence; cap control points; run
   registration at a lower cadence than detection (it changes slowly once locked — the
   `max_warm_sec`/lock FSM already supports reuse). Move it off the per-frame loop to a
   sub-rate task.
6. **Downscale IR before `hot_blobs`.** Run `np.percentile` + `connectedComponentsWithStats`
   on a downscaled IR image; scale blob coords back up. Cap `max_blobs` (already configurable).
7. **Sub-rate the heavy reasoning.** Spatial clustering, threat re-scoring, and fusion do
   not need to run every frame for a smooth feed. Run detection/track every frame; run
   spatial/threat/fusion every k frames or on a separate cadence. (No feature removed —
   same outputs, lower frequency.)
8. **ReID cadence.** Keep appearance refresh at its configured stride; skip ReID embed for
   crops < `min_crop_px` (already supported) and when det count is high.

### Phase C — Parallelism (only if Phase A/B insufficient, and within 8 GB)
9. **Separate EO and IR inference cadence** so a busy IR frame can't stall EO. Options:
   keep one GPU thread but **prioritize EO** (process EO every cycle, IR opportunistically);
   or a second lightweight thread *only* if VRAM headroom confirmed by profiling. EO
   responsiveness is the priority, so prioritization beats true parallelism here.
10. **Pin/pipeline GPU work.** Use `torch.cuda.Event` timing; consider CUDA stream overlap
    of decode/preprocess with inference if measured GPU gaps justify it.

### Phase D — Render/transport polish
11. Reduce `st.image` cost: send appropriately-sized frames (respect `ui.max_display_width`),
    avoid re-encoding unchanged regions; consider a lighter live-video component if Streamlit
    image refresh proves to be the ceiling after A–C.
12. Verify RTSP path under real network jitter (UDP vs TCP trade-off via `rtsp_transport`),
    confirm reconnect/watchdog behavior eliminates the "occasional feed freeze."

---

## 3. Expected gains (to be confirmed by the profiling pass)

| Phase | What changes | Expected effect | Confidence |
|---|---|---|---|
| A | Display reads raw frame + data overlay; 1 cheap snapshot; NVML cached; higher UI rate | Live feed smooth & spike-immune; biggest single win | **High** (structural) |
| B | Bounded RANSAC, downscaled IR blobs, sub-rate reasoning | Lower & steadier overlay lag; fewer CPU spikes | Medium (depends on scene) |
| C | EO-priority scheduling | EO stays live under IR/crowd load | Medium |
| D | Render/transport tuning | Removes residual ceiling & network freezes | Medium |

No numeric promises until Phase A of the profiling plan (`01` §6) yields measured p50/p95.

---

## 4. Guardrails / non-goals

- **No feature changes.** Same detections, events, fusion, threat, reports — only *when*
  and *on which thread* they run changes.
- **Preserve correctness of the archive.** Sub-rating reasoning must not corrupt mission
  memory; keep per-frame detection recording (already sampled via `detection_sample_stride`).
- **Keep 8 GB safe.** No second model loaded on the real-time path; RF-DETR remains
  swappable (see `02`).
- **Measure before/after.** Each phase lands with a before/after on the §6 dashboard so
  gains are evidenced, not assumed.

---

## 5. Suggested order of execution
A3 (kill redundant snapshot/NVML — quick win) → A1/A2 (raw fast-path + data overlay) →
A4 (raise rate) → instrument (`01` §6) → B5/B6 (RANSAC + IR downscale) → B7 (sub-rate) →
re-measure → C/D only if needed.

Sequenced against the detector and repo-split work in `05-implementation-plan-and-ui.md`.
