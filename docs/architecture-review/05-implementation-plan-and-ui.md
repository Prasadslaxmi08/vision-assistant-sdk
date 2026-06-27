# Prioritized Implementation Plan & ISR UI Plan — EO/IR Mission Console

**Status:** Planning deliverable. Consolidates the other four documents into a single
sequenced plan. No code changed.
**Date:** 2026-06-23

> This is the **stabilization phase**: architecture review, latency reduction, detector
> modernization (RF-DETR via abstraction), profiling/benchmarking, repo-split prep, and
> ISR-console polish. **No new AI features, agents, memory systems, or reasoning modules.**

---

## Part 1 — Prioritized implementation plan

Ordered by value-to-risk. Each milestone is independently shippable and leaves the app
runnable with tests green. Cross-refs: `01` performance, `02` detector, `03` latency,
`04` repo-split.

### Milestone 0 — Instrumentation first (so everything else is measured)
*Why first:* you cannot rank or prove optimizations without numbers, and the system has
never been run on the box with live feeds.
- Run `streamlit run app.py` on the GPU box with a real/recorded EO (and IR) feed; capture
  baseline behavior.
- Implement the lightweight per-stage timing + queue/frame-age metrics (`01` §6.1–6.2).
- Add the dev-only profiling dashboard at 1 Hz (`01` §6.3), behind a config flag.
- **Exit:** baseline p50/p95 per stage for sparse-EO, dense-EO, EO+IR-fusion scenes.

### Milestone 1 — Quick latency wins (low risk, high felt impact)
From `03` Phase A3:
- One `snapshot()` per refresh; cache `gpu_stats()` at ~1 Hz; remove double frame `.copy()`.
- Split `frame_view()` (cheap) vs `panel_view()` (slow rerun only).
- **Exit:** measurable drop in per-refresh UI cost on the dashboard.

### Milestone 2 — Decouple display from inference (the core fix)
From `03` Phase A1/A2/A4 — the single biggest win:
- Render the **latest raw frame** immediately (capture-thread-owned `latest_raw`, not
  inference-owned).
- AI publishes an `OverlayResult` (coordinates, not pixels) via lock-free swap; render
  composites overlay on the live frame; overlay may lag.
- Raise/decouple the feed rate above 5 Hz toward stream FPS.
- **Exit:** feed FPS independent of detector/fusion cost; AI spikes no longer freeze video
  (verified on dashboard: capture-vs-display FPS lines stay flat under load).

### Milestone 3 — Detector abstraction + RF-DETR
From `02`:
- Introduce `Detector` protocol + `build_detector(config)`; route `console`/`pipeline`/
  `analyzers` through it (YOLO path identical to today).
- Implement `RFDetrDetector`; add `detection.backend` config selector.
- **Exit:** backend switchable via `config.yaml` only; downstream untouched.

### Milestone 4 — Benchmark detectors
From `02` §4:
- Build the benchmark harness; run YOLOv11 vs RF-DETR on EO + IR workloads on the box.
- Confirm RF-DETR variant + Qwen-3B(4-bit) < 8 GB.
- **Exit:** `detector-benchmark-<date>.md` with measured table + recommendation.

### Milestone 5 — Spike-proof the AI path
From `03` Phase B:
- Bound RANSAC (early-exit, cap points, sub-rate); downscale IR before `hot_blobs`;
  sub-rate spatial/threat/fusion relative to detection.
- **Exit:** steadier overlay lag; RANSAC/blob spikes gone from the dashboard p95.

### Milestone 6 — Fusion-availability correctness
From the prompt (fusion only when both sensors present) + `04`:
- Audit/confirm: Live = two independent streams (EO + IR); Video = EO-only / IR-only /
  EO+IR pair; Image = EO-only / IR-only / EO+IR pair. Fusion control enabled **only** when
  both inputs exist (already gated on `fusion_available` — verify across all three modes).
- Keep the fusion architecture sensor-agnostic (no hardcoded sensor assumptions; sensor
  profiles already in config). **Exit:** all input combinations behave per spec.

### Milestone 7 — ISR UI cleanup
See Part 2. **Exit:** operator view free of AI/tech naming; right-panel toolset complete.

### Milestone 8 — Repo-split prep (in place)
From `04` Phases 0–1 (safe in-monorepo refactors):
- Move `Event` (+ sibling records) to `utils/types`; introduce shared contracts package;
  split config; wrap DB writes behind `StoreWriter` and VLM behind `VlmGateway`.
- **Exit:** boundary defined; physical split becomes mechanical (defer actual split).

### Milestone 9 — Physical repository split (optional, last)
From `04` Phases 2–4. Only after latency + detector work lands and the boundary is proven.

**Recommended sequencing rationale:** the prompt's #1 priority is a smooth live feed, so
Milestones 0–2 come first and deliver the biggest operator-visible improvement. Detector
modernization (3–4) and spike-proofing (5) follow. UI/repo work (6–9) is lower-risk polish
and structural prep that doesn't block the latency goal.

---

## Part 2 — ISR operator-console UI plan

**Goal:** a professional surveillance / ISR / mission-ops console, not an ML demo.

### Current state (already aligned — verified in `ui/console_view.py` & `app.py`)
- ✅ Center = EO primary + IR **picture-in-picture**, with **Swap EO/IR**.
- ✅ **Fusion** toggle beside the PiP, disabled unless `fusion_available` (both feeds live).
- ✅ Left sidebar = **System**, **Sensor Streams**, **Operational Settings** only.
- ✅ Right side = Mission Intelligence summary + Timeline, Target Activity, Query, Reporting.
- ✅ AI-stack caption ("YOLOv11·ByteTrack·Qwen") already removed; title is "EO/IR Mission Console".

### Gaps to close (cleanup, not new features) — RESOLVED in M7 (2026-06-24)
1. **Complete the right-panel toolset** to match the spec list:
   - Present: Timeline, Target Activity, Intelligence Summary, Query Tools, Report Generation.
   - ✅ **Thermal Events** panel wired into `console_view.panels()` (`thermal_tab.render(mid)`).
   - ✅ **Object Explorer vs Target Activity — DECIDED: one panel.** "Target Activity" *is* the
     consolidated target/object explorer (per-identity lifecycle via `objects_tab`); there is
     no separate live-vs-lifecycle split. Documented in `console_view.panels()`.
2. ✅ **Language audit done.** Operator-facing strings swept of model/framework terms (Qwen2.5-VL,
   VLM, "7B model", "reasoning") → ISR wording ("intelligence", "narrative phrasing", "analysing",
   "assessment"). Code/docstrings/`model_manager` keep technical names. Dead `video_tab`/`live_tab`
   left untouched (not in the operator surface).
3. ⏳ **`use_container_width` deprecation — DEFERRED (per "not urgent").** Streamlit 1.58 still
   accepts it (warns only). A mechanical sweep to `width="stretch"/"content"` across
   `st.image/button/download_button/dataframe` is the remaining item; deferred because it can't be
   visually verified without the GPU box + a running server, and is consistency-only.
4. ✅ **Dead code removed.** `mission_picker` (+ its private `_wallclock` helper and the unused
   `get_memory_store` import) deleted from `ui/common.py`. `thermal_tab` is now wired (per #1).
5. ✅ **Layout polish.** Added `.streamlit/config.toml` dark ISR theme; an operations header of
   status chips (EO/IR live + fps, target count, threat band, fusion-lock state) renders above
   the imagery in `_feed()` (replaces the old 4-metric row); center stays dedicated to imagery.

### Non-goals for the UI
- No new analytics, no new AI panels, no new reasoning surfaces. Only restructure, relabel,
  complete the specified toolset, and clean dead code.

---

## Part 3 — Final deliverables checklist (this phase)

| Deliverable | Document | Status |
|---|---|---|
| Performance report (+ profiling/dashboard plan) | `01-performance-latency-report.md` | ✅ written |
| Detector benchmark report + framework design | `02-detector-benchmark-report.md` | ✅ written |
| Latency optimization roadmap (decoupled rendering) | `03-latency-optimization-roadmap.md` | ✅ written |
| Repository split plan | `04-repository-split-plan.md` | ✅ written |
| Prioritized implementation plan + UI plan | `05-implementation-plan-and-ui.md` (this) | ✅ written |

**Measured artifacts still owed (require the GPU box + feeds):** baseline per-stage
profiling numbers, the filled detector benchmark table, and before/after latency proof per
milestone. These are produced when the plan is executed on the target machine.

---

## Part 4 — Decisions needed before execution
1. Start implementing after these plans, or revise the plans first?
2. Repo-split: shared `eoir-store` package vs duplicated thin writer? (recommend shared)
3. Live VLM overlays cross-process (needs the `04` §4.2 stream) or offline reporting OK?
4. RF-DETR variant to target first (base vs nano) given 8 GB + Qwen-3B?
5. Is a labeled EO/IR set available for real mAP, or report relative metrics only?
6. Object Explorer vs Target Activity — one panel or two? **RESOLVED (M7): one panel** —
   "Target Activity" is the consolidated explorer.
