# Repository Split Plan — EO/IR Mission Console ↔ AI Mission Analyst

**Status:** Planning deliverable (no code moved yet). Part of the architecture-stabilization phase.
**Date:** 2026-06-23

---

## 1. Goal

Separate today's single codebase (~7,800 LOC mixing computer vision, EO/IR
processing, persistence, threat analysis, query, reporting, and VLM reasoning)
into **two repositories** with a clean, one-directional contract:

| | Repository A | Repository B |
|---|---|---|
| **Name** | **EO/IR Mission Console** | **AI Mission Analyst** |
| **Focus** | Real-time CV: capture, detection, tracking, ReID, EO/IR registration + fusion, event generation, operator visualization | Agentic AI: memory, RAG, query assistant, report generation, mission reasoning, higher-level intelligence |
| **Runtime** | Hard real-time, GPU detector, operator console | Batch / on-demand, GPU VLM, headless services |
| **Consumes** | Sensor feeds | Mission outputs from Repo A |

The driving principle from the review: **CV → AI coupling is already strictly
unidirectional.** No CV module imports any AI module. The split is therefore
mostly *mechanical*, with three named seams to repair first.

---

## 2. Current coupling map (measured, not assumed)

Verified by reading every module's imports.

### CV layer never imports AI — confirmed clean
`detection/`, `tracking/`, `reasoning/`, `events/`, `ingestion/` import **nothing**
from `memory/`, `query/`, `intelligence/`, or `vlm/`.

### AI layer imports CV in exactly three places

| AI module | Imports from CV | Severity |
|---|---|---|
| `intelligence/mission_intelligence.py` | `Event` from `events.event_manager` (1 line) | **Trivial** — data class only |
| `vlm/vlm_worker.py` | `Event` from `events.event_manager` (1 line) | **Trivial** — data class only |
| `intelligence/analyzers.py` | `YOLODetector`, `EventManager`, `ObjectTracker`, `Reidentifier`, `FusionEngine`, `SpatialReasoner`, `ThreatScorer`, `VideoReader` (8 imports) | **Hard** — instantiates the whole CV pipeline |

### The integration point (belongs to neither, or to a thin third layer)
`pipeline/console.py` and `pipeline/pipeline.py` import from **both** layers — they
are orchestrators, not split candidates. See §6.

---

## 3. Module assignment

### Repository A — EO/IR Mission Console (stays / is the "host")

```
src/config/            settings.py            (split: see §3.3)
src/ingestion/         frame_buffer, rtsp_handler, video_reader
src/detection/         detector.py            (+ detector abstraction — see detector plan)
src/tracking/          tracker.py, reid.py
src/reasoning/         registration.py, fusion.py, spatial.py, threat.py
src/events/            event_manager.py
src/utils/             types, image_utils, visualization, system_stats, logger
src/pipeline/          console.py             (orchestrator, see §6)
ui/                    console_view.py, image_tab/video_tab (input), model_manager (CV half)
app.py, run_cli.py
```

### Repository B — AI Mission Analyst (extracts)

```
src/memory/            schema.py, mission_store.py          (567 LOC, zero CV imports)
src/query/             assistant.py, intents.py, prompts.py (636 LOC, zero CV imports)
src/intelligence/      mission_intelligence.py, mission_report.py, exporters.py
src/vlm/               qwen_vlm.py, vlm_worker.py, prompts.py
ui/                    timeline_tab, objects_tab, thermal_tab, query_tab, reports_tab, common
```

The review confirms `memory/`, `query/`, `vlm/qwen_vlm.py`, `vlm/prompts.py`,
`intelligence/mission_report.py`, and `intelligence/exporters.py` have **zero**
internal CV imports today — they move with no edits.

### 3.3 Shared code → a small shared package

Three things are needed by both repos. Publish them as a tiny versioned package
(`eoir-contracts`, or vendor by copy initially):

- **Data contracts:** `Detection`, `FrameResult`, `Modality`, `Event`, `Interaction`,
  `FusionAssessment` — the serializable record types crossing the boundary.
  → Move `Event` out of `events/event_manager.py` into `utils/types.py` **now**
  (fixes the two trivial couplings before any split).
- **Config primitives:** the Pydantic base + the subset of `settings.py` each side needs.
  Split `AppConfig` into `ConsoleConfig` (Repo A) and `AnalystConfig` (Repo B); both
  re-export shared primitives.
- **`logger`** (Loguru wrapper) — duplicate or share.

---

## 4. How the two repositories communicate

The contract is **the mission record**, not Python imports. Repo A produces
mission data; Repo B consumes it. Two supported transports:

### 4.1 Primary contract — the SQLite mission archive (already exists)
`MissionStore` already writes a self-contained `mission_memory.db` with a stable
schema (`schema.py`, `SCHEMA_VERSION`). This **is** the interface.

- **Repo A** keeps a *write-only* shim around the DB (or a thin client) so the live
  console records frames/events/threat/thermal exactly as today.
- **Repo B** owns the full `MissionStore` (reads + the async writer) plus all
  readers (`query/`, `mission_report.py`). It treats the DB as its system of record.

Because `schema.py` and the readers have **no CV imports**, the entire read/query/
report stack moves to Repo B untouched. The only thing Repo A needs is the *writer*
surface (`record_frame/record_events/update_threat/record_interaction/record_thermal/
update_object_media/open_mission/close_mission`).

**Decision required:** does the live writer live in Repo A (duplicated thin writer
against the shared schema) or does Repo A depend on a small `eoir-store` package
that both share? Recommended: **shared `eoir-store` package** exposing the schema +
writer + reader, so the schema has one owner. (See §7 Open Questions.)

### 4.2 Streaming contract — for live AI reactions (optional, later)
For live VLM cueing without a shared process, Repo A emits an **event/result stream**
(JSON lines over a local socket, ZeroMQ, or a Redis/SQLite queue). `VLMRequest` is
already a clean dataclass (image + detections + events + notes + timestamps) — it
becomes the wire message. Repo B's `VLMWorker` consumes it. This replaces the current
in-process `vlm_worker.submit(...)` call inside `console.py`.

```
Repo A (Console)                         Repo B (Analyst)
  capture→detect→track→fuse              VLMWorker  ── Qwen2.5-VL
        │                                MissionStore reads
        ├── writes mission rows ───────► query/ + mission_report/
        └── emits VLMRequest stream ───► VLMWorker.submit()
                                              │
        ◄──── (optional) summary/alert ◄──────┘  back-channel for operator display
```

For demo/portfolio simplicity, **4.1 alone is sufficient** (Repo B reads the DB and
generates reports/answers offline). 4.2 is only needed if live VLM overlays must
come from a *separate process*.

---

## 5. Shared dependencies

| Dependency | Repo A | Repo B | Shared? |
|---|---|---|---|
| numpy, opencv | ✅ | ✅ (light) | both |
| torch | ✅ (detector) | ✅ (VLM) | each pins its own |
| ultralytics / supervision | ✅ | ❌ | A only |
| transformers / bitsandbytes / qwen-vl-utils | ❌ | ✅ | B only |
| reportlab / openpyxl | ❌ | ✅ | B only (exporters) |
| sqlite3 (stdlib) | ✅ (writer) | ✅ (full) | both, via shared schema |
| pydantic / loguru | ✅ | ✅ | shared primitives |

**Key win:** splitting removes `transformers`/`bitsandbytes`/`reportlab`/`openpyxl`
from Repo A's install and `ultralytics`/`supervision` from Repo B's — each repo gets a
lighter, faster, less OOM-prone environment. This directly helps the Console's 8 GB
budget (no VLM libs loaded in the real-time process if VLM runs in Repo B).

---

## 6. The two hard seams (must be handled)

### Seam 1 — `analyzers.py` (offline Image/Video analyzers)
It instantiates the entire CV pipeline **and** the AI stack — it is a vertical slice,
not a layer. Options:

- **A (recommended):** Split it. `VideoAnalyzer`/`ImageAnalyzer` are *Console*
  features (they run the CV pipeline). Keep them in **Repo A**, and have them write to
  the shared store + emit the same records the live path does. Repo B then reports on
  them identically to a live mission. Minimal conceptual change.
- **B:** Move to Repo B and have it call Repo A's CV pipeline as a library dependency
  — but that re-introduces a B→A dependency and a heavy CV install in B. **Reject.**
- **C:** Leave as a temporary bridge module in whichever repo holds the orchestrator
  until time allows option A. Acceptable as an interim.

### Seam 2 — `pipeline/console.py` orchestrator
It wires CV + AI together in one process. After the split it becomes a **Repo A**
component that talks to Repo B via §4 (DB writer + optional VLMRequest stream) instead
of importing `MissionIntelligence`/`MissionStore`/`VLMWorker` directly. Concretely:

- `self.mission` (MissionIntelligence): the *live aggregation* it does (intel feed,
  timeline view, stats) is needed for the operator console **now**. Two choices:
  keep a lightweight live-aggregator in Repo A, or read it back from the store. Keeping
  a thin live view in A is simpler for latency (no DB round-trip on the hot path).
- `self.store`: replace with the shared writer shim.
- `self.vlm` / `self.vlm_worker`: move behind §4.2 stream (or keep in-process during
  the transition with a feature flag).

### Seam 3 — the `Event` (and friends) data classes
Move `Event` (and ideally `Interaction`, `FusionAssessment` record types) into the
shared `utils/types`/contracts package **first**. This is a one-line import change in
`mission_intelligence.py` and `vlm_worker.py` and unblocks everything else.

---

## 7. Least-disruptive migration sequence

Do this **in the current monorepo first** (refactor in place), then physically split.
Each step keeps the app runnable and tests green.

**Phase 0 — Prep (in place, no behavior change)**
1. Move `Event` + sibling record types into `utils/types.py`; fix the 2 imports.
2. Introduce a `contracts` (or `eoir_core`) sub-package; re-export the shared data
   classes, config primitives, logger.
3. Split `AppConfig` into `ConsoleConfig` + `AnalystConfig` that compose shared primitives.

**Phase 1 — Define the boundary (in place)**
4. Wrap all DB writes from `console.py`/`analyzers.py` behind a narrow `StoreWriter`
   interface; wrap reads behind `StoreReader`. Confirm CV code only ever calls `StoreWriter`.
5. Make `console.py` talk to the VLM only through a `VlmGateway` interface (in-process
   impl today; stream impl later).

**Phase 2 — Extract Repo B (physical)**
6. Create `ai-mission-analyst` repo. Move `memory/`, `query/`, `vlm/`,
   `intelligence/{mission_intelligence,mission_report,exporters}.py`, and the 5
   memory-backed UI tabs. Add the shared package as a dependency.
7. Repo B ships: a report CLI/service (`mission_report.gather → render`), the
   QueryAssistant service, and (optionally) the `VLMWorker` stream consumer.

**Phase 3 — Slim Repo A (physical)**
8. In `eoir-mission-console`, remove `transformers`/`bitsandbytes`/`reportlab`/
   `openpyxl` from deps. Replace direct AI imports with the shared writer + VlmGateway.
9. Decide `analyzers.py` per Seam 1 (recommend: stays in A, writes to store).

**Phase 4 — Integrate & verify**
10. Run Console (Repo A) recording to the shared DB; run Analyst (Repo B) reading it
    for reports/queries. Validate parity with today's outputs.

Phases 0–1 are pure in-place refactors — safe to land immediately and they make the
eventual physical split trivial.

---

## 8. Open questions for you

1. **Schema ownership:** shared `eoir-store` package (recommended) vs. duplicated thin
   writer in Repo A?
2. **Live VLM overlays:** must live VLM summaries appear on the operator console in
   real time (needs §4.2 stream), or is offline/near-real-time reporting from Repo B
   acceptable for now?
3. **`analyzers.py`:** keep offline analyzers in the Console (Repo A) — agreed?
4. **Monorepo vs. polyrepo timing:** land Phases 0–1 now, defer physical split until
   after latency work? (Recommended — latency fixes are higher priority per your prompt.)

---

*This document plans the separation only. No modules have been moved. See the
prioritized implementation plan for sequencing against the latency and detector work.*
