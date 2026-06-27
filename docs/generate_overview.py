"""Generate the project overview PDF (features + work done).

Run:  .venv\\Scripts\\python docs\\generate_overview.py
Output: docs/Vision-Assistant-Overview.pdf

Self-contained (only reportlab). Content is hand-curated to match the actual
contents of the repository.
"""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (HRFlowable, ListFlowable, ListItem, PageBreak,
                                Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

NAVY = colors.HexColor("#1F4E78")
LIGHT = colors.HexColor("#F2F6FB")
ACCENT = colors.HexColor("#2E7D32")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Title"], textColor=NAVY, fontSize=24,
                    leading=28, spaceAfter=2)
SUB = ParagraphStyle("SUB", parent=styles["Normal"], fontSize=11,
                     textColor=colors.HexColor("#444444"), alignment=TA_CENTER,
                     leading=15)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=NAVY,
                    fontSize=14, spaceBefore=14, spaceAfter=4)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], textColor=ACCENT,
                    fontSize=11, spaceBefore=8, spaceAfter=2)
BODY = ParagraphStyle("BODY", parent=styles["BodyText"], fontSize=9.7,
                      leading=14, alignment=TA_LEFT)
SMALL = ParagraphStyle("SMALL", parent=styles["Normal"], fontSize=8,
                       textColor=colors.grey)
CELL = ParagraphStyle("CELL", parent=BODY, fontSize=8.6, leading=11.5)
CELLB = ParagraphStyle("CELLB", parent=CELL, textColor=NAVY, fontName="Helvetica-Bold")
MONO = ParagraphStyle("MONO", parent=styles["Code"], fontSize=7.6, leading=9.2,
                      textColor=colors.HexColor("#222222"))


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def bullets(items):
    return ListFlowable(
        [ListItem(Paragraph(t, BODY), leftIndent=6, value="•") for t in items],
        bulletType="bullet", start="•", leftIndent=12, spaceBefore=2, spaceAfter=2)


def kv_table(rows, head, col_w):
    data = [[Paragraph(h, CELLB) for h in head]]
    for r in rows:
        data.append([Paragraph(c, CELL) for c in r])
    t = Table(data, colWidths=col_w, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9D6E5")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    for i, r in enumerate(data[1:], start=1):
        data[i][0].style = CELLB
    return t


def rule():
    return HRFlowable(width="100%", thickness=0.8, color=colors.HexColor("#C9D6E5"),
                      spaceBefore=6, spaceAfter=6)


story = []

# ── Title ────────────────────────────────────────────────────────────────
story += [
    Spacer(1, 8),
    Paragraph("&#128752;  EO/IR Mission Console", H1),
    Paragraph("Project Overview &amp; Engineering Summary", SUB),
    Spacer(1, 4),
    Paragraph("A dual-sensor operator console for real-time object detection, "
              "tracking, re-identification, cross-sensor EO/IR fusion, threat "
              "scoring, persistent mission memory, and VLM-driven mission "
              "intelligence — built to run locally on an "
              "NVIDIA RTX 5060 (8&nbsp;GB, Blackwell).", SUB),
    Spacer(1, 10), rule(),
]

# ── 1. What it is ──────────────────────────────────────────────────────────
story.append(Paragraph("1 &nbsp; What the System Does", H2))
story.append(Paragraph(
    "The application drives <b>two independent sensor streams</b> — a wide/zoom "
    "electro-optical (EO) channel and an infrared (IR/thermal) channel — through "
    "a shared real-time pipeline. It runs <b>YOLOv11</b> detection and "
    "<b>ByteTrack</b> tracking, gives every target a <b>stable identity</b> via an "
    "appearance re-identification layer, <b>automatically registers the two "
    "uncalibrated sensors</b> from shared targets to fuse thermal contacts into "
    "the EO view, reasons over spatial relationships and <b>threat level</b> per "
    "target, and selectively invokes a <b>Qwen2.5-VL</b> Vision-Language Model to "
    "produce natural-language mission intelligence. Everything is persisted to a "
    "<b>SQLite mission archive</b> so a mission can be queried in natural language "
    "and exported as an enriched report (Markdown, JSON, Excel, illustrated PDF).",
    BODY))

story.append(Paragraph("Operator console + analysis modes", H3))
story.append(bullets([
    "<b>Live Operator Console</b> — two RTSP/video sources run concurrently; the "
    "EO feed is primary with an IR <b>picture-in-picture</b> (swap + fusion "
    "toggle), live boxes, stable IDs, telemetry, and right-hand operational "
    "panels (timeline, target activity, query, reporting) over the live mission.",
    "<b>Video Analysis</b> — a full-file pass (EO-only, IR-only, or an EO+IR pair) "
    "producing a timeline-based mission summary with an overall VLM summary.",
    "<b>Still-Image Analysis</b> — a single EO or IR image, or an EO+IR pair, "
    "producing EO analysis, IR analysis, and a fused assessment.",
]))

# ── 2. Architecture ────────────────────────────────────────────────────────
story.append(Paragraph("2 &nbsp; Architecture &amp; Data Flow", H2))
story.append(Paragraph(
    "The core design principle is to <b>decouple cheap, every-frame compute "
    "(detection + tracking) from the expensive VLM</b>, which is gated behind a "
    "rate-limited async worker. A second principle keeps the system inside the "
    "<b>8&nbsp;GB VRAM budget</b>: the two sensor streams do not run two models — "
    "<b>one inference thread round-robins both streams through the shared "
    "YOLO</b> (no concurrent CUDA, no second detector). The EO stream is primary "
    "and drives memory / events / spatial / threat; IR contributes thermal and, "
    "once registered, fused contacts.", BODY))
flow = (
    "  EO  RTSP/Video ---+                                   +--- IR  RTSP/Video\n"
    "                    v                                   v\n"
    "          [ Ingestion: drop-oldest FrameBuffer per stream ]   (capture threads)\n"
    "                    |                                   |\n"
    "                    +------>  one shared YOLOv11  <-----+   round-robin (inference thread)\n"
    "                    v                                   v\n"
    "      [ ByteTrack + ReID: stable IDs ]      [ ByteTrack + ReID: stable IDs ]\n"
    "                    |                                   |\n"
    "                    |        [ CrossSensorRegistrar ]   |   IR->EO similarity transform\n"
    "                    |          (RANSAC over shared       |   (auto, no calibration)\n"
    "                    |           targets, lock FSM)       |\n"
    "                    v                  |                 v\n"
    "          [ Event Mgr ]               +--> [ FusionEngine ] project IR heat into EO\n"
    "                    |                                   |\n"
    "          [ Spatial ] -> [ Threat scoring ] <----------+\n"
    "                    |                                   |\n"
    "          significant events                           v\n"
    "                    +------------------>  [ VLM Worker ]  Qwen2.5-VL (async, gated)\n"
    "                    v                                   v\n"
    "          [ MissionStore (SQLite, WAL) ] <-> [ Query Assistant ] + enriched reports"
)
story.append(Preformatted(flow, MONO))

# ── 3. Module map ──────────────────────────────────────────────────────────
story.append(PageBreak())
story.append(Paragraph("3 &nbsp; Component Map", H2))
story.append(Paragraph(
    "~7,800 lines of Python across a modular <font face='Courier'>src/</font> "
    "package and a Streamlit <font face='Courier'>ui/</font> layer. Every stage "
    "is independently testable and swappable — the entire pipeline is unit-tested "
    "GPU-free via injectable detectors and synthetic databases.", BODY))
modules = [
    ("app.py / run_cli.py", "Operator-console UI entry point and a headless CLI runner for RTSP/video → intel."),
    ("config/ + config.yaml", "All runtime behaviour driven from one YAML file, loaded into typed Pydantic models."),
    ("ingestion/*", "Drop-oldest FrameBuffer, RTSP/webcam handler (auto-reconnect, real-FPS), video reader."),
    ("detection/detector", "YOLOv11 (Ultralytics) wrapper with FP16 inference and warmup."),
    ("tracking/tracker + reid", "ByteTrack association + model-free appearance ReID → stable global IDs."),
    ("events/event_manager", "Significant-event detection (the VLM-trigger layer) + per-identity memory."),
    ("reasoning/spatial", "Model-free geometry: proximity, following, grouping, restricted-area interactions."),
    ("reasoning/registration  [new]", "CrossSensorRegistrar — auto IR→EO similarity transform via RANSAC over shared targets."),
    ("reasoning/fusion", "Projects IR thermal blobs into the EO frame; adaptive 8-bit hot-blob + polarity handling."),
    ("reasoning/threat", "ThreatScorer — explainable, time-decaying per-identity threat banding (NONE…CRITICAL)."),
    ("memory/  [new]", "MissionStore — async non-blocking SQLite writer + WAL reads; full mission archive."),
    ("query/  [new]", "QueryAssistant — NL questions → 10 whitelisted intents (no free-form SQL)."),
    ("vlm/qwen_vlm + vlm_worker", "Qwen2.5-VL 4-bit inference (EO/IR/fused prompts) behind an async, gated scheduler."),
    ("intelligence/mission_intelligence", "Live intel feed, timeline, stats, object snapshots, report export."),
    ("intelligence/mission_report  [new]", "Enriched report (md/json/xlsx/pdf) over any archived mission, incl. grounded Q&amp;A."),
    ("intelligence/analyzers + exporters", "Offline Image/Video analyzers and the Excel/PDF report builders."),
    ("pipeline/console  [new]", "MissionConsole — dual-stream EO+IR backend, shared-YOLO round-robin, fusion gating."),
    ("pipeline/pipeline", "Original single-stream real-time orchestrator (kept for single-sensor use)."),
    ("utils/*", "Logger (Loguru), shared types, image utils, drawing/HUD, NVML GPU telemetry."),
    ("ui/*", "Console view + operational-panel modules over a cached shared model manager."),
]
story.append(kv_table(modules, ["Module", "Responsibility"], [62 * mm, 108 * mm]))

# ── 4. Significant events ──────────────────────────────────────────────────
story.append(Paragraph("4 &nbsp; Significant Events (VLM Triggers)", H2))
story.append(Paragraph(
    "The Event Manager maintains per-track motion histories and emits discrete, "
    "de-duplicated events. All thresholds are tunable in "
    "<font face='Courier'>config.yaml → events</font>.", BODY))
events = [
    ("NEW_OBJECT", "a target is seen for the very first time this session"),
    ("OBJECT_REIDENTIFIED  [new]", "a previously-seen target re-appears (recognised by ReID — no longer mis-reported as 'new')"),
    ("OBJECT_LEFT", "a confirmed track disappears from the scene"),
    ("STOPPED", "a moving target becomes stationary for N frames"),
    ("DIRECTION_CHANGE", "a target's heading changes beyond a threshold"),
    ("PERSON_ENTER_VEHICLE", "a person track vanishes adjacent to a vehicle"),
    ("THERMAL_SIGNATURE", "a new hot region appears (IR modality)"),
    ("CROWD_FORMED", "person count crosses a density threshold"),
    ("THREAT_ESCALATION  [new]", "a target's threat band rises (highest priority — drives VLM ordering)"),
]
story.append(kv_table(events, ["Event", "Fires when…"], [62 * mm, 108 * mm]))

# ── 5. Engineering work done ───────────────────────────────────────────────
story.append(PageBreak())
story.append(Paragraph("5 &nbsp; Engineering Work Delivered", H2))
story.append(Paragraph(
    "Beyond the core detection / tracking / VLM pipeline, the following "
    "capabilities were designed, implemented, and smoke-tested in this codebase:",
    BODY))

story.append(Paragraph("5.1 &nbsp; Cross-sensor EO/IR registration &amp; fusion (calibration-free)", H3))
story.append(Paragraph(
    "The operator's EO and IR are <b>separate, unaligned optics</b> (EO 66° wide "
    "+ 40× zoom; IR ~12° FOV) with <b>no boresight, no telemetry, no "
    "calibration</b>, and the IR is 8-bit AGC (auto-gain, polarity can invert). "
    "A new <font face='Courier'>CrossSensorRegistrar</font> recovers the IR→EO "
    "<b>similarity transform</b> (scale + rotation + translation) automatically "
    "from <b>shared targets</b>: a deterministic 2-point <b>RANSAC over object / "
    "blob centroids</b> (geometry, not cross-spectral appearance — which dodges "
    "thermal intensity inversion), EMA-smoothed behind a lock state machine "
    "(UNLOCKED→ACQUIRING→LOCKED→WARM). The <b>recovered scale is the relative "
    "zoom</b>, so no zoom telemetry is needed. The reworked "
    "<font face='Courier'>FusionEngine</font> then projects IR thermal blobs into "
    "the EO frame (adaptive 8-bit hot-blob thresholding + white/black-hot "
    "polarity) and raises a <b>CONTACT_OUTSIDE_EO</b> slew cue when heat appears "
    "beyond the EO field of view.", BODY))

story.append(Paragraph("5.2 &nbsp; Dual-stream operator console", H3))
story.append(Paragraph(
    "A new <font face='Courier'>MissionConsole</font> backend runs <b>two "
    "independent sources</b> (EO + IR) with one inference thread <b>round-robining "
    "both through the shared YOLO</b> — 8&nbsp;GB-safe, no second model, no "
    "concurrent CUDA. EO is primary (memory / events / spatial / threat); IR "
    "contributes thermal, and fusion is gated on both streams being connected and "
    "locked. The UI was rebuilt into an <b>operator console</b>: a config sidebar, "
    "a center EO feed with IR <b>picture-in-picture</b> (swap + fusion toggles), "
    "and right-hand operational panels (Timeline / Target Activity / Query / "
    "Reporting) over the live mission.", BODY))

story.append(Paragraph("5.3 &nbsp; Persistent mission memory (SQLite)", H3))
story.append(Paragraph(
    "A <font face='Courier'>MissionStore</font> persists every mission to "
    "<b>SQLite</b> via an async non-blocking writer thread with WAL reads: "
    "missions, per-identity object lifecycle, detections, events, VLM "
    "observations, spatial interactions, and thermal/fusion events. This turns a "
    "live session into a queryable, reportable archive (SQLite chosen over "
    "Postgres deliberately — single-file, zero-ops, fits the local box).", BODY))

story.append(Paragraph("5.4 &nbsp; Spatial reasoning &amp; explainable threat scoring", H3))
story.append(Paragraph(
    "A model-free <font face='Courier'>SpatialReasoner</font> derives "
    "<b>proximity, following, grouping, and restricted-area</b> interactions from "
    "per-track geometry (streak + cooldown noise suppression). A "
    "<font face='Courier'>ThreatScorer</font> synthesis layer (no new detector) "
    "combines events, interactions, fusions, and dwell into <b>explainable, "
    "weighted, per-second-decaying</b> threat factors, banding each identity "
    "NONE→LOW→MEDIUM→HIGH→CRITICAL and emitting a "
    "<b>THREAT_ESCALATION</b> event on band increase.", BODY))

story.append(Paragraph("5.5 &nbsp; Natural-language Query Assistant (injection-proof)", H3))
story.append(Paragraph(
    "A <font face='Courier'>QueryAssistant</font> answers natural-language "
    "questions over the mission archive with <b>no free-form SQL</b>: each "
    "question routes to one of <b>10 whitelisted intents</b>, each owning a single "
    "hard-coded, parameterised, read-only query — so injection is impossible by "
    "construction. The <b>headline answer is computed in code</b>; the reused "
    "Qwen2.5-VL (text-only) only reclassifies and rephrases — it <i>never "
    "fabricates numbers</i>. Works with or without the VLM (keyword fallback "
    "classifier).", BODY))

story.append(Paragraph("5.6 &nbsp; Enriched reports, ReID, telemetry &amp; robustness", H3))
story.append(bullets([
    "<b>Enriched mission reports</b> — <font face='Courier'>mission_report</font> "
    "builds a report (Markdown / JSON / Excel / illustrated PDF) over <i>any</i> "
    "archived mission: threat overview, object thumbnail gallery, merged timeline, "
    "and an optional grounded Q&amp;A section via the Query Assistant.",
    "<b>Appearance ReID</b> — a model-free ReID layer (HSV histograms for EO, "
    "intensity+gradient for IR) assigns stable global IDs across ByteTrack id "
    "churn, so a returning target reads <i>'re-appeared'</i>, not <i>'new'</i> "
    "(no extra GPU model — respects the 8&nbsp;GB budget).",
    "<b>Live telemetry &amp; GPU panel</b> — stream resolution, true measured FPS, "
    "processing FPS, and real-time GPU/VRAM/utilisation via NVML (the in-app "
    "answer to the Task-Manager 'CUDA shows 0%' confusion).",
    "<b>RTSP FPS fix</b> — rejects the bogus 90000&nbsp;fps H.264 RTP-clock value "
    "and measures real cadence from frame arrivals.",
    "<b>ISR prompts</b> — person-characteristics and suspicious-item VLM prompt "
    "templates for EO and IR.",
]))

# ── 6. Tech stack ──────────────────────────────────────────────────────────
story.append(Paragraph("6 &nbsp; Technology Stack", H2))
stack = [
    ("Detection", "YOLOv11 (Ultralytics) — FP16, anchor-free, TensorRT-exportable"),
    ("Tracking", "ByteTrack via supervision (high+low-confidence association)"),
    ("Re-ID", "OpenCV histogram appearance signatures (CPU, no extra model)"),
    ("EO/IR fusion", "Calibration-free similarity registration (RANSAC over shared targets) + adaptive 8-bit thermal"),
    ("Reasoning", "Qwen2.5-VL (3B/7B) in 4-bit via transformers + bitsandbytes; in-code spatial + threat synthesis"),
    ("Persistence", "SQLite (WAL) async mission archive — missions, objects, events, interactions, thermal"),
    ("Query", "Whitelisted-intent NL assistant (no free-form SQL) over the archive"),
    ("UI", "Streamlit operator console — EO feed + IR PiP + operational panels"),
    ("Reporting", "openpyxl (Excel) · reportlab (PDF) · Markdown/JSON"),
    ("Telemetry", "nvidia-ml-py (NVML) GPU/VRAM, psutil"),
    ("Platform", "Python 3.10 · CUDA 12.8 (Blackwell) · RTX 5060 8 GB"),
]
story.append(kv_table(stack, ["Stage", "Technology"], [42 * mm, 128 * mm]))

# ── 7. Run it ──────────────────────────────────────────────────────────────
story.append(Paragraph("7 &nbsp; Running the Application", H2))
story.append(Preformatted(
    "# Operator console (EO + IR)\n"
    "  .venv\\Scripts\\streamlit run app.py\n\n"
    "# Headless CLI — RTSP for 60s, export a report on exit\n"
    "  python run_cli.py --source rtsp://user:pass@host:554/stream --duration 60 --report\n\n"
    "# All behaviour is tunable in config/config.yaml (detection, tracking, reid,\n"
    "# vlm, events, spatial, fusion, registration, eo_sensor/ir_sensor, threat,\n"
    "# memory, query, mission, ui). Edit and restart to apply.", MONO))
story.append(Paragraph(
    "<b>8&nbsp;GB VRAM note.</b> The VLM runs in 4-bit and defaults to the 3B "
    "variant for comfortable headroom alongside YOLO; the full stack uses roughly "
    "4.8&nbsp;GB. The shared model manager keeps weights loaded once per process.",
    BODY))

story.append(Spacer(1, 10))
story.append(rule())
story.append(Paragraph(
    f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · "
    "EO/IR Mission Intelligence Assistant — engineering overview.", SMALL))


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(200 * mm, 10 * mm, f"Page {doc.page}")
    canvas.drawString(18 * mm, 10 * mm, "EO/IR Mission Intelligence Assistant")
    canvas.restoreState()


def build(path: Path) -> None:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="EO/IR Mission Intelligence Assistant — Overview",
                            author="Vision-Assistant")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    path.write_bytes(buf.getvalue())


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "Vision-Assistant-Overview.pdf"
    build(out)
    print(f"Wrote {out} ({out.stat().st_size} bytes)")
