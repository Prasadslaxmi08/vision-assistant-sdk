"""Enriched mission reports built from persisted memory (V2 phase 6).

:class:`MissionIntelligence` builds reports from *live* in-memory state. This
module reconstructs a full report for ANY mission already archived in the
:class:`~src.memory.mission_store.MissionStore` (SQLite), folding in everything
the V2 reasoning layers persisted:

  * events (incl. ``THREAT_ESCALATION``), spatial interactions, EO/IR thermal /
    fusion assessments — merged into one chronological timeline;
  * per-identity lifecycle + threat levels (objects table);
  * the best-thumbnail object gallery (objects.snapshot blobs);
  * an optional natural-language **Q&A section** from the Phase-5 Query Assistant.

Output bytes feed the same exporters (:mod:`src.intelligence.exporters`) used by
the live pipeline, so report formatting stays consistent across the app.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import logger

# Questions auto-answered in the report's Q&A section (grounded by mission memory).
_REPORT_QUESTIONS = [
    "Summarize the mission",
    "List suspicious activities",
    "What is the overall threat level?",
    "How many unique people were there?",
]
_LEVEL_RANK = {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def _fmt_ts(seconds: Optional[float]) -> str:
    s = max(0.0, float(seconds or 0.0))
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


@dataclass
class ReportData:
    mission_id: int
    title: str
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)
    final_summary: str = ""
    threat_headline: str = ""
    snapshots: List[Dict[str, Any]] = field(default_factory=list)
    intel_feed: List[str] = field(default_factory=list)
    qa: List[Tuple[str, str]] = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Gather
# --------------------------------------------------------------------------- #
def gather(store, mission_id: int, qa_assistant=None) -> ReportData:
    """Reconstruct all report inputs for ``mission_id`` from the store."""
    missions = {m["mission_id"]: m for m in store.list_missions()}
    meta = missions.get(mission_id, {})
    name = meta.get("name") or f"mission {mission_id}"
    title = f"Mission Report — {name}"

    objects = store.list_objects(mission_id)
    events = store.read(
        "SELECT source_ts, type, description FROM events WHERE mission_id=? "
        "ORDER BY source_ts", (mission_id,))
    interactions = store.interactions(mission_id)
    thermal = store.thermal_events(mission_id)
    vlm = store.read(
        "SELECT source_ts, kind, text FROM vlm_observations WHERE mission_id=? "
        "ORDER BY source_ts", (mission_id,))

    # -- Merged, chronological timeline ------------------------------------
    rows: List[Dict[str, Any]] = []
    for e in events:
        rows.append({"source_ts": e["source_ts"], "type": e["type"],
                     "detail": e["description"], "kind": "event"})
    for it in interactions:
        rows.append({"source_ts": it["source_ts"], "type": it["type"],
                     "detail": it["description"], "kind": "interaction"})
    for th in thermal:
        rows.append({"source_ts": th["source_ts"], "type": "THERMAL",
                     "detail": th["description"], "kind": "fusion"})
    for v in vlm:
        rows.append({"source_ts": v["source_ts"],
                     "type": "VLM_" + str(v["kind"]).upper(),
                     "detail": v["text"], "kind": "summary"})
    rows.sort(key=lambda r: r["source_ts"])
    timeline = [{"time": _fmt_ts(r["source_ts"]), "type": r["type"],
                 "detail": r["detail"], "kind": r["kind"]} for r in rows]

    # -- Statistics --------------------------------------------------------
    by_class = Counter(o["class_name"] for o in objects if o["class_name"])
    people = sum(1 for o in objects if o["class_name"] == "person")
    ev_counter = Counter(e["type"] for e in events)
    stats = {
        "unique_objects": len(objects),
        "unique_people": people,
        "by_class": dict(by_class),
        "events": dict(ev_counter),
        "timeline_len": len(timeline),
        "interactions": len(interactions),
        "thermal_events": len(thermal),
    }

    # -- Threat headline ---------------------------------------------------
    threatful = [o for o in objects if (o.get("threat_score") or 0) > 0]
    if threatful:
        top = max(threatful, key=lambda o: o["threat_score"])
        worst = max(threatful, key=lambda o: _LEVEL_RANK.get(o["threat_level"], 0))
        threat_headline = (
            f"Peak threat {worst['threat_level']} — highest scorer "
            f"{top['class_name']} #{top['global_id']} ({top['threat_score']:.1f}); "
            f"{len(threatful)} object(s) carried a non-zero threat score.")
    else:
        threat_headline = "Mission threat level NONE — no elevated-threat objects."

    # -- Object thumbnail gallery -----------------------------------------
    snapshots = [
        {"jpeg": o["snapshot"], "identity": o["global_id"],
         "class_name": o["class_name"], "confidence": o.get("max_confidence", 0.0),
         "time": _fmt_ts(o.get("first_seen_ts"))}
        for o in objects if o.get("snapshot")]

    # -- Narrative summary + Q&A ------------------------------------------
    last_summary = next((v["text"] for v in reversed(vlm)
                         if str(v["kind"]).lower() == "summary"), "")
    qa_pairs: List[Tuple[str, str]] = []
    if qa_assistant is not None:
        for q in _REPORT_QUESTIONS:
            try:
                qa_pairs.append((q, qa_assistant.ask(q, mission_id).answer))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Report Q&A '{}' failed: {}", q, exc)
        if not last_summary and qa_pairs:
            last_summary = qa_pairs[0][1]

    final_summary = " ".join(
        s for s in (last_summary, threat_headline) if s).strip() \
        or "No VLM summary was generated for this mission."

    intel_feed = [f"[{r['time']}] {r['detail']}" for r in timeline
                  if r["kind"] in ("event", "interaction", "fusion")][-30:]

    return ReportData(mission_id=mission_id, title=title, timeline=timeline,
                      stats=stats, final_summary=final_summary,
                      threat_headline=threat_headline, snapshots=snapshots,
                      intel_feed=intel_feed, qa=qa_pairs)


# --------------------------------------------------------------------------- #
#  Renderers
# --------------------------------------------------------------------------- #
def to_markdown(data: ReportData) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    s = data.stats
    lines = [f"# {data.title}", "", f"_Generated: {now}_", "",
             "## Executive Summary", "", data.final_summary, "",
             "## Threat Overview", "", data.threat_headline, "",
             "## Statistics", "",
             f"- Unique tracked objects: **{s.get('unique_objects', 0)}** "
             f"({s.get('unique_people', 0)} people)",
             f"- Spatial interactions: **{s.get('interactions', 0)}**",
             f"- Thermal/IR events: **{s.get('thermal_events', 0)}**"]
    for cls, n in (s.get("by_class") or {}).items():
        lines.append(f"  - {cls}: {n}")
    lines += ["", "### Event counts"]
    for ev, n in (s.get("events") or {}).items():
        lines.append(f"- {ev}: {n}")
    if data.qa:
        lines += ["", "## Mission Q&A", ""]
        for q, a in data.qa:
            lines += [f"**Q: {q}**", "", a, ""]
    lines += ["", "## Timeline", "", "| Time | Type | Detail |",
              "|------|------|--------|"]
    for e in data.timeline:
        lines.append(f"| {e['time']} | {e['type']} | "
                     f"{e['detail'].replace('|', chr(92) + '|')} |")
    return "\n".join(lines)


def to_json_bytes(data: ReportData) -> bytes:
    payload = {
        "mission_id": data.mission_id,
        "title": data.title,
        "stats": data.stats,
        "threat_headline": data.threat_headline,
        "final_summary": data.final_summary,
        "qa": [{"question": q, "answer": a} for q, a in data.qa],
        "timeline": data.timeline,
        "intel_feed": data.intel_feed,
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def to_excel_bytes(data: ReportData) -> bytes:
    from src.intelligence.exporters import timeline_to_xlsx
    return timeline_to_xlsx(data.timeline, data.stats,
                            final_summary=data.final_summary, title=data.title)


def to_pdf_bytes(data: ReportData) -> bytes:
    from src.intelligence.exporters import mission_to_pdf
    # Fold the Q&A into the intel feed so it appears in the printable report.
    intel = list(data.intel_feed)
    if data.qa:
        intel = [f"Q&A — {q}: {a}" for q, a in data.qa] + intel
    return mission_to_pdf(data.timeline, data.stats,
                          final_summary=data.final_summary, title=data.title,
                          intel_feed=intel, snapshots=data.snapshots)


def to_all_bytes(data: ReportData) -> Dict[str, bytes]:
    """All four formats as bytes (Markdown + JSON always; Excel/PDF best-effort)."""
    out: Dict[str, bytes] = {
        "markdown": to_markdown(data).encode("utf-8"),
        "json": to_json_bytes(data),
    }
    for kind, fn in (("excel", to_excel_bytes), ("pdf", to_pdf_bytes)):
        try:
            out[kind] = fn(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("{} export skipped: {}", kind, exc)
    return out


def write_all(data: ReportData, output_dir: Path, name_prefix: str = "mission",
              blobs: Optional[Dict[str, bytes]] = None) -> Dict[str, str]:
    """Persist all formats to disk; returns the written paths by kind.

    Pass ``blobs`` (from a prior :func:`to_all_bytes`) to avoid rebuilding the
    PDF/Excel a second time.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = {"markdown": "md", "json": "json", "excel": "xlsx", "pdf": "pdf"}
    paths: Dict[str, str] = {}
    for kind, blob in (blobs if blobs is not None else to_all_bytes(data)).items():
        p = output_dir / f"{name_prefix}_{stamp}.{ext[kind]}"
        p.write_bytes(blob)
        paths[kind] = str(p)
    logger.info("Stored-mission report written: {}",
                ", ".join(Path(p).name for p in paths.values()))
    return paths
