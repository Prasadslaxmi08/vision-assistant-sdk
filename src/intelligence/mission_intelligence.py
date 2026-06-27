"""Mission intelligence aggregation and reporting.

Consumes the streams produced by the rest of the pipeline — per-frame
detections, discrete events, and VLM scene summaries — and turns them into:

  * a live, bounded **intel feed** (natural-language messages for the UI panel),
  * a chronological **timeline** keyed on media timestamps,
  * exportable **mission reports** (Markdown + JSON),
  * structured **EO/IR/fused image reports**.

This module holds no model dependencies, so it is cheap to import and easy to
unit-test. It is the natural seam where future Agentic AI / RAG layers will plug
in (see README roadmap).
"""
from __future__ import annotations

import json
import threading
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque, Dict, List, Optional

# Import the record type from the shared contract, not the CV event module
# (repo-split prep, doc 04 — removes an AI→CV import).
from src.contracts import Event
from src.utils.image_utils import crop_to_jpeg
from src.utils.logger import logger


def _fmt_ts(seconds: float) -> str:
    """Format a media-relative timestamp as HH:MM:SS."""
    seconds = max(0.0, seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@dataclass
class IntelMessage:
    timestamp: float          # wall-clock
    source_ts: float          # media position
    text: str
    kind: str = "summary"     # "summary" | "event" | "alert"

    def display(self) -> str:
        return f"[{_fmt_ts(self.source_ts)}] {self.text}"


@dataclass
class ObjectSnapshot:
    """One representative thumbnail (JPEG bytes) of a recorded object/identity."""
    identity: int
    class_name: str
    jpeg: bytes
    source_ts: float          # media time of first sighting
    confidence: float


@dataclass
class TimelineEntry:
    source_ts: float
    frame_id: int
    label: str
    detail: str
    kind: str = "event"


@dataclass
class ImageReport:
    """Structured output of single/dual-image analysis."""
    eo_analysis: str = ""
    ir_analysis: str = ""
    fused_assessment: str = ""
    detections: List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# EO/IR Image Analysis Report", ""]
        if self.detections:
            lines += ["## Automated Detections",
                      ", ".join(self.detections), ""]
        if self.eo_analysis:
            lines += ["## EO Analysis", self.eo_analysis, ""]
        if self.ir_analysis:
            lines += ["## IR Analysis", self.ir_analysis, ""]
        if self.fused_assessment:
            lines += ["## Fused Assessment", self.fused_assessment, ""]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serialisable view (Output Layer)."""
        return {
            "detections": list(self.detections),
            "eo_analysis": self.eo_analysis,
            "ir_analysis": self.ir_analysis,
            "fused_assessment": self.fused_assessment,
        }


class MissionIntelligence:
    """Thread-safe accumulator for live mission state and reporting."""

    def __init__(self, intel_history: int = 12, timeline_max: int = 500,
                 max_snapshots: int = 48):
        self._lock = threading.Lock()
        self.intel_feed: Deque[IntelMessage] = deque(maxlen=intel_history)
        self.timeline: List[TimelineEntry] = []
        self._timeline_max = timeline_max
        self.class_counter: Counter = Counter()
        self.unique_tracks: set[int] = set()
        self.event_counter: Counter = Counter()
        self.latest_summary: str = ""
        self.start_time: Optional[float] = None
        # Best thumbnail per recorded identity, for the report object gallery.
        self.object_snapshots: Dict[int, ObjectSnapshot] = {}
        self._max_snapshots = max_snapshots

    # --------------------------------------------------------- ingestion
    def note_frame(self, detections, source_ts: float, image=None) -> None:
        with self._lock:
            if self.start_time is None:
                self.start_time = source_ts
            for d in detections:
                if d.track_id is not None and d.track_id not in self.unique_tracks:
                    self.unique_tracks.add(d.track_id)
                    self.class_counter[d.class_name] += 1
            if image is not None:
                self._capture_snapshots(detections, source_ts, image)

    def _capture_snapshots(self, detections, source_ts: float, image) -> None:
        """Keep the best (highest-confidence) thumbnail per identity. Caller
        holds the lock. Re-encodes only when a clearly better crop appears, so
        the per-frame cost stays negligible once a track has stabilised."""
        for d in detections:
            tid = d.track_id
            if tid is None:
                continue
            prev = self.object_snapshots.get(tid)
            if prev is None and len(self.object_snapshots) >= self._max_snapshots:
                continue  # gallery full — don't start tracking new identities
            if prev is not None and d.confidence <= prev.confidence + 0.05:
                continue  # existing crop is already as good
            jpeg = crop_to_jpeg(image, d.bbox)
            if jpeg is None:
                continue
            self.object_snapshots[tid] = ObjectSnapshot(
                identity=tid, class_name=d.class_name, jpeg=jpeg,
                source_ts=prev.source_ts if prev else source_ts,
                confidence=d.confidence,
            )

    def _snapshots_for_report(self) -> List[Dict]:
        with self._lock:
            snaps = sorted(self.object_snapshots.values(),
                           key=lambda s: (s.source_ts, s.identity))
            return [{"jpeg": s.jpeg, "identity": s.identity,
                     "class_name": s.class_name, "confidence": s.confidence,
                     "time": _fmt_ts(s.source_ts)} for s in snaps]

    def add_events(self, events: List[Event]) -> None:
        if not events:
            return
        with self._lock:
            for ev in events:
                self.event_counter[ev.type.value] += 1
                self.timeline.append(TimelineEntry(
                    source_ts=ev.source_ts, frame_id=ev.frame_id,
                    label=ev.type.value, detail=ev.description, kind="event",
                ))
                self.intel_feed.append(IntelMessage(
                    timestamp=ev.timestamp, source_ts=ev.source_ts,
                    text=ev.description, kind="event",
                ))
            if len(self.timeline) > self._timeline_max:
                self.timeline = self.timeline[-self._timeline_max:]

    def add_interactions(self, interactions) -> None:
        """Record spatial interactions into the timeline + intel feed."""
        if not interactions:
            return
        with self._lock:
            for it in interactions:
                self.event_counter[it.type.value] += 1
                self.timeline.append(TimelineEntry(
                    source_ts=it.source_ts, frame_id=it.frame_id,
                    label=it.type.value, detail=it.description, kind="interaction",
                ))
                self.intel_feed.append(IntelMessage(
                    timestamp=it.timestamp, source_ts=it.source_ts,
                    text=it.description, kind="interaction",
                ))
            if len(self.timeline) > self._timeline_max:
                self.timeline = self.timeline[-self._timeline_max:]

    def add_fusion(self, assessments) -> None:
        """Record EO/IR fusion assessments into the timeline + intel feed."""
        if not assessments:
            return
        with self._lock:
            for a in assessments:
                self.event_counter[a.type.value] += 1
                self.timeline.append(TimelineEntry(
                    source_ts=a.source_ts, frame_id=a.frame_id,
                    label=a.type.value, detail=a.description, kind="fusion",
                ))
                self.intel_feed.append(IntelMessage(
                    timestamp=a.timestamp, source_ts=a.source_ts,
                    text=a.description, kind="fusion",
                ))
            if len(self.timeline) > self._timeline_max:
                self.timeline = self.timeline[-self._timeline_max:]

    def add_summary(self, text: str, source_ts: float,
                    timestamp: float, kind: str = "summary") -> None:
        with self._lock:
            self.latest_summary = text
            self.intel_feed.append(IntelMessage(
                timestamp=timestamp, source_ts=source_ts, text=text, kind=kind,
            ))
            self.timeline.append(TimelineEntry(
                source_ts=source_ts, frame_id=-1, label="VLM_SUMMARY",
                detail=text, kind="summary",
            ))

    # ----------------------------------------------------------- queries
    def detections_summary(self) -> str:
        with self._lock:
            if not self.class_counter:
                return "no objects observed"
            return ", ".join(f"{n} {c}{'s' if n > 1 else ''}"
                             for c, n in self.class_counter.most_common())

    def intel_messages(self) -> List[str]:
        with self._lock:
            return [m.display() for m in self.intel_feed]

    def timeline_view(self) -> List[Dict]:
        with self._lock:
            return [
                {"time": _fmt_ts(t.source_ts), "type": t.label,
                 "detail": t.detail, "kind": t.kind}
                for t in sorted(self.timeline, key=lambda e: e.source_ts)
            ]

    def stats(self) -> Dict:
        with self._lock:
            return {
                "unique_objects": len(self.unique_tracks),
                "by_class": dict(self.class_counter),
                "events": dict(self.event_counter),
                "timeline_len": len(self.timeline),
            }

    # ----------------------------------------------------------- reporting
    def build_mission_report(self, title: str = "Mission Intelligence Report",
                             final_summary: str = "") -> str:
        """Render a full Markdown mission report from accumulated state."""
        stats = self.stats()
        timeline = self.timeline_view()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        lines = [
            f"# {title}", "", f"_Generated: {now}_", "",
            "## Executive Summary", "",
            final_summary or self.latest_summary or "_No VLM summary generated._",
            "", "## Statistics", "",
            f"- Unique tracked objects: **{stats['unique_objects']}**",
        ]
        for cls, n in stats["by_class"].items():
            lines.append(f"  - {cls}: {n}")
        lines += ["", "### Event counts"]
        for ev, n in stats["events"].items():
            lines.append(f"- {ev}: {n}")
        lines += ["", "## Timeline", "", "| Time | Type | Detail |",
                  "|------|------|--------|"]
        for entry in timeline:
            detail = entry["detail"].replace("|", "\\|")
            lines.append(f"| {entry['time']} | {entry['type']} | {detail} |")
        return "\n".join(lines)

    # --------------------------------------------------------- output layer
    def to_dict(self, final_summary: str = "") -> Dict:
        """JSON-serialisable snapshot of the live scene (Output Layer).

        This is the structured handoff an external app (or the future AI Mission
        Analyst) consumes — no files, no heavy report dependencies."""
        return {
            "stats": self.stats(),
            "timeline": self.timeline_view(),
            "final_summary": final_summary or self.latest_summary,
            "intel_feed": [asdict(m) for m in self.intel_feed],
        }

    def export_json(self, output_dir: Path, name_prefix: str = "scene",
                    final_summary: str = "") -> Dict[str, str]:
        """Write a Markdown + JSON snapshot (no PDF/Excel — those moved to the
        Analyst repo). Returns the written paths."""
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_path = output_dir / f"{name_prefix}_{stamp}.md"
        json_path = output_dir / f"{name_prefix}_{stamp}.json"
        md_path.write_text(self.build_mission_report(final_summary=final_summary),
                           encoding="utf-8")
        json_path.write_text(json.dumps(self.to_dict(final_summary), indent=2),
                             encoding="utf-8")
        logger.info("Scene snapshot written: {}, {}", md_path.name, json_path.name)
        return {"markdown": str(md_path), "json": str(json_path)}

    def reset(self) -> None:
        with self._lock:
            self.intel_feed.clear()
            self.timeline.clear()
            self.class_counter.clear()
            self.unique_tracks.clear()
            self.event_counter.clear()
            self.object_snapshots.clear()
            self.latest_summary = ""
            self.start_time = None
