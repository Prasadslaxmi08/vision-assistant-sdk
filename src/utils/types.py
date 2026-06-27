"""Shared, lightweight data structures used across the pipeline.

Kept dependency-free (only dataclasses + numpy typing) so every module can
import them without creating import cycles.

This module is also the **cross-repository data contract** (repo-split prep,
doc 04 Phase 0): the serializable record types that cross the CV→AI boundary —
``Detection``, ``FrameResult``, ``Modality``, ``Event``, ``Interaction``,
``FusionAssessment`` (+ their enums) — live here so neither layer has to import
the other's modules to name them. They are re-exported by :mod:`src.contracts`
and, for back-compatibility, still importable from their original homes
(``events.event_manager``, ``reasoning.spatial``, ``reasoning.fusion``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class Modality(str, Enum):
    """Sensor modality of an incoming frame / stream."""
    EO = "EO"        # electro-optical (visible spectrum)
    IR = "IR"        # infrared / thermal
    UNKNOWN = "UNKNOWN"


@dataclass
class Frame:
    """A single captured frame plus metadata as it flows through the pipeline."""
    image: np.ndarray            # BGR uint8 HxWx3
    frame_id: int                # monotonically increasing index from the source
    timestamp: float             # wall-clock capture time (epoch seconds)
    source_ts: float = 0.0       # media timestamp in seconds (video position)
    modality: Modality = Modality.EO

    @property
    def shape(self) -> Tuple[int, int]:
        h, w = self.image.shape[:2]
        return h, w


@dataclass
class Detection:
    """One YOLO detection, optionally enriched with a tracker id."""
    bbox: Tuple[float, float, float, float]   # xyxy
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None            # stable global id after ReID
    raw_track_id: Optional[int] = None        # original ByteTrack id (debug)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass
class FrameResult:
    """Detection + tracking output for a single frame."""
    frame_id: int
    timestamp: float
    source_ts: float
    detections: List[Detection] = field(default_factory=list)
    modality: Modality = Modality.EO

    def count_by_class(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.detections:
            out[d.class_name] = out.get(d.class_name, 0) + 1
        return out


# ───────────────────────── significance / reasoning records ─────────────────
# These cross the CV→AI boundary (persisted to mission memory, fed to the VLM).
# They were defined in events/event_manager, reasoning/spatial and
# reasoning/fusion; moved here so the AI layer never imports a CV module just to
# name a record type (repo-split prep, doc 04 Seam 3).

class EventType(str, Enum):
    NEW_OBJECT = "NEW_OBJECT"
    OBJECT_REIDENTIFIED = "OBJECT_REIDENTIFIED"
    OBJECT_LEFT = "OBJECT_LEFT"
    STOPPED = "STOPPED"
    DIRECTION_CHANGE = "DIRECTION_CHANGE"
    PERSON_ENTER_VEHICLE = "PERSON_ENTER_VEHICLE"
    THERMAL_SIGNATURE = "THERMAL_SIGNATURE"
    CROWD_FORMED = "CROWD_FORMED"
    THREAT_ESCALATION = "THREAT_ESCALATION"


# Priority drives VLM-trigger ordering when several events fire at once.
EVENT_PRIORITY: Dict["EventType", int] = {
    EventType.THREAT_ESCALATION: 6,
    EventType.PERSON_ENTER_VEHICLE: 5,
    EventType.THERMAL_SIGNATURE: 4,
    EventType.DIRECTION_CHANGE: 3,
    EventType.STOPPED: 3,
    EventType.NEW_OBJECT: 2,
    EventType.OBJECT_REIDENTIFIED: 2,
    EventType.CROWD_FORMED: 2,
    EventType.OBJECT_LEFT: 1,
}


@dataclass
class Event:
    """A discrete significant occurrence emitted by the event layer."""
    type: EventType
    timestamp: float          # wall-clock
    source_ts: float          # media position (seconds) — used for timelines
    frame_id: int
    description: str
    track_ids: List[int] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    position: Optional[Tuple[float, float]] = None
    metadata: dict = field(default_factory=dict)

    @property
    def priority(self) -> int:
        return EVENT_PRIORITY.get(self.type, 0)


class InteractionType(str, Enum):
    PROXIMITY = "PROXIMITY"
    FOLLOWING = "FOLLOWING"
    GROUP = "GROUP"
    RESTRICTED_AREA = "RESTRICTED_AREA"


@dataclass
class Interaction:
    """An object-to-object spatial relationship emitted by spatial reasoning."""
    type: InteractionType
    timestamp: float
    source_ts: float
    frame_id: int
    description: str
    subject_id: int
    object_id: Optional[int] = None
    members: List[int] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class FusionType(str, Enum):
    ACTIVE_VEHICLE = "ACTIVE_VEHICLE"
    THERMAL_CONFIRMED = "THERMAL_CONFIRMED"
    CONCEALED_HEAT = "CONCEALED_HEAT"
    CONTACT_OUTSIDE_EO = "CONTACT_OUTSIDE_EO"


@dataclass
class FusionAssessment:
    """A cross-sensor EO/IR thermal↔visual correlation result."""
    type: FusionType
    timestamp: float
    source_ts: float
    frame_id: int
    description: str
    global_id: Optional[int] = None
    area_px: int = 0
    position: Optional[Tuple[float, float]] = None   # EO-frame coords (for overlay)
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)
