"""Event detection — the "significance" layer between tracking and the VLM.

YOLO + ByteTrack run on *every* frame. The VLM is expensive, so it must only
fire when something interesting happens. This module consumes per-frame
tracking results, maintains short motion histories per track, and emits
discrete :class:`Event` objects for:

  * NEW_OBJECT          — a track id seen for the first time
  * OBJECT_LEFT         — a confirmed track disappeared
  * STOPPED             — a moving target became stationary
  * DIRECTION_CHANGE    — a target's heading changed beyond a threshold
  * PERSON_ENTER_VEHICLE— a person track vanished adjacent to a vehicle
  * THERMAL_SIGNATURE   — a new hot region appeared (IR modality only)
  * CROWD_FORMED        — person count crossed a density threshold

Events carry enough context (track ids, classes, positions) for the mission
intelligence layer to build prompts and timelines.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from src.config.settings import EventsConfig
from src.utils.logger import logger
# Event/EventType/EVENT_PRIORITY now live in the shared data-contract module
# (repo-split prep, doc 04). Re-imported so existing callers can keep doing
# ``from src.events.event_manager import Event, EventType``.
from src.utils.types import (Detection, EVENT_PRIORITY, Event, EventType,
                             FrameResult, Modality)

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "train", "boat", "airplane"}
PERSON_CLASS = "person"


@dataclass
class _TrackState:
    """Rolling per-track motion history."""
    centers: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=32))
    last_bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    class_name: str = ""
    last_seen_frame: int = 0
    stationary_frames: int = 0
    stopped_reported: bool = False
    last_heading: Optional[float] = None
    last_position: Optional[Tuple[float, float]] = None


class EventManager:
    def __init__(self, config: EventsConfig):
        self.config = config
        self._tracks: Dict[int, _TrackState] = {}
        self._active_ids: set[int] = set()
        # Identities ever seen this session (survives departures) so a returning
        # target reads as "re-appeared", not "new". Relies on the ReID layer
        # handing the same global id back to the same object.
        self._known_identities: set[int] = set()
        self._cooldowns: Dict[str, float] = {}     # event-key -> last fire ts
        self._person_count_prev = 0
        self._thermal_prev_mask_area = 0
        self.event_log: List[Event] = []

    # ------------------------------------------------------------------ API
    def process(self, result: FrameResult, frame_image: Optional[np.ndarray] = None) -> List[Event]:
        """Update state from one frame's results and return new events."""
        events: List[Event] = []
        now = result.timestamp
        seen_ids: set[int] = set()

        tracked = [d for d in result.detections if d.track_id is not None]

        for det in tracked:
            tid = det.track_id  # type: ignore[assignment]
            seen_ids.add(tid)
            st = self._tracks.get(tid)
            if st is None:
                st = _TrackState(class_name=det.class_name)
                self._tracks[tid] = st
                if tid in self._known_identities:
                    # ReID recognised a target we've seen before this session.
                    ev = self._maybe(Event(
                        type=EventType.OBJECT_REIDENTIFIED, timestamp=now,
                        source_ts=result.source_ts, frame_id=result.frame_id,
                        description=f"{det.class_name} #{tid} re-appeared "
                                    f"(same as earlier)",
                        track_ids=[tid], classes=[det.class_name],
                        position=det.center,
                    ), key=f"reid:{tid}")
                else:
                    self._known_identities.add(tid)
                    ev = self._maybe(Event(
                        type=EventType.NEW_OBJECT, timestamp=now,
                        source_ts=result.source_ts, frame_id=result.frame_id,
                        description=f"New {det.class_name} detected (track #{tid})",
                        track_ids=[tid], classes=[det.class_name], position=det.center,
                    ), key=f"new:{tid}")
                if ev:
                    events.append(ev)

            self._update_track(st, det, result.frame_id)
            events.extend(self._check_motion(st, det, result, now))

        # Departures: confirmed tracks that vanished this frame.
        gone = self._active_ids - seen_ids
        for tid in gone:
            st = self._tracks.get(tid)
            if not st:
                continue
            ev = self._handle_departure(tid, st, tracked, result, now)
            if ev:
                events.append(ev)

        # Crowd density.
        ev = self._check_crowd(tracked, result, now)
        if ev:
            events.append(ev)

        # Thermal signature (IR only, needs the raw frame).
        if result.modality == Modality.IR and frame_image is not None:
            ev = self._check_thermal(frame_image, result, now)
            if ev:
                events.append(ev)

        self._active_ids = seen_ids
        for e in events:
            self.event_log.append(e)
            logger.debug("EVENT {} @f{}: {}", e.type.value, e.frame_id, e.description)
        return events

    def reset(self) -> None:
        self._tracks.clear()
        self._active_ids.clear()
        self._known_identities.clear()
        self._cooldowns.clear()
        self._person_count_prev = 0
        self.event_log.clear()

    # -------------------------------------------------------------- helpers
    def _update_track(self, st: _TrackState, det: Detection, frame_id: int) -> None:
        st.centers.append(det.center)
        st.last_bbox = det.bbox
        st.class_name = det.class_name
        st.last_seen_frame = frame_id
        st.last_position = det.center

    def _check_motion(self, st: _TrackState, det: Detection,
                      result: FrameResult, now: float) -> List[Event]:
        events: List[Event] = []
        if len(st.centers) < 2:
            return events

        win = min(self.config.motion_window, len(st.centers))
        recent = list(st.centers)[-win:]
        speed = _avg_step(recent)
        tid = det.track_id  # type: ignore[assignment]

        # STOPPED detection.
        if speed < self.config.stop_speed_px:
            st.stationary_frames += 1
            if (st.stationary_frames >= self.config.stop_min_frames
                    and not st.stopped_reported):
                st.stopped_reported = True
                ev = self._maybe(Event(
                    type=EventType.STOPPED, timestamp=now, source_ts=result.source_ts,
                    frame_id=result.frame_id,
                    description=f"{det.class_name} #{tid} has stopped",
                    track_ids=[tid], classes=[det.class_name], position=det.center,
                ), key=f"stop:{tid}")
                if ev:
                    events.append(ev)
        else:
            st.stationary_frames = 0
            st.stopped_reported = False

        # DIRECTION_CHANGE detection: compare the heading of the first half of
        # the motion window against the second half. Comparing against the
        # immediately-previous frame would only ever see tiny per-frame deltas
        # and miss gradual turns; the split-window baseline captures the actual
        # change in travel direction.
        if speed >= self.config.direction_min_speed_px and len(recent) >= 4:
            mid = len(recent) // 2
            head_old = _heading(recent[0], recent[mid])
            head_new = _heading(recent[mid], recent[-1])
            # Require real motion in BOTH halves so a stop-and-go isn't a "turn".
            if (_avg_step(recent[:mid + 1]) >= self.config.direction_min_speed_px
                    and _avg_step(recent[mid:]) >= self.config.direction_min_speed_px):
                delta = _angle_diff(head_new, head_old)
                if delta >= self.config.direction_change_deg:
                    ev = self._maybe(Event(
                        type=EventType.DIRECTION_CHANGE, timestamp=now,
                        source_ts=result.source_ts, frame_id=result.frame_id,
                        description=f"{det.class_name} #{tid} changed direction "
                                    f"({delta:.0f}°)",
                        track_ids=[tid], classes=[det.class_name],
                        position=det.center, metadata={"delta_deg": delta},
                    ), key=f"dir:{tid}")
                    if ev:
                        events.append(ev)
            st.last_heading = head_new
        return events

    def _handle_departure(self, tid: int, st: _TrackState,
                          tracked: List[Detection], result: FrameResult,
                          now: float) -> Optional[Event]:
        # Person near a vehicle when they vanish -> "entered vehicle".
        if st.class_name == PERSON_CLASS and st.last_position is not None:
            for det in tracked:
                if det.class_name in VEHICLE_CLASSES:
                    dist = _point_box_distance(st.last_position, det.bbox)
                    if dist <= self.config.enter_vehicle_distance_px:
                        self._tracks.pop(tid, None)
                        return self._maybe(Event(
                            type=EventType.PERSON_ENTER_VEHICLE, timestamp=now,
                            source_ts=result.source_ts, frame_id=result.frame_id,
                            description=f"Person #{tid} appears to have entered "
                                        f"{det.class_name} #{det.track_id}",
                            track_ids=[tid] + ([det.track_id] if det.track_id else []),
                            classes=[PERSON_CLASS, det.class_name],
                            position=st.last_position,
                        ), key=f"enter:{tid}")
        # Generic departure.
        cls = st.class_name
        self._tracks.pop(tid, None)
        return self._maybe(Event(
            type=EventType.OBJECT_LEFT, timestamp=now, source_ts=result.source_ts,
            frame_id=result.frame_id,
            description=f"{cls} #{tid} left the scene",
            track_ids=[tid], classes=[cls],
        ), key=f"left:{tid}")

    def _check_crowd(self, tracked: List[Detection],
                     result: FrameResult, now: float) -> Optional[Event]:
        persons = sum(1 for d in tracked if d.class_name == PERSON_CLASS)
        prev = self._person_count_prev
        self._person_count_prev = persons
        if persons >= 5 and persons > prev:
            return self._maybe(Event(
                type=EventType.CROWD_FORMED, timestamp=now,
                source_ts=result.source_ts, frame_id=result.frame_id,
                description=f"Crowd forming: {persons} people present",
                classes=[PERSON_CLASS], metadata={"count": persons},
            ), key="crowd")
        return None

    def _check_thermal(self, image: np.ndarray, result: FrameResult,
                       now: float) -> Optional[Event]:
        import cv2

        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, self.config.thermal_hotspot_threshold, 255,
                                cv2.THRESH_BINARY)
        area = int(np.count_nonzero(mask))
        prev = self._thermal_prev_mask_area
        self._thermal_prev_mask_area = area
        # Fire on a *new* sizeable hotspot (growth beyond min area).
        if area >= self.config.thermal_min_area_px and area > prev * 1.5:
            ys, xs = np.nonzero(mask)
            pos = (float(xs.mean()), float(ys.mean())) if len(xs) else None
            return self._maybe(Event(
                type=EventType.THERMAL_SIGNATURE, timestamp=now,
                source_ts=result.source_ts, frame_id=result.frame_id,
                description=f"New thermal signature detected (~{area}px hot region)",
                position=pos, metadata={"area_px": area},
            ), key="thermal")
        return None

    def _maybe(self, event: Event, key: str) -> Optional[Event]:
        """Apply per-key cooldown de-duplication."""
        last = self._cooldowns.get(key, -1e9)
        if event.timestamp - last < self.config.cooldown_sec:
            return None
        self._cooldowns[key] = event.timestamp
        return event


# ----------------------------------------------------------------- geometry
def _avg_step(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    steps = [math.dist(points[i], points[i - 1]) for i in range(1, len(points))]
    return sum(steps) / len(steps)


def _heading(p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
    return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))


def _angle_diff(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return d if d <= 180.0 else 360.0 - d


def _point_box_distance(pt: Tuple[float, float],
                        box: Tuple[float, float, float, float]) -> float:
    x, y = pt
    x1, y1, x2, y2 = box
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return math.hypot(dx, dy)
