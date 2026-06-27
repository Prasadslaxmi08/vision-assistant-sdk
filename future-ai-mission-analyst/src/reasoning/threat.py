"""Threat & Anomaly Scoring — synthesize signals into risk levels.

This layer doesn't run new detectors; it *fuses the evidence* the rest of the
pipeline already produces — events, spatial interactions, EO/IR fusion
assessments, plus dwell/displacement — into an explainable per-object threat
score, and rolls those up into a mission-level risk.

Each contributing signal adds a weighted increment with a human-readable factor
label (so a score is always explainable: "loitering + restricted area"). Scores
**decay** over time, so threat reflects *recent* behaviour and fades when an
actor stops being suspicious. When an object (or the mission) crosses into a
higher band, a ``THREAT_ESCALATION`` event is emitted — surfaced in the feed /
timeline and used to prioritise the VLM.

Bands: NONE < LOW < MEDIUM < HIGH < CRITICAL (thresholds are configurable).
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.config.settings import ThreatConfig
from src.events.event_manager import Event, EventType
from src.utils.logger import logger

_LEVELS = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
_RANK = {lvl: i for i, lvl in enumerate(_LEVELS)}


@dataclass
class _State:
    class_name: str
    first_ts: float
    last_ts: float
    anchor: Tuple[float, float]
    score: float = 0.0
    level: str = "NONE"
    max_disp: float = 0.0
    last_loiter_ts: float = -1e9
    factors: Counter = field(default_factory=Counter)


class ThreatScorer:
    def __init__(self, config: ThreatConfig):
        self.config = config
        self._states: Dict[int, _State] = {}
        self._mission_score = 0.0
        self._mission_level = "NONE"
        self._mission_factors: Counter = Counter()
        self._last_ts: Optional[float] = None
        logger.info("Threat scorer enabled (levels @ {}/{}/{}/{})",
                    config.level_low, config.level_medium, config.level_high,
                    config.level_critical)

    def reset(self) -> None:
        self._states.clear()
        self._mission_score = 0.0
        self._mission_level = "NONE"
        self._mission_factors.clear()
        self._last_ts = None

    # ------------------------------------------------------------------ API
    def assess(self, frame_id: int, source_ts: float, now: float, tracked,
               events: List[Event], interactions, fusions
               ) -> Tuple[List[Event], Dict[int, Tuple[float, str]]]:
        """Update threat from this frame's signals. Returns (escalation events,
        {global_id: (score, level)} updates to persist)."""
        if not self.config.enabled:
            return [], {}

        dt = 0.0 if self._last_ts is None else max(0.0, now - self._last_ts)
        self._last_ts = now
        decay = self.config.decay_per_sec ** dt if dt else 1.0

        # Decay existing scores first (recency weighting).
        for st in self._states.values():
            st.score *= decay
        self._mission_score *= decay

        present: Set[int] = set()
        for d in tracked:
            gid = int(d.track_id)
            present.add(gid)
            self._track_dwell(gid, d.class_name, now, d.center)

        for e in events:
            w = self._event_weight(e.type)
            gid = e.track_ids[0] if e.track_ids else None
            if w and gid is not None:
                self._add(int(gid), w, _pretty(e.type.value), now)
            if e.type == EventType.CROWD_FORMED:
                self._mission_add(self.config.w_crowd, "crowd")

        for it in interactions:
            t = it.type.value
            if t == "FOLLOWING":
                self._add(it.subject_id, self.config.w_following, "following", now)
            elif t == "RESTRICTED_AREA":
                self._add(it.subject_id, self.config.w_restricted,
                          "restricted area", now)
            elif t == "GROUP":
                for m in (it.members or []):
                    self._add(m, self.config.w_group_member, "grouping", now)
            elif t == "PROXIMITY":
                self._add(it.subject_id, self.config.w_proximity, "proximity", now)
                if it.object_id is not None:
                    self._add(it.object_id, self.config.w_proximity, "proximity", now)

        for fa in fusions:
            t = fa.type.value
            if t == "ACTIVE_VEHICLE" and fa.global_id is not None:
                self._add(fa.global_id, self.config.w_active_vehicle,
                          "active vehicle", now)
            elif t == "THERMAL_CONFIRMED" and fa.global_id is not None:
                self._add(fa.global_id, self.config.w_thermal_confirmed,
                          "thermal", now)
            elif t == "CONCEALED_HEAT":
                self._mission_add(self.config.w_concealed_heat, "concealed heat")

        alerts, updates = self._finalize(frame_id, source_ts, now, present)
        self._prune(now)
        return alerts, updates

    # --------------------------------------------------------- public reads
    def mission_threat(self) -> dict:
        top_obj = sorted(self._states.items(), key=lambda kv: kv[1].score,
                         reverse=True)
        return {
            "level": self._mission_level,
            "score": round(self._mission_score, 2),
            "factors": [f for f, _ in self._mission_factors.most_common(4)],
            "top_objects": [
                {"global_id": gid, "class_name": st.class_name,
                 "score": round(st.score, 2), "level": st.level,
                 "factors": [f for f, _ in st.factors.most_common(3)]}
                for gid, st in top_obj[:5] if st.score > 0.1],
        }

    def object_threat(self, gid: int) -> Optional[dict]:
        st = self._states.get(gid)
        if st is None:
            return None
        return {"global_id": gid, "score": round(st.score, 2), "level": st.level,
                "factors": [f for f, _ in st.factors.most_common(4)]}

    # --------------------------------------------------------------- helpers
    def _track_dwell(self, gid: int, cls: str, now: float, center) -> None:
        st = self._states.get(gid)
        if st is None:
            self._states[gid] = _State(class_name=cls, first_ts=now,
                                       last_ts=now, anchor=center)
            return
        st.last_ts = now
        st.class_name = cls or st.class_name
        disp = math.dist(center, st.anchor)
        st.max_disp = max(st.max_disp, disp)
        if disp > self.config.loiter_displacement_px * 1.5:
            # Subject moved on — reset the loiter anchor/timer.
            st.anchor = center
            st.first_ts = now
            st.max_disp = 0.0
            return
        # Stationary long enough → loitering (re-add periodically to escalate).
        if (disp <= self.config.loiter_displacement_px
                and now - st.first_ts >= self.config.loiter_seconds
                and now - st.last_loiter_ts >= self.config.loiter_seconds):
            self._add(gid, self.config.w_loiter, "loitering", now)
            st.last_loiter_ts = now

    def _add(self, gid: int, weight: float, factor: str, now: float) -> None:
        st = self._states.get(gid)
        if st is None:
            st = _State(class_name="object", first_ts=now, last_ts=now,
                        anchor=(0.0, 0.0))
            self._states[gid] = st
        st.score += weight
        st.factors[factor] += weight

    def _mission_add(self, weight: float, factor: str) -> None:
        self._mission_score += weight
        self._mission_factors[factor] += weight

    def _event_weight(self, etype: EventType) -> float:
        return {
            EventType.STOPPED: self.config.w_stopped,
            EventType.DIRECTION_CHANGE: self.config.w_direction_change,
            EventType.PERSON_ENTER_VEHICLE: self.config.w_enter_vehicle,
        }.get(etype, 0.0)

    def _level(self, score: float) -> str:
        c = self.config
        if score >= c.level_critical:
            return "CRITICAL"
        if score >= c.level_high:
            return "HIGH"
        if score >= c.level_medium:
            return "MEDIUM"
        if score >= c.level_low:
            return "LOW"
        return "NONE"

    def _finalize(self, frame_id, source_ts, now, present
                  ) -> Tuple[List[Event], Dict[int, Tuple[float, str]]]:
        alerts: List[Event] = []
        updates: Dict[int, Tuple[float, str]] = {}
        for gid, st in self._states.items():
            new_level = self._level(st.score)
            escalated = _RANK[new_level] > _RANK[st.level]
            st.level = new_level
            if gid in present or escalated:
                updates[gid] = (round(st.score, 2), new_level)
            if escalated and new_level != "NONE":
                factors = ", ".join(f for f, _ in st.factors.most_common(3))
                alerts.append(Event(
                    type=EventType.THREAT_ESCALATION, timestamp=now,
                    source_ts=source_ts, frame_id=frame_id,
                    description=f"{st.class_name} #{gid} threat level raised to "
                                f"{new_level} ({factors})",
                    track_ids=[gid], classes=[st.class_name],
                    metadata={"score": round(st.score, 2), "level": new_level,
                              "factors": factors}))

        # Mission-level roll-up: max object score blended with mission anomalies.
        max_obj = max((st.score for st in self._states.values()), default=0.0)
        rolled = max(self._mission_score, max_obj)
        new_mlevel = self._level(rolled)
        if _RANK[new_mlevel] > _RANK[self._mission_level] and new_mlevel != "NONE":
            factors = ", ".join(f for f, _ in self._mission_factors.most_common(3)) \
                or "elevated object activity"
            alerts.append(Event(
                type=EventType.THREAT_ESCALATION, timestamp=now,
                source_ts=source_ts, frame_id=frame_id,
                description=f"Mission threat level raised to {new_mlevel} ({factors})",
                track_ids=[], classes=[],
                metadata={"scope": "mission", "level": new_mlevel}))
        self._mission_level = new_mlevel
        return alerts, updates

    def _prune(self, now: float) -> None:
        ttl = self.config.state_ttl_sec
        for gid in [g for g, s in self._states.items()
                    if now - s.last_ts > ttl and s.score < 0.1]:
            self._states.pop(gid, None)


def _pretty(event_type: str) -> str:
    return event_type.replace("_", " ").lower()
