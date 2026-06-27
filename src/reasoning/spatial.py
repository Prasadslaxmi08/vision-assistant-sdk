"""Spatial Reasoning Engine — object-to-object relationships.

Consumes the per-frame tracked detections (already carrying stable ReID ids) and
infers spatial relationships that single-object events can't express:

  * PROXIMITY        — two objects stay close for a sustained period
  * FOLLOWING        — one mover trails another along the same heading
  * GROUP            — three or more persons cluster together and persist
  * RESTRICTED_AREA  — an object enters an operator-defined keep-out zone

All inference is cheap geometry over short motion histories — no extra model, so
it runs inline with the real-time loop. Relationships must persist for a minimum
number of frames before they are emitted (noise suppression), and a per-relation
cooldown prevents re-spamming the same interaction. Emitted
:class:`Interaction` objects are persisted to mission memory and surfaced in the
intel feed / timeline.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from src.config.settings import SpatialConfig
from src.utils.logger import logger
# Interaction/InteractionType now live in the shared data-contract module
# (repo-split prep, doc 04); re-imported for back-compatible call sites.
from src.utils.types import Detection, Interaction, InteractionType

PERSON = "person"
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "train", "boat", "airplane"}


@dataclass
class _Group:
    members: Set[int]
    streak: int
    emitted: bool


class SpatialReasoner:
    def __init__(self, config: SpatialConfig):
        self.config = config
        self._hist: Dict[int, Deque[Tuple[float, float]]] = {}
        self._cls: Dict[int, str] = {}
        self._streaks: Dict[tuple, int] = {}
        self._emitted: Set[tuple] = set()
        self._last_emit: Dict[tuple, float] = {}
        self._groups: List[_Group] = []
        logger.info("Spatial reasoner enabled (proximity<{:.0f}px, group>={})",
                    config.proximity_distance_px, config.group_min_size)

    def reset(self) -> None:
        self._hist.clear()
        self._cls.clear()
        self._streaks.clear()
        self._emitted.clear()
        self._last_emit.clear()
        self._groups.clear()

    # ------------------------------------------------------------------ API
    def process(self, detections: List[Detection], frame_id: int,
                source_ts: float, frame_shape: Tuple[int, int],
                now: Optional[float] = None) -> List[Interaction]:
        if not self.config.enabled:
            return []
        now = now if now is not None else source_ts
        tracked = [d for d in detections if d.track_id is not None]
        for d in tracked:
            tid = int(d.track_id)
            self._hist.setdefault(tid, deque(maxlen=self.config.history_len)).append(d.center)
            self._cls[tid] = d.class_name
        present = {int(d.track_id) for d in tracked}
        self._prune(present)

        out: List[Interaction] = []
        active: Set[tuple] = set()
        out += self._proximity(tracked, frame_id, source_ts, now, active)
        out += self._following(tracked, frame_id, source_ts, now, active)
        out += self._restricted(tracked, frame_id, source_ts, frame_shape, now, active)
        out += self._grouping(tracked, frame_id, source_ts, now, active)

        # Break streaks for relations not seen this frame.
        for key in list(self._streaks):
            if key not in active:
                self._streaks.pop(key, None)
                self._emitted.discard(key)
        return out

    # ----------------------------------------------------------- detectors
    def _proximity(self, tracked, frame_id, source_ts, now, active) -> List[Interaction]:
        out = []
        thr = self.config.proximity_distance_px
        for i in range(len(tracked)):
            for j in range(i + 1, len(tracked)):
                a, b = tracked[i], tracked[j]
                # Skip vehicle-vehicle (rarely a meaningful "interaction").
                if a.class_name in VEHICLE_CLASSES and b.class_name in VEHICLE_CLASSES:
                    continue
                dist = math.dist(a.center, b.center)
                if dist > thr:
                    continue
                ida, idb = sorted((int(a.track_id), int(b.track_id)))
                key = ("PROX", ida, idb)
                active.add(key)
                if self._fire(key, self.config.proximity_min_frames, now):
                    ca = self._cls.get(ida, "object"); cb = self._cls.get(idb, "object")
                    out.append(Interaction(
                        type=InteractionType.PROXIMITY, timestamp=now,
                        source_ts=source_ts, frame_id=frame_id,
                        description=f"{ca} #{ida} and {cb} #{idb} are in close "
                                    f"proximity (~{dist:.0f}px)",
                        subject_id=ida, object_id=idb,
                        metadata={"distance_px": round(dist, 1)}))
        return out

    def _following(self, tracked, frame_id, source_ts, now, active) -> List[Interaction]:
        out = []
        persons = [d for d in tracked if d.class_name == PERSON]
        for a in persons:
            for b in persons:
                if a is b:
                    continue
                ida, idb = int(a.track_id), int(b.track_id)
                if not self._is_following(ida, idb):
                    continue
                key = ("FOLLOW", ida, idb)
                active.add(key)
                if self._fire(key, self.config.following_min_frames, now):
                    out.append(Interaction(
                        type=InteractionType.FOLLOWING, timestamp=now,
                        source_ts=source_ts, frame_id=frame_id,
                        description=f"person #{ida} appears to be following person #{idb}",
                        subject_id=ida, object_id=idb,
                        metadata={"lag_px": round(math.dist(a.center, b.center), 1)}))
        return out

    def _restricted(self, tracked, frame_id, source_ts, frame_shape, now, active) -> List[Interaction]:
        out = []
        zones = self.config.restricted_zones or []
        if not zones:
            return out
        h, w = frame_shape[:2]
        for d in tracked:
            cx, cy = d.center
            for zi, z in enumerate(zones):
                zx1, zy1, zx2, zy2 = z[0] * w, z[1] * h, z[2] * w, z[3] * h
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    tid = int(d.track_id)
                    key = ("ZONE", tid, zi)
                    active.add(key)
                    if self._fire(key, self.config.restricted_min_frames, now):
                        out.append(Interaction(
                            type=InteractionType.RESTRICTED_AREA, timestamp=now,
                            source_ts=source_ts, frame_id=frame_id,
                            description=f"{d.class_name} #{tid} entered restricted "
                                        f"zone {zi + 1}",
                            subject_id=tid, object_id=None,
                            metadata={"zone": zi + 1}))
        return out

    def _grouping(self, tracked, frame_id, source_ts, now, active) -> List[Interaction]:
        persons = [d for d in tracked if d.class_name == PERSON]
        clusters = self._cluster(persons, self.config.group_radius_px)
        clusters = [c for c in clusters if len(c) >= self.config.group_min_size]

        out: List[Interaction] = []
        new_groups: List[_Group] = []
        for members in clusters:
            prev = self._match_group(members)
            if prev is not None:
                grp = _Group(members=members, streak=prev.streak + 1,
                             emitted=prev.emitted)
            else:
                grp = _Group(members=members, streak=1, emitted=False)
            if not grp.emitted and grp.streak >= self.config.group_min_frames:
                grp.emitted = True
                ids = sorted(members)
                out.append(Interaction(
                    type=InteractionType.GROUP, timestamp=now, source_ts=source_ts,
                    frame_id=frame_id,
                    description=f"Group of {len(ids)} persons formed "
                                f"(#{', #'.join(map(str, ids))})",
                    subject_id=ids[0], object_id=None, members=ids,
                    metadata={"size": len(ids)}))
            new_groups.append(grp)
        self._groups = new_groups
        return out

    # --------------------------------------------------------------- helpers
    def _is_following(self, a: int, b: int) -> bool:
        ha, hb = self._hist.get(a), self._hist.get(b)
        if not ha or not hb or len(ha) < 5 or len(hb) < 5:
            return False
        va, vb = _velocity(ha), _velocity(hb)
        sa, sb = math.hypot(*va), math.hypot(*vb)
        m = self.config.following_min_speed_px
        if sa < m or sb < m:
            return False
        if (va[0] * vb[0] + va[1] * vb[1]) / (sa * sb) < 0.82:  # ~35° heading
            return False
        ax, ay = ha[-1]; bx, by = hb[-1]
        ab = (bx - ax, by - ay)
        dist = math.hypot(*ab)
        if dist < 10 or dist > self.config.following_max_lag_px:
            return False
        # Leader must be ahead of the follower along the leader's travel direction.
        return (ab[0] * vb[0] + ab[1] * vb[1]) / (dist * sb) >= 0.7

    @staticmethod
    def _cluster(persons, radius) -> List[Set[int]]:
        """Single-linkage clustering of person centers within ``radius``."""
        ids = [int(p.track_id) for p in persons]
        parent = {i: i for i in ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                if math.dist(persons[i].center, persons[j].center) <= radius:
                    parent[find(ids[i])] = find(ids[j])
        groups: Dict[int, Set[int]] = {}
        for i in ids:
            groups.setdefault(find(i), set()).add(i)
        return list(groups.values())

    def _match_group(self, members: Set[int]) -> Optional[_Group]:
        best, best_j = None, 0.5  # require Jaccard >= 0.5 to be "the same" group
        for g in self._groups:
            inter = len(members & g.members)
            union = len(members | g.members)
            j = inter / union if union else 0.0
            if j >= best_j:
                best, best_j = g, j
        return best

    def _fire(self, key: tuple, min_frames: int, now: float) -> bool:
        """Advance a relation's streak; emit once it persists, honouring cooldown."""
        self._streaks[key] = self._streaks.get(key, 0) + 1
        if key in self._emitted or self._streaks[key] < min_frames:
            return False
        if now - self._last_emit.get(key, -1e9) < self.config.cooldown_sec:
            return False
        self._emitted.add(key)
        self._last_emit[key] = now
        return True

    def _prune(self, present: Set[int]) -> None:
        for tid in list(self._hist):
            if tid not in present:
                self._hist.pop(tid, None)
                self._cls.pop(tid, None)


def _velocity(hist: Deque[Tuple[float, float]]) -> Tuple[float, float]:
    """Mean per-step velocity over the recent history window."""
    pts = list(hist)[-6:]
    if len(pts) < 2:
        return (0.0, 0.0)
    dx = (pts[-1][0] - pts[0][0]) / (len(pts) - 1)
    dy = (pts[-1][1] - pts[0][1]) / (len(pts) - 1)
    return (dx, dy)
