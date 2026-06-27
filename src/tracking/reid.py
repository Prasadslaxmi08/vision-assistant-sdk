"""Appearance-based re-identification (ReID) — stable identities across track breaks.

ByteTrack assigns a *fresh* ``track_id`` whenever a target is occluded for longer
than its lost-track buffer, leaves and re-enters the frame, or its detection
confidence flickers. Downstream that looks like a brand-new object, so the
:class:`~src.events.event_manager.EventManager` keeps announcing "New person
detected" for someone it has already seen.

This module closes that gap **without loading another model** (important on an
8 GB GPU): it derives a cheap appearance signature from the detection crop and
matches new raw tracks against a short-term gallery of recently-seen identities.
Each track is therefore handed a *stable global id* that survives ByteTrack id
churn. The raw ByteTrack id is preserved on ``Detection.raw_track_id`` for
debugging.

Signature (all OpenCV, no GPU, no extra deps):
  * EO  : upper/lower-body HSV colour histograms (captures shirt vs trousers).
  * IR  : intensity + gradient-magnitude histograms (colour is meaningless on
          thermal, so we lean on brightness profile and edge texture).
The signature is L2-normalised; identities are compared by cosine similarity.

Matching is gated by class (a person never matches a car), by recency
(``max_age_sec``), and optionally by last-known position (``max_distance_px``).
Identities currently held by a live track are never offered as candidates, so two
people on screen at once can never collapse into one id.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import cv2
import numpy as np

from src.config.settings import ReIDConfig
from src.utils.logger import logger
from src.utils.types import Detection, Modality


@dataclass
class _GalleryEntry:
    """One known identity and its rolling appearance signature."""
    global_id: int
    class_name: str
    embedding: Optional[np.ndarray]      # None until a usable crop is seen
    last_seen_frame: int
    last_seen_ts: float
    last_center: tuple
    last_bbox: tuple


def _l2(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec / n if n > 1e-6 else vec


class Reidentifier:
    """Maps churny ByteTrack ids onto stable, appearance-backed global ids."""

    def __init__(self, config: ReIDConfig):
        self.config = config
        self._gallery: Dict[int, _GalleryEntry] = {}
        self._track_to_global: Dict[int, int] = {}   # raw ByteTrack id -> global id
        self._track_last_seen: Dict[int, int] = {}    # raw id -> last frame seen
        self._next_global_id = 1
        self._reid_classes: Set[str] = set(config.classes)
        logger.info("ReID enabled (classes={}, sim>={:.2f}, gallery<={})",
                    sorted(self._reid_classes), config.similarity_threshold,
                    config.gallery_max_size)

    # ------------------------------------------------------------------ API
    def assign(self, detections: List[Detection], frame_image: np.ndarray,
               frame_id: int, ts: float,
               modality: Modality = Modality.EO) -> List[Detection]:
        """Rewrite ``track_id`` on each confirmed detection to a stable global id.

        Non-tracked detections (no ``track_id``) are left untouched. Returns the
        same list (mutated in place) for convenient chaining.
        """
        if not self.config.enabled:
            return detections

        tracked = [d for d in detections if d.track_id is not None]

        # Phase 1 — resolve raw tracks we already know. Identities held by a
        # live track *this frame* are "active" and are never offered as match
        # candidates (stops two on-screen targets collapsing into one id).
        active_globals: Set[int] = set()
        new_tracks: List[Detection] = []
        for det in tracked:
            raw = int(det.track_id)
            gid = self._track_to_global.get(raw)
            if gid is not None and gid in self._gallery:
                active_globals.add(gid)
            else:
                new_tracks.append(det)

        # Phase 2 — a raw id we haven't mapped yet: try to recognise it.
        for det in new_tracks:
            raw = int(det.track_id)
            emb = self._embed(frame_image, det.bbox, modality)
            gid = self._match(emb, det, ts, active_globals)
            if gid is not None:
                # This identity is now embodied by the new raw id; drop any
                # older raw mapping so a recycled ByteTrack id can't double-map.
                self._reclaim(gid)
            else:
                gid = self._mint(det, emb, frame_id, ts)
            self._track_to_global[raw] = gid
            active_globals.add(gid)

        # Phase 3 — rewrite ids and refresh gallery for every tracked detection.
        refresh = (self.config.embedding_refresh_frames > 0
                   and frame_id % self.config.embedding_refresh_frames == 0)
        for det in tracked:
            raw = int(det.track_id)
            gid = self._track_to_global[raw]
            entry = self._gallery[gid]
            if refresh or entry.embedding is None:
                emb = self._embed(frame_image, det.bbox, modality)
                if emb is not None:
                    entry.embedding = (emb if entry.embedding is None
                                       else _l2(0.7 * entry.embedding + 0.3 * emb))
            entry.last_seen_frame = frame_id
            entry.last_seen_ts = ts
            entry.last_center = det.center
            entry.last_bbox = det.bbox
            self._track_last_seen[raw] = frame_id
            det.raw_track_id = raw
            det.track_id = gid

        # Phase 4 — housekeeping: release stale raw mappings, prune the gallery.
        self._expire_raw(frame_id)
        self._prune(ts)
        return detections

    def reset(self) -> None:
        self._gallery.clear()
        self._track_to_global.clear()
        self._track_last_seen.clear()
        self._next_global_id = 1

    # -------------------------------------------------------------- matching
    def _match(self, emb: Optional[np.ndarray], det: Detection, ts: float,
               reserved: Set[int]) -> Optional[int]:
        """Best inactive gallery identity above the similarity threshold, or None."""
        if emb is None or det.class_name not in self._reid_classes:
            return None
        best_gid: Optional[int] = None
        best_sim = self.config.similarity_threshold
        for gid, e in self._gallery.items():
            if gid in reserved or e.embedding is None:
                continue
            if e.class_name != det.class_name:
                continue
            if (ts - e.last_seen_ts) > self.config.max_age_sec:
                continue
            if (self.config.max_distance_px > 0 and e.last_center
                    and math.dist(det.center, e.last_center) > self.config.max_distance_px):
                continue
            sim = float(np.dot(emb, e.embedding))
            if sim >= best_sim:
                best_sim, best_gid = sim, gid
        if best_gid is not None:
            logger.debug("ReID match: raw#{} -> global#{} (sim={:.2f})",
                         det.track_id, best_gid, best_sim)
        return best_gid

    def _reclaim(self, gid: int) -> None:
        """Forget any raw->global mapping for ``gid`` (it has a new raw owner)."""
        for raw in [r for r, g in self._track_to_global.items() if g == gid]:
            self._track_to_global.pop(raw, None)
            self._track_last_seen.pop(raw, None)

    def _mint(self, det: Detection, emb: Optional[np.ndarray],
              frame_id: int, ts: float) -> int:
        gid = self._next_global_id
        self._next_global_id += 1
        self._gallery[gid] = _GalleryEntry(
            global_id=gid, class_name=det.class_name, embedding=emb,
            last_seen_frame=frame_id, last_seen_ts=ts,
            last_center=det.center, last_bbox=det.bbox,
        )
        return gid

    # ------------------------------------------------------------- embedding
    def _embed(self, image: np.ndarray, bbox: tuple,
               modality: Modality) -> Optional[np.ndarray]:
        """Cheap appearance signature for a detection crop (L2-normalised)."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < self.config.min_crop_px or y2 - y1 < self.config.min_crop_px:
            return None
        crop = image[y1:y2, x1:x2]
        # Trim a horizontal margin to suppress background bleed at the box edges.
        ch, cw = crop.shape[:2]
        mx = int(cw * 0.15)
        if cw - 2 * mx > 4:
            crop = crop[:, mx:cw - mx]

        bins = self.config.hist_bins
        gray_modality = modality == Modality.IR or crop.ndim == 2

        if crop.ndim == 3 and not gray_modality:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            mid = hsv.shape[0] // 2  # split upper / lower body
            feats = []
            for band in (hsv[:mid], hsv[mid:]):
                if band.size == 0:
                    band = hsv
                feats.append(cv2.calcHist([band], [0], None, [bins], [0, 180]).flatten())
                feats.append(cv2.calcHist([band], [1], None, [bins], [0, 256]).flatten())
            feat = np.concatenate(feats)
        else:
            gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            inten = cv2.calcHist([gray], [0], None, [bins], [0, 256]).flatten()
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            mag = cv2.magnitude(gx, gy)
            grad, _ = np.histogram(mag, bins=bins, range=(0.0, 255.0))
            feat = np.concatenate([inten, grad.astype(np.float32)])

        return _l2(feat.astype(np.float32))

    # ----------------------------------------------------------- housekeeping
    def _expire_raw(self, frame_id: int) -> None:
        """Release raw->global mappings whose ByteTrack id has gone quiet.

        Kept alive for ``track_grace_frames`` so a track that ByteTrack re-emits
        with the *same* raw id (within its lost-track buffer) stays mapped with
        zero appearance search. Past that, the raw id may be recycled for a
        different object, so we drop the mapping and force a fresh match.
        """
        grace = self.config.track_grace_frames
        stale = [raw for raw, seen in self._track_last_seen.items()
                 if frame_id - seen > grace]
        for raw in stale:
            self._track_to_global.pop(raw, None)
            self._track_last_seen.pop(raw, None)

    def _prune(self, ts: float) -> None:
        reserved = set(self._track_to_global.values())
        # Drop identities that have aged out and aren't held by a live track.
        for gid in list(self._gallery):
            e = self._gallery[gid]
            if gid not in reserved and (ts - e.last_seen_ts) > self.config.max_age_sec:
                del self._gallery[gid]
        # Hard size cap: evict the oldest unreserved identities first.
        overflow = len(self._gallery) - self.config.gallery_max_size
        if overflow > 0:
            evictable = sorted(
                (e for e in self._gallery.values() if e.global_id not in reserved),
                key=lambda e: e.last_seen_ts,
            )
            for e in evictable[:overflow]:
                self._gallery.pop(e.global_id, None)
