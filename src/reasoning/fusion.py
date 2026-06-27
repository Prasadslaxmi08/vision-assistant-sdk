"""EO/IR Fusion Engine — cross-sensor thermal↔visual correlation.

The EO and IR sensors are **independent optics** (different field of view, zoom,
and alignment). They are *not* the same image, so fusion correlates a thermal
signature seen by the IR camera with a visual target seen by the EO camera only
after the two frames have been spatially **registered** (see
:mod:`src.reasoning.registration`, which estimates the IR→EO mapping online from
shared targets — no calibration). Every IR hotspot is projected into the EO frame
through that mapping before correlation.

Conclusions (each carries the registration confidence):
  * ACTIVE_VEHICLE     — IR heat lands on an EO-tracked vehicle ⇒ running/recent engine.
  * THERMAL_CONFIRMED  — IR heat lands on an EO-tracked person/object ⇒ live warm target.
  * CONCEALED_HEAT     — IR heat maps *inside* the EO field of view but onto no EO
                         detection ⇒ a target hidden/occluded in the visible band.
  * CONTACT_OUTSIDE_EO — IR heat maps *outside* the EO field of view ⇒ a thermal
                         contact the EO camera isn't covering (cue to slew/zoom EO).
                         This output only exists because the sensors are decoupled.

Thermal extraction is **adaptive** for 8-bit AGC IR: there is no fixed radiometric
scale (brightness drifts with auto-gain), so "hot" is a per-frame percentile, not
a constant grey value.

For a lone IR stream (no EO), :meth:`process` still performs intra-frame
thermal-to-detection analysis — the original single-sensor behaviour, which is
valid within one image.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np

from src.config.settings import FusionConfig
from src.reasoning.registration import KeyPoint, Registration
from src.utils.logger import logger
# FusionType/FusionAssessment now live in the shared data-contract module
# (repo-split prep, doc 04); re-imported for back-compatible call sites.
from src.utils.types import Detection, FusionAssessment, FusionType, Modality

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "train", "boat", "airplane"}


class FusionEngine:
    def __init__(self, config: FusionConfig):
        self.config = config
        self._streaks: Dict[tuple, int] = {}
        self._emitted: Set[tuple] = set()
        self._last_emit: Dict[tuple, float] = {}
        logger.info("EO/IR fusion enabled (adaptive={}, white_hot={})",
                    config.adaptive, config.white_hot)

    def reset(self) -> None:
        self._streaks.clear()
        self._emitted.clear()
        self._last_emit.clear()

    # ===================================================== cross-sensor fusion
    def fuse_pair(self, eo_detections: List[Detection],
                  blobs: Sequence[tuple], registration: Registration,
                  frame_id: int, source_ts: float,
                  now: Optional[float] = None) -> List[FusionAssessment]:
        """Correlate IR hot blobs (already extracted, in IR coords) with EO
        detections after projecting each blob into the EO frame via
        ``registration``. Requires a usable registration."""
        if not self.config.enabled or registration is None or not registration.usable:
            return []
        now = now if now is not None else source_ts
        conf = round(registration.confidence, 2)
        eo_tracked = [d for d in eo_detections if d.track_id is not None]

        out: List[FusionAssessment] = []
        active: Set[tuple] = set()
        for (bx1, by1, bx2, by2, area, cx, cy) in blobs:
            eo_box = registration.map_box((bx1, by1, bx2, by2))
            ecx, ecy = registration.map_point(cx, cy)
            if eo_box is None or ecx is None:
                continue

            # Heat that maps outside the EO field of view — EO isn't covering it.
            if not registration.in_eo_view(ecx, ecy, margin=0.02 * registration.eo_size[0]):
                key = ("OUT", int(ecx // self.config.concealed_cell_px),
                       int(ecy // self.config.concealed_cell_px))
                active.add(key)
                if self._fire(key, now):
                    out.append(FusionAssessment(
                        type=FusionType.CONTACT_OUTSIDE_EO, timestamp=now,
                        source_ts=source_ts, frame_id=frame_id, area_px=area,
                        position=(ecx, ecy), confidence=conf,
                        description="Thermal contact outside EO coverage — "
                                    "recommend slewing/zooming EO to investigate",
                        metadata={"area_px": area, "registration": registration.state}))
                continue

            det, frac = self._best_overlap(eo_box, eo_tracked)
            if det is not None and frac >= self.config.correlate_overlap:
                gid = int(det.track_id)
                if det.class_name in VEHICLE_CLASSES:
                    key, ftype = ("AV", gid), FusionType.ACTIVE_VEHICLE
                    desc = (f"{det.class_name} #{gid} shows a strong IR signature "
                            f"on the EO target — likely active (running/recent engine)")
                else:
                    key, ftype = ("TC", gid), FusionType.THERMAL_CONFIRMED
                    desc = (f"{det.class_name} #{gid} confirmed by IR signature "
                            f"(warm/active target)")
                active.add(key)
                if self._fire(key, now):
                    out.append(FusionAssessment(
                        type=ftype, timestamp=now, source_ts=source_ts,
                        frame_id=frame_id, global_id=gid, area_px=area,
                        position=(ecx, ecy), confidence=conf, description=desc,
                        metadata={"area_px": area, "overlap": round(frac, 2),
                                  "registration": registration.state}))
            elif area >= self.config.concealed_min_area_px:
                key = ("CH", int(ecx // self.config.concealed_cell_px),
                       int(ecy // self.config.concealed_cell_px))
                active.add(key)
                if self._fire(key, now):
                    out.append(FusionAssessment(
                        type=FusionType.CONCEALED_HEAT, timestamp=now,
                        source_ts=source_ts, frame_id=frame_id, area_px=area,
                        position=(ecx, ecy), confidence=conf,
                        description=f"Thermal signature (~{area}px) inside EO view "
                                    f"with no visual detection — possible concealed "
                                    f"or occluded target",
                        metadata={"area_px": area, "registration": registration.state}))
        self._expire(active)
        return out

    # ================================================== single-IR-stream mode
    def process(self, image: np.ndarray, detections: List[Detection],
                frame_id: int, source_ts: float,
                modality: Modality = Modality.EO,
                now: Optional[float] = None) -> List[FusionAssessment]:
        """Intra-frame thermal analysis for a lone IR stream (no EO to fuse).

        Hot blobs are correlated to detections within the *same* IR image — valid
        only because there is a single sensor; this is NOT cross-sensor fusion.
        """
        if not self.config.enabled or modality != Modality.IR or image is None:
            return []
        now = now if now is not None else source_ts
        blobs = self.hot_blobs(image)
        if not blobs:
            return []
        tracked = [d for d in detections if d.track_id is not None]

        out: List[FusionAssessment] = []
        active: Set[tuple] = set()
        for (bx1, by1, bx2, by2, area, cx, cy) in blobs:
            det, frac = self._best_overlap((bx1, by1, bx2, by2), tracked)
            if det is not None and frac >= self.config.correlate_overlap:
                gid = int(det.track_id)
                if det.class_name in VEHICLE_CLASSES:
                    key, ftype = ("AV", gid), FusionType.ACTIVE_VEHICLE
                    desc = (f"{det.class_name} #{gid} shows a strong thermal "
                            f"signature — likely active (running/recent engine)")
                else:
                    key, ftype = ("TC", gid), FusionType.THERMAL_CONFIRMED
                    desc = (f"{det.class_name} #{gid} confirmed by thermal "
                            f"signature (warm/active target)")
                active.add(key)
                if self._fire(key, now):
                    out.append(FusionAssessment(
                        type=ftype, timestamp=now, source_ts=source_ts,
                        frame_id=frame_id, global_id=gid, area_px=area,
                        position=(cx, cy), description=desc,
                        metadata={"area_px": area, "overlap": round(frac, 2)}))
            elif area >= self.config.concealed_min_area_px:
                key = ("CH", int(cx // self.config.concealed_cell_px),
                       int(cy // self.config.concealed_cell_px))
                active.add(key)
                if self._fire(key, now):
                    out.append(FusionAssessment(
                        type=FusionType.CONCEALED_HEAT, timestamp=now,
                        source_ts=source_ts, frame_id=frame_id, area_px=area,
                        position=(cx, cy),
                        description=f"Unexplained thermal signature (~{area}px) with "
                                    f"no detection — possible concealed/occluded target",
                        metadata={"area_px": area}))
        self._expire(active)
        return out

    # --------------------------------------------------------------- helpers
    def hot_blobs(self, image: np.ndarray):
        """Hot connected components as (x1,y1,x2,y2,area,cx,cy), largest first.

        Adaptive (default) for 8-bit AGC IR: the threshold is a per-frame
        percentile so it tracks auto-gain instead of assuming a fixed grey level.

        The percentile + connected-components pass — a per-frame spike source on a
        large IR frame — runs on a **downscaled** image when wider than
        ``blob_downscale_width``; blob boxes, centroids and areas are scaled back
        to full resolution on the way out, so coordinates remain EO-frame-correct
        for the registration/projection (latency roadmap M5/B6).
        """
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not self.config.white_hot:
            gray = 255 - gray  # black-hot: invert so "hot" is bright
        scale = 1.0
        dw = self.config.blob_downscale_width
        if dw and gray.shape[1] > dw:
            scale = gray.shape[1] / float(dw)
            new_h = max(1, int(round(gray.shape[0] / scale)))
            gray = cv2.resize(gray, (dw, new_h), interpolation=cv2.INTER_AREA)
        if self.config.adaptive:
            pct = float(np.percentile(gray, self.config.hot_percentile))
            thr = max(self.config.min_hot_threshold, pct)
        else:
            thr = self.config.hot_threshold
        _, mask = cv2.threshold(gray, float(thr), 255, cv2.THRESH_BINARY)
        n, _, stats, cents = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        area_scale = scale * scale
        blobs = []
        for i in range(1, n):
            # Compare against the full-res area floor (areas are in downscaled px).
            area = int(round(int(stats[i, cv2.CC_STAT_AREA]) * area_scale))
            if area < self.config.min_blob_area_px:
                continue
            x = stats[i, cv2.CC_STAT_LEFT] * scale; y = stats[i, cv2.CC_STAT_TOP] * scale
            w = stats[i, cv2.CC_STAT_WIDTH] * scale; h = stats[i, cv2.CC_STAT_HEIGHT] * scale
            cx, cy = float(cents[i][0]) * scale, float(cents[i][1]) * scale
            blobs.append((int(x), int(y), int(x + w), int(y + h), area, cx, cy))
        blobs.sort(key=lambda b: b[4], reverse=True)
        return blobs[: self.config.max_blobs]

    @staticmethod
    def _best_overlap(blob_box, tracked):
        """Detection whose box best covers the blob box, with covered fraction."""
        bx1, by1, bx2, by2 = blob_box
        blob_area = max(1.0, (bx2 - bx1) * (by2 - by1))
        best, best_frac = None, 0.0
        for d in tracked:
            x1, y1, x2, y2 = d.bbox
            ix1, iy1 = max(bx1, x1), max(by1, y1)
            ix2, iy2 = min(bx2, x2), min(by2, y2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            frac = (iw * ih) / blob_area
            if frac > best_frac:
                best, best_frac = d, frac
        return best, best_frac

    def _fire(self, key: tuple, now: float) -> bool:
        self._streaks[key] = self._streaks.get(key, 0) + 1
        if key in self._emitted or self._streaks[key] < self.config.min_frames:
            return False
        if now - self._last_emit.get(key, -1e9) < self.config.cooldown_sec:
            return False
        self._emitted.add(key)
        self._last_emit[key] = now
        return True

    def _expire(self, active: Set[tuple]) -> None:
        for key in list(self._streaks):
            if key not in active:
                self._streaks.pop(key, None)
                self._emitted.discard(key)


# --------------------------------------------------------- registration inputs
def detection_keypoints(detections: List[Detection]) -> List[KeyPoint]:
    """EO/IR detections → registration control points (tracked only).

    Weight = confidence·area: confident, larger targets give the most reliably
    localised centroids, so they survive the RANSAC point cap (M5/B5)."""
    pts = []
    for d in detections:
        if d.track_id is not None:
            cx, cy = d.center
            pts.append(KeyPoint(cx, cy, d.class_name, "det",
                                weight=float(d.confidence) * d.area))
    return pts


def blob_keypoints(blobs: Sequence[tuple]) -> List[KeyPoint]:
    """IR hot blobs → class-agnostic registration control points (weight = area)."""
    return [KeyPoint(cx, cy, None, "blob", weight=float(area))
            for (*_, area, cx, cy) in blobs]
