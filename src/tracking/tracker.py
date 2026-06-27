"""Multi-object tracking with ByteTrack (via the ``supervision`` library).

Detection and tracking are intentionally decoupled: the detector emits a
``supervision.Detections`` container, and this module assigns persistent
``track_id`` values across frames using ByteTrack's two-stage association
(high- and low-confidence detections matched to Kalman-predicted tracks).
"""
from __future__ import annotations

from typing import List

from src.config.settings import TrackingConfig
from src.utils.logger import logger
from src.utils.types import Detection


class ObjectTracker:
    def __init__(self, config: TrackingConfig):
        import supervision as sv

        self.config = config
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.track_activation_threshold,
            lost_track_buffer=config.lost_track_buffer,
            minimum_matching_threshold=config.minimum_matching_threshold,
            frame_rate=config.frame_rate,
            minimum_consecutive_frames=config.minimum_consecutive_frames,
        )
        logger.info("ByteTrack initialised (buffer={}, fps={})",
                    config.lost_track_buffer, config.frame_rate)

    def update(self, sv_detections: "object", base_detections: List[Detection]) -> List[Detection]:
        """Assign track ids to detections for the current frame.

        ``sv_detections`` is the supervision container from the detector;
        ``base_detections`` is the parallel list of :class:`Detection`. We run
        ByteTrack on the container, then map the returned ``tracker_id`` back
        onto fresh :class:`Detection` objects (ByteTrack may reorder / drop).
        """
        tracked = self.tracker.update_with_detections(sv_detections)
        out: List[Detection] = []
        for xyxy, conf, cls_id, track_id in zip(
            tracked.xyxy, tracked.confidence, tracked.class_id, tracked.tracker_id
        ):
            class_id = int(cls_id)
            class_name = _lookup_name(base_detections, class_id)
            out.append(Detection(
                bbox=tuple(float(v) for v in xyxy),
                confidence=float(conf),
                class_id=class_id,
                class_name=class_name,
                track_id=int(track_id) if track_id is not None else None,
            ))
        return out

    def reset(self) -> None:
        """Clear all track state (call between independent clips/streams)."""
        self.tracker.reset()


def _lookup_name(detections: List[Detection], class_id: int) -> str:
    for d in detections:
        if d.class_id == class_id:
            return d.class_name
    return str(class_id)


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def merge_detections(
    base: List[Detection],
    tracked: List[Detection],
    iou_thresh: float = 0.5,
) -> List[Detection]:
    """Union of confirmed tracks and not-yet-confirmed raw detections.

    ByteTrack only emits a track once it has persisted ``minimum_consecutive_
    frames`` — so brand-new or briefly-seen objects carry no ``track_id`` yet.
    For a real-time *detection* display we still want to show them. This returns
    every confirmed track plus any raw detection that doesn't overlap one,
    giving immediate visual feedback without sacrificing stable track ids.
    """
    out = list(tracked)
    for d in base:
        if not any(_iou(d.bbox, t.bbox) >= iou_thresh for t in tracked):
            out.append(d)  # unconfirmed detection, no track_id
    return out
