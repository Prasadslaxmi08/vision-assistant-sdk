"""Sliced (tiled) inference for small-object recall.

Wraps any :class:`~src.detection.base.Detector` and runs it over overlapping tiles
of a high-resolution frame, then merges the tile detections with the full-frame
pass using class-aware NMS. This recovers small / distant / low-contrast targets
(e.g. a far-off boat in a thermal image) that vanish when the whole frame is
downsampled to the detector's native resolution.

It honours the exact same ``detect(image) -> (list[Detection], sv.Detections)``
contract, so the tracker / fusion / analyzers consume it unchanged. Tiling is
~``(tiles + 1)`` detector calls per image, so it is meant for still-image analysis
— the real-time engine uses the bare detector.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.config.settings import TilingConfig
from src.utils.logger import logger
from src.utils.types import Detection


class SlicedDetector:
    def __init__(self, base, cfg: TilingConfig):
        self.base = base
        self.cfg = cfg
        logger.info("Tiled inference enabled (tile={}, overlap={}, min_side={})",
                    cfg.tile, cfg.overlap, cfg.min_side)

    # --- Detector protocol passthrough -----------------------------------
    @property
    def name(self) -> str:
        return f"{self.base.name}+tiled"

    @property
    def classes(self) -> Optional[List[int]]:
        return self.base.classes

    def warmup(self, size: Optional[int] = None) -> None:
        self.base.warmup(size)

    # --- tiling ----------------------------------------------------------
    def _tile_origins(self, w: int, h: int) -> List[Tuple[int, int, int, int]]:
        """(x0, y0, x1, y1) tile boxes covering the image with overlap."""
        t = self.cfg.tile
        step = max(1, int(t * (1.0 - self.cfg.overlap)))
        xs = list(range(0, max(1, w - t + 1), step)) or [0]
        ys = list(range(0, max(1, h - t + 1), step)) or [0]
        if xs[-1] + t < w:
            xs.append(w - t)
        if ys[-1] + t < h:
            ys.append(h - t)
        boxes = []
        for y0 in ys:
            for x0 in xs:
                x0c, y0c = max(0, x0), max(0, y0)
                boxes.append((x0c, y0c, min(w, x0c + t), min(h, y0c + t)))
        return boxes[: self.cfg.max_tiles]

    def detect(self, image: np.ndarray) -> Tuple[List[Detection], "object"]:
        import supervision as sv

        h, w = image.shape[:2]
        merged: List[Detection] = []
        # Full-frame pass — best for large objects + global context.
        full, _ = self.base.detect(image)
        merged.extend(full)

        # Tiled passes — only worthwhile on large frames.
        if max(h, w) >= self.cfg.min_side:
            for (x0, y0, x1, y1) in self._tile_origins(w, h):
                tile = image[y0:y1, x0:x1]
                if tile.size == 0:
                    continue
                dets, _ = self.base.detect(tile)
                for d in dets:
                    bx1, by1, bx2, by2 = d.bbox
                    merged.append(Detection(
                        bbox=(bx1 + x0, by1 + y0, bx2 + x0, by2 + y0),
                        confidence=d.confidence, class_id=d.class_id,
                        class_name=d.class_name))

        merged = _class_aware_nms(merged, self.cfg.iou_merge)
        return merged, _to_sv(merged, sv)


# --------------------------------------------------------------- helpers
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
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _class_aware_nms(dets: List[Detection], iou_thresh: float) -> List[Detection]:
    """Greedy per-class NMS: keep the highest-confidence box, drop overlaps."""
    kept: List[Detection] = []
    for cid in {d.class_id for d in dets}:
        group = sorted([d for d in dets if d.class_id == cid],
                       key=lambda d: d.confidence, reverse=True)
        survivors: List[Detection] = []
        for d in group:
            if all(_iou(d.bbox, s.bbox) < iou_thresh for s in survivors):
                survivors.append(d)
        kept.extend(survivors)
    kept.sort(key=lambda d: d.confidence, reverse=True)
    return kept


def _to_sv(dets: List[Detection], sv):
    """Build a ``supervision.Detections`` from merged detections (for the tracker)."""
    if not dets:
        return sv.Detections.empty()
    xyxy = np.array([d.bbox for d in dets], dtype=float)
    conf = np.array([d.confidence for d in dets], dtype=float)
    cls = np.array([d.class_id for d in dets], dtype=int)
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)
