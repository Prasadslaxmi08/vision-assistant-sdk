"""YOLOv11 object detector (Ultralytics).

Thin, configuration-driven wrapper that returns framework-agnostic
:class:`Detection` objects and a ``supervision.Detections`` container so the
tracker can consume results without re-coupling to Ultralytics internals.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from src.config.settings import DetectionConfig
from src.utils.logger import logger
from src.utils.types import Detection


class YOLODetector:
    def __init__(self, config: DetectionConfig, device: str = "cuda:0", half: bool = True):
        # Imported lazily so the rest of the app (and the UI) can load even if
        # heavy CV/ML wheels aren't installed yet.
        from ultralytics import YOLO

        self.config = config
        self.device = device
        self.half = half and device.startswith("cuda")
        logger.info("Loading YOLO model: {}", config.model_path)
        self.model = YOLO(config.model_path)
        try:
            self.model.to(device)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not move YOLO to {} ({}); using default device", device, exc)
            self.device = "cpu"
            self.half = False
        self.class_names = self.model.names
        logger.info("YOLO ready on {} (half={}, {} classes)",
                    self.device, self.half, len(self.class_names))

    # --- Detector protocol (src/detection/base.py) -----------------------
    @property
    def name(self) -> str:
        return Path(self.config.model_path).stem        # e.g. "yolo11m"

    @property
    def classes(self) -> Optional[List[int]]:
        return self.config.classes

    def detect(self, image: np.ndarray) -> Tuple[List[Detection], "object"]:
        """Run detection on a BGR frame.

        Returns ``(detections, sv_detections)`` where ``sv_detections`` is a
        ``supervision.Detections`` ready for the tracker. Track ids are filled
        in later by :class:`~src.tracking.tracker.ObjectTracker`.
        """
        import supervision as sv

        results = self.model.predict(
            source=image,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.img_size,
            classes=self.config.classes,
            max_det=self.config.max_detections,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

        sv_det = sv.Detections.from_ultralytics(results)
        detections: List[Detection] = []
        for xyxy, conf, cls_id in zip(sv_det.xyxy, sv_det.confidence, sv_det.class_id):
            detections.append(Detection(
                bbox=tuple(float(v) for v in xyxy),
                confidence=float(conf),
                class_id=int(cls_id),
                class_name=self.class_names.get(int(cls_id), str(cls_id)),
            ))
        return detections, sv_det

    def warmup(self, size: Optional[int] = None) -> None:
        """Run one dummy inference so the first real frame isn't slow."""
        s = size or self.config.img_size
        dummy = np.zeros((s, s, 3), dtype=np.uint8)
        try:
            self.detect(dummy)
            logger.info("YOLO warmup complete")
        except Exception as exc:  # noqa: BLE001
            logger.warning("YOLO warmup skipped: {}", exc)
