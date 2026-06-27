"""Object detection backends + the backend-agnostic abstraction (doc 02).

Import :class:`Detector` for the contract and :func:`build_detector` for the
config-driven factory. ``YOLODetector`` is the default backend; ``RFDetrDetector``
is the optional RF-DETR alternative (imported lazily by the factory).
"""
from src.detection.base import Detector, build_detector
from src.detection.detector import YOLODetector

__all__ = ["Detector", "build_detector", "YOLODetector"]
