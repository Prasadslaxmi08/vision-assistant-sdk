"""Detector backend abstraction (detector modernization, doc 02).

Every backend returns the **same** ``(list[Detection], supervision.Detections)``
pair, so the tracker / ReID / fusion / events layers are untouched when the
detector is swapped. :func:`build_detector` picks the implementation from
``config.detection.backend`` — switching backends is a config edit, not a code
change. YOLOv11 stays the default; RF-DETR is a first-class, config-selectable
alternative (one detector loaded at a time, so peak VRAM is unchanged).
"""
from __future__ import annotations

from typing import List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from src.utils.types import Detection


@runtime_checkable
class Detector(Protocol):
    """Backend-agnostic detector contract consumed by the whole pipeline."""

    def detect(self, image: np.ndarray) -> Tuple[List[Detection], "object"]:
        """BGR frame -> (framework-agnostic detections, ``sv.Detections`` for the tracker)."""
        ...

    def warmup(self, size: Optional[int] = None) -> None: ...

    @property
    def name(self) -> str: ...                       # e.g. "yolo11m", "rf-detr-base"

    @property
    def classes(self) -> Optional[List[int]]: ...    # COCO id filter, or None for all


def build_detector(config, tiled: bool = False) -> "Detector":
    """Construct the configured detector backend from an ``AppConfig``.

    Reads ``config.detection.backend`` (``"yolo11"`` | ``"rf_detr"``) and returns
    the matching implementation, configured with the shared device / half-precision
    settings. Backend modules are imported lazily so a YOLO-only install never
    needs the ``rfdetr`` package (and vice-versa).

    When ``tiled`` is set and ``config.detection.tiling.enabled``, the detector is
    wrapped in a :class:`~src.detection.sliced.SlicedDetector` for small-object
    recall on high-resolution stills (still honours the ``Detector`` contract, so
    downstream code is unchanged). The real-time engine leaves ``tiled=False``.
    """
    det = config.detection
    device = config.system.device
    half = config.system.half_precision
    backend = det.backend
    if backend == "yolo11":
        from src.detection.detector import YOLODetector
        # Resolve a bundled weight against the SDK root so an *embedding* host
        # (whose working directory is its own project) loads the SDK's copy
        # instead of re-downloading yolo11m.pt into the host's folder.
        try:
            from pathlib import Path
            mp = Path(det.model_path)
            if not mp.is_absolute():
                bundled = config.abs_path(det.model_path)
                if bundled.exists():
                    det.model_path = str(bundled)
        except Exception:  # noqa: BLE001 — never block detector construction
            pass
        base = YOLODetector(det, device=device, half=half)
    elif backend == "rf_detr":
        from src.detection.rf_detr_detector import RFDetrDetector
        base = RFDetrDetector(det, device=device, half=half)
    else:
        raise ValueError(
            f"Unknown detection.backend {backend!r}; expected 'yolo11' or 'rf_detr'.")

    if tiled and det.tiling.enabled:
        from src.detection.sliced import SlicedDetector
        return SlicedDetector(base, det.tiling)
    return base
