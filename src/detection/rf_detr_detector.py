"""RF-DETR object detector backend (Roboflow ``rfdetr``).

A first-class, config-selectable alternative to YOLOv11 that honours the exact
same contract: ``detect(image) -> (list[Detection], supervision.Detections)``,
so the tracker / ReID / fusion / events layers consume it identically. RF-DETR is
a *swappable replacement* for YOLO (one detector loaded at a time), so peak VRAM
is one detector + the VLM — the same envelope as today.

>>> NOT YET VALIDATED ON THE BOX. The ``rfdetr`` package is an optional dependency
    (imported lazily, so a YOLO-only install is unaffected) and its exact predict
    signature + COCO class-id space must be confirmed on the RTX 5060 alongside the
    detector benchmark. The adapter assumes RF-DETR emits COCO **0-indexed 80-class**
    ids matching YOLO; ``warmup()`` logs the observed class names so the operator can
    verify the mapping before trusting the ``classes`` filter. See doc 02 §3.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.config.settings import DetectionConfig
from src.utils.logger import logger
from src.utils.types import Detection


class RFDetrDetector:
    def __init__(self, config: DetectionConfig, device: str = "cuda:0",
                 half: bool = True):
        try:
            import rfdetr  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on optional wheel
            raise ImportError(
                "detection.backend='rf_detr' requires the optional 'rfdetr' "
                "package. Install it on the box (pip install rfdetr) or set "
                "detection.backend='yolo11' in config.yaml."
            ) from exc

        self.config = config
        self.rf = config.rf_detr
        self.device = device
        self.half = half and device.startswith("cuda")
        self._class_filter = set(config.classes) if config.classes else None
        self.model = self._load()
        self._names = self._coco_names()
        logger.info("RF-DETR ready (variant={}, img_size={}, device={}, half={})",
                    self.rf.variant, self.rf.img_size, self.device, self.half)

    # ---------------------------------------------------------------- load
    def _load(self):
        """Instantiate the requested RF-DETR variant.

        The ``rfdetr`` constructor signature has varied across releases
        (``resolution`` / ``pretrain_weights``), so we degrade gracefully."""
        import rfdetr
        variant_map = {
            "nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
            "base": "RFDETRBase", "large": "RFDETRLarge",
        }
        cls_name = variant_map.get(self.rf.variant, "RFDETRBase")
        cls = getattr(rfdetr, cls_name, None) or getattr(rfdetr, "RFDETRBase")
        kwargs = {}
        if self.rf.img_size:
            kwargs["resolution"] = self.rf.img_size
        if self.rf.weights:
            kwargs["pretrain_weights"] = self.rf.weights
        try:
            return cls(**kwargs)
        except TypeError:  # older signature without these kwargs
            return cls(pretrain_weights=self.rf.weights) if self.rf.weights else cls()
        except ValueError as exc:
            # The configured img_size isn't valid for this variant (each variant
            # has its own divisibility rule, e.g. base wants /56, nano wants /32).
            # Fall back to the variant's own default resolution rather than failing.
            logger.warning("RF-DETR {} rejected resolution {} ({}); using the "
                           "variant default.", self.rf.variant, self.rf.img_size, exc)
            kwargs.pop("resolution", None)
            return cls(**kwargs)

    @staticmethod
    def _coco_names() -> dict:
        try:
            from rfdetr.util.coco_classes import COCO_CLASSES
            if isinstance(COCO_CLASSES, dict):
                return {int(k): str(v) for k, v in COCO_CLASSES.items()}
            return {i: str(n) for i, n in enumerate(COCO_CLASSES)}
        except Exception:  # noqa: BLE001
            return {}

    # -------------------------------------------------------------- detect
    def detect(self, image: np.ndarray) -> Tuple[List[Detection], "object"]:
        import cv2
        import supervision as sv  # noqa: F401  (kept for parity with YOLO path)

        # RF-DETR's predict() calls torchvision F.to_tensor, which rejects negative
        # strides — so a plain ``image[:, :, ::-1]`` view fails. cvtColor returns a
        # fresh contiguous RGB array.
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        sv_det = self.model.predict(rgb, threshold=self.config.conf_threshold)

        # Filter to the configured COCO ids (same id space the pipeline expects).
        if (self._class_filter is not None and sv_det.class_id is not None
                and len(sv_det) > 0):
            keep = np.array([int(c) in self._class_filter
                             for c in sv_det.class_id], dtype=bool)
            sv_det = sv_det[keep]

        detections: List[Detection] = []
        if sv_det.class_id is not None:
            for xyxy, conf, cls_id in zip(sv_det.xyxy, sv_det.confidence,
                                          sv_det.class_id):
                cid = int(cls_id)
                detections.append(Detection(
                    bbox=tuple(float(v) for v in xyxy),
                    confidence=float(conf),
                    class_id=cid,
                    class_name=self._names.get(cid, str(cid)),
                ))
        return detections, sv_det

    def warmup(self, size: Optional[int] = None) -> None:
        s = size or self.rf.img_size
        dummy = np.zeros((s, s, 3), dtype=np.uint8)
        try:
            dets, _ = self.detect(dummy)
            logger.info("RF-DETR warmup complete (mapping check: {} names loaded)",
                        len(self._names))
        except Exception as exc:  # noqa: BLE001
            logger.warning("RF-DETR warmup skipped: {}", exc)

    # --- Detector protocol -----------------------------------------------
    @property
    def name(self) -> str:
        return f"rf-detr-{self.rf.variant}"

    @property
    def classes(self) -> Optional[List[int]]:
        return self.config.classes
