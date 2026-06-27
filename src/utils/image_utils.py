"""Image helpers: modality detection, resizing, BGR<->RGB, PIL conversion."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from src.utils.types import Modality


def guess_modality(image: np.ndarray, ir_saturation_thresh: float = 0.12) -> Modality:
    """Heuristically classify a frame as EO or IR.

    Thermal/IR imagery is typically near-grayscale (low colour saturation) even
    when stored in 3 channels (white-hot / black-hot palettes) — colourised IR
    palettes are the exception. We measure mean HSV saturation: low saturation
    strongly implies IR. This is a *hint* for prompting/visualisation, not a
    hard classification, and the UI always lets the user override it.
    """
    if image.ndim == 2 or image.shape[2] == 1:
        return Modality.IR
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_sat = float(hsv[:, :, 1].mean()) / 255.0
    return Modality.IR if mean_sat < ir_saturation_thresh else Modality.EO


def resize_keep_aspect(image: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale so width <= max_width, preserving aspect ratio. Never upscales."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / float(w)
    return cv2.resize(image, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def to_pil(image_bgr: np.ndarray) -> Image.Image:
    """Convert a BGR uint8 array to a PIL RGB image (what Qwen2.5-VL expects)."""
    return Image.fromarray(bgr_to_rgb(image_bgr))


def to_thermal_palette(gray: np.ndarray, palette: int = cv2.COLORMAP_INFERNO) -> np.ndarray:
    """Apply a thermal colour map to a single-channel IR frame for display."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, palette)


def crop_bbox(image: np.ndarray, bbox: Tuple[float, float, float, float], pad: int = 0) -> np.ndarray:
    """Crop an xyxy bbox with optional padding, clamped to image bounds."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(w, int(x2) + pad)
    y2 = min(h, int(y2) + pad)
    if x2 <= x1 or y2 <= y1:
        return image
    return image[y1:y2, x1:x2]


def crop_to_jpeg(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    max_width: int = 160,
    pad_frac: float = 0.08,
    quality: int = 70,
) -> "bytes | None":
    """Crop a detection bbox and return a small JPEG (bytes) for embedding.

    Pads the box by ``pad_frac`` of its size for context, downscales to
    ``max_width``, and JPEG-encodes. Returns ``None`` if the crop is empty.
    Used to build the "recorded objects" thumbnail gallery in reports.
    """
    x1, y1, x2, y2 = bbox
    pad = int(round(max(x2 - x1, y2 - y1) * pad_frac))
    crop = crop_bbox(image, bbox, pad=pad)
    if crop is None or crop.size == 0:
        return None
    crop = resize_keep_aspect(crop, max_width)
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    ok, enc = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return enc.tobytes() if ok else None
