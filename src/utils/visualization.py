"""Drawing utilities: bounding boxes, track ids, confidence, and HUD overlays.

Pure OpenCV so it works headless and is fast enough for the live loop.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np

from src.utils.types import Detection, FrameResult, Modality

# Deterministic palette so a given track id keeps the same colour over time.
_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def _color_for(track_id: int | None, class_id: int) -> tuple[int, int, int]:
    key = track_id if track_id is not None else class_id
    return _PALETTE[key % len(_PALETTE)]


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    show_conf: bool = True,
    show_track_id: bool = True,
) -> np.ndarray:
    """Return a copy of ``frame`` with annotated detections."""
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        color = _color_for(det.track_id, det.class_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        parts = [det.class_name]
        if show_track_id and det.track_id is not None:
            parts.append(f"#{det.track_id}")
        if show_conf:
            parts.append(f"{det.confidence:.2f}")
        label = " ".join(parts)

        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(0, y1 - th - base - 2)
        cv2.rectangle(out, (x1, ly), (x1 + tw + 4, ly + th + base + 2), color, -1)
        cv2.putText(
            out, label, (x1 + 2, ly + th),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


def draw_hud(
    frame: np.ndarray,
    result: FrameResult,
    fps: float = 0.0,
    extra: str = "",
) -> np.ndarray:
    """Overlay a translucent header with frame stats."""
    out = frame
    counts = result.count_by_class()
    summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "no objects"
    line1 = f"[{result.modality.value}] f#{result.frame_id}  fps:{fps:4.1f}  tracks:{len(result.detections)}"
    line2 = summary if not extra else f"{summary}  |  {extra}"

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], 52), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
    cv2.putText(out, line1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(out, line2, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)
    return out


_FUSION_COLORS = {
    "ACTIVE_VEHICLE": (0, 165, 255), "THERMAL_CONFIRMED": (0, 255, 255),
    "CONCEALED_HEAT": (0, 0, 255), "CONTACT_OUTSIDE_EO": (255, 0, 255),
}


def draw_fusion_markers(img: np.ndarray, markers: List[dict],
                        registration: dict) -> np.ndarray:
    """Draw EO/IR fusion contact markers + the registration banner in place.

    ``markers`` is a list of ``{"pos": (x, y) | None, "type": str}`` and
    ``registration`` a ``{"state", "scale", "confidence", "locked"}`` dict.
    """
    h, w = img.shape[:2]
    for mk in markers:
        pos = mk.get("pos")
        if pos is None:
            continue
        x, y = int(pos[0]), int(pos[1])
        typ = mk.get("type", "")
        col = _FUSION_COLORS.get(typ, (255, 255, 255))
        if typ == "CONTACT_OUTSIDE_EO":
            # Clamp to the frame edge and draw a directional cue.
            ex, ey = min(max(x, 12), w - 12), min(max(y, 12), h - 12)
            cv2.circle(img, (ex, ey), 14, col, 2)
            cv2.putText(img, "IR contact >", (max(0, ex - 40), max(12, ey - 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        else:
            cv2.drawMarker(img, (x, y), col, cv2.MARKER_DIAMOND, 22, 2)
            cv2.putText(img, typ.replace("_", " ").title(), (x + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    if registration:
        banner = (f"FUSION {registration.get('state', '')}  "
                  f"scale={registration.get('scale', 0):.2f}  "
                  f"conf={registration.get('confidence', 0):.2f}")
        cv2.putText(img, banner, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if registration.get("locked") else (0, 200, 255),
                    2, cv2.LINE_AA)
    return img


def compose_overlay(
    raw: np.ndarray,
    detections: List[Detection],
    frame_id: int,
    fps: float,
    modality: Modality,
    *,
    show_conf: bool = True,
    show_track_id: bool = True,
    fusion_markers: List[dict] | None = None,
    registration: dict | None = None,
) -> np.ndarray:
    """Render-thread compositing: draw AI overlay *data* onto the live raw frame.

    This runs on the UI/render thread (latency roadmap M2), not the inference
    thread. The ``raw`` frame may be newer than the overlay (AI lags), so boxes
    trail moving targets slightly — the accepted trade for a feed that never
    freezes on an inference spike. Returns a fresh array (``raw`` is not mutated).
    """
    out = draw_detections(raw, detections, show_conf=show_conf,
                          show_track_id=show_track_id)   # returns a fresh copy
    result = FrameResult(frame_id, 0.0, 0.0, detections, modality)
    out = draw_hud(out, result, fps=fps)
    if fusion_markers or registration:
        draw_fusion_markers(out, fusion_markers or [], registration or {})
    return out
