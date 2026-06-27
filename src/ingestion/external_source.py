"""External (push) frame source — for hosts that already own capture.

The RTSP / video / webcam sources in this package *pull* frames on their own
background thread. But a surveillance app or Ground Control Station usually
already manages its own cameras (RTSP, PTZ, recording) and simply wants the AI
layer to look at the frames it is *already* decoding.

:class:`ExternalFrameSource` is the Input-Layer adapter for that case. It owns no
capture thread of its own: the host calls :meth:`submit` for every frame it
decodes, and the frame is wrapped into a :class:`~src.utils.types.Frame` and
pushed into the shared :class:`~src.ingestion.frame_buffer.FrameBuffer` exactly
like the pull sources do. The engine's inference loop cannot tell the difference.

It satisfies the same duck-typed source contract the engine expects
(``start`` / ``stop`` / ``connected`` / ``stream_info``), so the engine treats a
push stream identically to a pulled RTSP stream.
"""
from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from src.ingestion.frame_buffer import FrameBuffer
from src.utils.types import Frame, Modality


class ExternalFrameSource:
    """A *push* source: the host feeds frames in via :meth:`submit`.

    Parameters
    ----------
    buffer:
        The shared frame buffer the engine drains for this stream.
    modality:
        Default modality stamped on submitted frames (EO or IR).
    name:
        Stream label ('EO' / 'IR'), used in telemetry.
    stale_after:
        Seconds without a new frame after which :attr:`connected` reports False,
        so the engine/host can show the feed as dropped. ``0`` disables the check
        (connected while running).
    """

    def __init__(self, buffer: FrameBuffer, modality: Modality,
                 name: str = "external", stale_after: float = 5.0):
        self.buffer = buffer
        self.modality = modality
        self.name = name
        self.stale_after = stale_after
        self._frame_id = 0
        self._running = False
        self._t0: Optional[float] = None
        self._last_ts = 0.0
        self.width = 0
        self.height = 0
        self._lock = threading.Lock()

    # -- source contract expected by the engine --------------------------------
    @property
    def connected(self) -> bool:
        """True while running and (if ``stale_after``) a frame arrived recently."""
        if not self._running:
            return False
        if self.stale_after <= 0:
            return True
        return self._last_ts > 0.0 and (time.time() - self._last_ts) < self.stale_after

    def start(self) -> "ExternalFrameSource":
        """No capture thread to spin up — just arm the source for submissions."""
        self._running = True
        return self

    def stop(self) -> None:
        self._running = False
        self.buffer.close()

    def stream_info(self) -> dict:
        """Resolution + (unknown, host-driven) FPS of the pushed stream."""
        return {"width": self.width, "height": self.height, "fps": 0.0}

    # -- host-facing push API --------------------------------------------------
    def submit(self, image: np.ndarray, source_ts: Optional[float] = None,
               modality: Optional[Modality] = None,
               timestamp: Optional[float] = None, copy: bool = True) -> bool:
        """Push one decoded BGR frame into the engine.

        ``image`` must be an HxWx3 BGR uint8 array (OpenCV's native layout — the
        same thing ``cv2.VideoCapture.read()`` returns). Returns False if the
        source is stopped or the frame was dropped because the buffer was full
        (real-time back-pressure — the engine keeps the freshest frame).

        ``copy`` defaults to True so the engine owns an immutable snapshot even if
        the host reuses its capture buffer; pass ``copy=False`` only when the
        host hands over a fresh array it will not touch again.
        """
        if not self._running:
            return False
        if image is None or image.ndim != 3:
            return False
        ts = timestamp if timestamp is not None else time.time()
        if self._t0 is None:
            self._t0 = ts
        s_ts = source_ts if source_ts is not None else (ts - self._t0)
        img = np.ascontiguousarray(image.copy()) if copy else image
        h, w = img.shape[:2]
        with self._lock:
            self.width, self.height = int(w), int(h)
            self._last_ts = ts
            fid = self._frame_id
            self._frame_id += 1
        frame = Frame(image=img, frame_id=fid, timestamp=ts, source_ts=s_ts,
                      modality=modality or self.modality)
        return self.buffer.put(frame)
