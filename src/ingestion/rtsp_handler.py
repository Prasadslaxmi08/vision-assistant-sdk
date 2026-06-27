"""RTSP / live-camera ingestion with automatic reconnection.

Runs a dedicated capture thread that reads from an RTSP URL (or a local webcam
index) and pushes frames into a :class:`FrameBuffer`. The thread:
  * opens the stream with a **low-latency capture layer** lifted from ``RTSP.py``
    — GStreamer first (``drop-on-latency``, ``max-buffers=1``, ``latency=0``,
    ``sync=false``) with an FFmpeg fallback that sets the aggressive low-latency
    options (``fflags;nobuffer|flags;low_delay|max_delay``) and a 1-frame buffer,
  * detects stalls via a read-timeout watchdog,
  * reconnects with a back-off when the stream drops.

Only the *streaming* layer is taken from ``RTSP.py``; its Tkinter UI, PTZ, zoom and
recording are intentionally dropped. Frame delivery stays on this repo's clean
:class:`FrameBuffer` (drop-oldest) so the rest of the pipeline is unchanged.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional, Union

import cv2

from src.ingestion.frame_buffer import FrameBuffer
from src.utils.image_utils import guess_modality
from src.utils.logger import logger
from src.utils.types import Frame, Modality

# GStreamer is tried first for RTSP when available (set USE_GSTREAMER=0 to force
# FFmpeg). The pip ``opencv-python`` wheel usually has no GStreamer backend, so the
# open simply fails and we fall back to FFmpeg — that path carries the same
# low-latency flags, so latency is good either way.
USE_GSTREAMER = os.environ.get("USE_GSTREAMER", "1") == "1"


def _gst_pipeline(url: str, transport: str) -> str:
    """Low-latency GStreamer pipeline for an H.264 RTSP source (from RTSP.py)."""
    return (
        f"rtspsrc location={url} protocols={transport} latency=0 "
        f"drop-on-latency=true do-rtcp=true retry=10 timeout=5 ! "
        f"rtph264depay ! h264parse config-interval=1 ! "
        f"avdec_h264 error-resilient=1 skip-frame=1 ! "
        f"videoconvert ! appsink sync=false drop=true max-buffers=1 emit-signals=false"
    )


class RTSPStreamHandler:
    def __init__(
        self,
        source: Union[str, int],
        buffer: FrameBuffer,
        transport: str = "tcp",
        reconnect_delay: float = 3.0,
        max_reconnects: int = -1,
        read_timeout: float = 10.0,
        modality: Optional[Modality] = None,
    ):
        self.source = source
        self.buffer = buffer
        self.transport = transport
        self.reconnect_delay = reconnect_delay
        self.max_reconnects = max_reconnects
        self.read_timeout = read_timeout
        self.modality = modality

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.connected = False
        self.reconnect_count = 0
        self._last_frame_time = 0.0
        # Wall-clock of the very first frame, used to make source_ts a sane
        # stream-relative elapsed time (00:00:00-based) instead of raw epoch
        # seconds. Persists across reconnects so the timeline stays monotonic.
        self._stream_start: Optional[float] = None

        # Stream telemetry, filled in once the capture opens.
        self.width = 0
        self.height = 0
        self.stream_fps = 0.0          # declared by the stream (often bogus on RTSP)
        self._measured_fps = 0.0       # measured from real frame arrival cadence

    def stream_info(self) -> dict:
        """Resolution + source FPS (0 until connected).

        Prefers the *measured* arrival rate, because RTSP/H.264 streams routinely
        report a bogus ``CAP_PROP_FPS`` of 90000 (that's the RTP 90 kHz timestamp
        clock, not a frame rate).
        """
        fps = self._measured_fps if self._measured_fps > 0 else self.stream_fps
        return {"width": self.width, "height": self.height,
                "fps": round(fps, 2)}

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> "RTSPStreamHandler":
        self._thread = threading.Thread(target=self._run, name="rtsp-capture", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self.buffer.close()

    # -- internals ----------------------------------------------------------
    def _open_rtsp(self, url: str) -> Optional[cv2.VideoCapture]:
        """Open an RTSP URL with the low-latency layer (GStreamer → FFmpeg)."""
        # FFmpeg low-latency capture options (from RTSP.py): no input buffering,
        # low-delay decode, capped reorder delay, and a socket timeout (µs).
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{self.transport}|fflags;nobuffer|flags;low_delay|"
            f"max_delay;100000|stimeout;{int(self.read_timeout * 1e6)}"
        )
        if USE_GSTREAMER:
            try:
                cap = cv2.VideoCapture(_gst_pipeline(url, self.transport),
                                       cv2.CAP_GSTREAMER)
                if cap.isOpened():
                    logger.info("RTSP opened via GStreamer (low-latency)")
                    return cap
                cap.release()
            except Exception as exc:  # noqa: BLE001 — GStreamer backend may be absent
                logger.debug("GStreamer open failed ({}); using FFmpeg", exc)
        return cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    def _open(self) -> Optional[cv2.VideoCapture]:
        if isinstance(self.source, str) and self.source.lower().startswith("rtsp"):
            cap = self._open_rtsp(self.source)
        else:
            cap = cv2.VideoCapture(self.source)  # webcam index or local file path
        if cap is None:
            return None
        # Keep OpenCV's internal buffer tiny so we always read fresh frames.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            return None
        # Capture stream telemetry (resolution + source FPS) for the UI.
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
        # Only trust a plausible declared fps; RTSP often reports 90000 (the RTP
        # 90 kHz clock) or 0. Real cadence is measured live in _run().
        declared = float(cap.get(cv2.CAP_PROP_FPS))
        if 1.0 < declared <= 120.0:
            self.stream_fps = declared
        logger.info("Stream opened: {}x{} @ {} fps (declared)",
                    self.width, self.height,
                    f"{self.stream_fps:.1f}" if self.stream_fps else "unknown")
        return cap

    def _run(self) -> None:
        idx = 0
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                self.connected = False
                if not self._should_retry():
                    break
                logger.warning("RTSP connect failed ({}). Retrying in {:.1f}s",
                               self.source, self.reconnect_delay)
                self._sleep(self.reconnect_delay)
                self.reconnect_count += 1
                continue

            self.connected = True
            self.reconnect_count = 0
            self._last_frame_time = time.time()
            prev_frame_t = 0.0
            logger.info("RTSP connected: {}", self.source)

            while not self._stop.is_set():
                ok, img = cap.read()
                now = time.time()
                if not ok or img is None:
                    if now - self._last_frame_time > self.read_timeout:
                        logger.warning("RTSP read timeout — reconnecting")
                        break
                    time.sleep(0.01)
                    continue
                # Measure true source fps from inter-frame arrival (EMA-smoothed).
                if prev_frame_t:
                    dt = now - prev_frame_t
                    if dt > 0:
                        inst = 1.0 / dt
                        self._measured_fps = (inst if self._measured_fps == 0.0
                                              else 0.9 * self._measured_fps + 0.1 * inst)
                prev_frame_t = now
                self._last_frame_time = now
                if self._stream_start is None:
                    self._stream_start = now
                modality = self.modality or guess_modality(img)
                self.buffer.put(Frame(
                    image=img, frame_id=idx, timestamp=now,
                    source_ts=now - self._stream_start, modality=modality,
                ))
                idx += 1

            cap.release()
            self.connected = False
            if self._stop.is_set() or not self._should_retry():
                break
            self.reconnect_count += 1
            self._sleep(self.reconnect_delay)

        self.buffer.close()
        logger.info("RTSP handler stopped ({} frames)", idx)

    def _should_retry(self) -> bool:
        return self.max_reconnects < 0 or self.reconnect_count < self.max_reconnects

    def _sleep(self, seconds: float) -> None:
        # Interruptible sleep so stop() is responsive.
        self._stop.wait(timeout=seconds)
