"""Video-file ingestion.

Provides:
  * :class:`VideoReader` — a simple iterator over frames (used for offline /
    timeline video analysis where we want every frame, in order).
  * :class:`ThreadedVideoSource` — a background producer that pushes frames into
    a :class:`FrameBuffer` (used by the live pipeline so decode and inference
    overlap on separate threads).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Iterator, Optional

import cv2

from src.ingestion.frame_buffer import FrameBuffer
from src.utils.image_utils import guess_modality
from src.utils.logger import logger
from src.utils.types import Frame, Modality


class VideoReader:
    """Synchronous frame-by-frame reader for offline video analysis."""

    def __init__(self, path: str | Path, modality: Optional[Modality] = None):
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.path}")
        # Guard against bogus fps (some containers report 0 or the 90000 RTP
        # clock); fall back to a sane 30 fps for timestamping.
        _fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = _fps if 1.0 < _fps <= 120.0 else 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._modality = modality
        self.duration_sec = self.frame_count / self.fps if self.fps else 0.0
        logger.info(
            "Opened video {} ({}x{}, {:.1f} fps, {} frames, {:.1f}s)",
            Path(self.path).name, self.width, self.height, self.fps,
            self.frame_count, self.duration_sec,
        )

    def __iter__(self) -> Iterator[Frame]:
        idx = 0
        while True:
            ok, img = self._cap.read()
            if not ok:
                break
            source_ts = idx / self.fps if self.fps else 0.0
            modality = self._modality or guess_modality(img)
            yield Frame(
                image=img,
                frame_id=idx,
                timestamp=time.time(),
                source_ts=source_ts,
                modality=modality,
            )
            idx += 1
        self.release()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


class ThreadedVideoSource:
    """Decode a video file on a background thread into a :class:`FrameBuffer`.

    ``target_fps``:
      * 0  -> decode as fast as possible (offline batch style).
      * >0 -> pace decoding to that fps (e.g. for realistic file "playback").
    """

    def __init__(
        self,
        path: str | Path,
        buffer: FrameBuffer,
        target_fps: float = 0.0,
        modality: Optional[Modality] = None,
        loop: bool = False,
    ):
        self.path = str(path)
        self.buffer = buffer
        self.target_fps = target_fps
        self.modality = modality
        self.loop = loop
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Stream telemetry, probed once on construction.
        self.width = 0
        self.height = 0
        self.stream_fps = 0.0
        try:
            probe = VideoReader(self.path, self.modality)
            self.width, self.height = probe.width, probe.height
            self.stream_fps = probe.fps
            probe.release()
        except Exception:  # noqa: BLE001 — telemetry is best-effort
            pass

    def stream_info(self) -> dict:
        """Resolution + source FPS of the video file."""
        return {"width": self.width, "height": self.height,
                "fps": round(self.stream_fps, 2)}

    def start(self) -> "ThreadedVideoSource":
        self._thread = threading.Thread(target=self._run, name="video-source", daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        idx = 0
        try:
            while not self._stop.is_set():
                reader = VideoReader(self.path, self.modality)
                interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
                for frame in reader:
                    if self._stop.is_set():
                        break
                    frame.frame_id = idx
                    idx += 1
                    self.buffer.put(frame)
                    if interval:
                        time.sleep(interval)
                reader.release()
                if not self.loop:
                    break
        except Exception as exc:  # noqa: BLE001
            logger.exception("Video source crashed: {}", exc)
        finally:
            self.buffer.close()
            logger.info("Video source finished ({} frames)", idx)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
