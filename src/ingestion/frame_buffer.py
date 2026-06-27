"""Thread-safe frame buffer with a drop-oldest policy.

For real-time ISR the *latest* frame matters far more than completeness. When
the consumer (YOLO+VLM) falls behind the producer (camera/RTSP), we drop the
oldest queued frames so detection always operates on near-live imagery instead
of accumulating unbounded latency.
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Optional

from src.utils.types import Frame


class FrameBuffer:
    """A bounded, lock-protected ring buffer for :class:`Frame` objects."""

    def __init__(self, maxsize: int = 64, drop_oldest: bool = True):
        self._buf: deque[Frame] = deque(maxlen=maxsize if drop_oldest else None)
        self._maxsize = maxsize
        self._drop_oldest = drop_oldest
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._closed = False
        self.dropped = 0          # diagnostics: total frames dropped
        self.received = 0
        # Newest frame ever queued, kept for *display*. Unlike the deque it is
        # NOT consumed by the inference reader, so the render path can show the
        # freshest captured frame decoupled from inference (latency roadmap M2).
        self._latest: Optional[Frame] = None

    def put(self, frame: Frame) -> bool:
        """Add a frame. Returns False if the frame was dropped (buffer full)."""
        with self._not_empty:
            if self._closed:
                return False
            self.received += 1
            # Always track the newest decoded frame for display, even if the
            # inference queue is full and we reject it below.
            self._latest = frame
            if not self._drop_oldest and len(self._buf) >= self._maxsize:
                return False
            before = len(self._buf)
            self._buf.append(frame)
            # deque(maxlen) silently evicts the oldest; count it as a drop.
            if self._drop_oldest and before == self._maxsize:
                self.dropped += 1
            self._not_empty.notify()
            return True

    def get(self, timeout: Optional[float] = 1.0) -> Optional[Frame]:
        """Pop the oldest frame, blocking up to ``timeout`` seconds.

        Returns None on timeout or when the buffer is closed and drained.
        """
        with self._not_empty:
            while not self._buf and not self._closed:
                if not self._not_empty.wait(timeout=timeout):
                    return None
            if self._buf:
                return self._buf.popleft()
            return None

    def get_latest(self, timeout: Optional[float] = 1.0) -> Optional[Frame]:
        """Pop the newest frame and discard everything older (lowest latency)."""
        with self._not_empty:
            while not self._buf and not self._closed:
                if not self._not_empty.wait(timeout=timeout):
                    return None
            if not self._buf:
                return None
            latest = self._buf.pop()
            self.dropped += len(self._buf)
            self._buf.clear()
            return latest

    def peek_latest(self) -> Optional[Frame]:
        """Return the newest queued frame WITHOUT removing it.

        For the decoupled display path: it survives the inference consumer
        draining the deque, so the feed always has the freshest captured frame
        to show regardless of how far inference has fallen behind.
        """
        with self._lock:
            return self._latest

    def close(self) -> None:
        with self._not_empty:
            self._closed = True
            self._not_empty.notify_all()

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buf)

    @property
    def is_closed(self) -> bool:
        return self._closed
