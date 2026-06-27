"""Asynchronous VLM scheduling.

The worker decouples expensive VLM inference from the real-time detection loop.
The pipeline submits *requests* (a frame snapshot + context); a background
thread consumes them subject to:

  * a periodic timer (run at least every ``periodic_interval_sec``),
  * event triggers (run when a significant event arrives, if enabled),
  * a minimum spacing between calls (``min_seconds_between_calls``),
  * a bounded pending queue that drops stale requests (keep newest).

Results are delivered via a thread-safe callback so the UI/mission layer can
consume them without blocking detection.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from src.config.settings import VLMConfig
# Record type from the shared contract, not the CV event module (repo-split
# prep, doc 04 — removes an AI→CV import).
from src.contracts import Event
from src.utils.logger import logger
from src.vlm.qwen_vlm import QwenVLM


@dataclass
class VLMRequest:
    image: np.ndarray
    modality: str
    detections: List[str]
    events: List[Event] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)   # spatial/fusion observations
    frame_id: int = 0
    source_ts: float = 0.0
    timestamp: float = 0.0
    reason: str = "periodic"   # "periodic" | "event"


@dataclass
class VLMResult:
    text: str
    request: VLMRequest
    latency: float
    timestamp: float


class VLMWorker:
    def __init__(self, vlm: QwenVLM, config: VLMConfig,
                 on_result: Optional[Callable[[VLMResult], None]] = None):
        self.vlm = vlm
        self.config = config
        self.on_result = on_result

        self._pending: List[VLMRequest] = []
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._last_call_ts = 0.0
        self._last_periodic_ts = 0.0
        self.busy = False
        self.total_calls = 0
        self.dropped = 0

    # ------------------------------------------------------------ lifecycle
    def start(self) -> "VLMWorker":
        if not self.config.enabled:
            logger.info("VLM disabled in config; worker not started")
            return self
        self._thread = threading.Thread(target=self._run, name="vlm-worker", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def stats(self) -> dict:
        """Queue health for profiling (pending / dropped / calls / busy)."""
        with self._lock:
            pending = len(self._pending)
        return {"pending": pending, "dropped": self.dropped,
                "total_calls": self.total_calls, "busy": self.busy}

    # -------------------------------------------------------------- submit
    def submit(self, request: VLMRequest) -> bool:
        """Queue a request. Honours periodic/event gating and queue bounds.

        Returns True if the request was accepted into the pending queue.
        """
        if not self.config.enabled:
            return False
        now = request.timestamp or time.time()

        is_event = bool(request.events or request.notes) and self.config.trigger_on_events
        periodic_due = (now - self._last_periodic_ts) >= self.config.periodic_interval_sec
        if not is_event and not periodic_due:
            return False
        if is_event:
            request.reason = "event"

        with self._lock:
            self._pending.append(request)
            # Keep only the newest N requests (drop stale ones under load).
            if len(self._pending) > self.config.max_pending_requests:
                overflow = len(self._pending) - self.config.max_pending_requests
                self._pending = self._pending[overflow:]
                self.dropped += overflow
            if periodic_due and not is_event:
                self._last_periodic_ts = now
        self._wake.set()
        return True

    # ---------------------------------------------------------------- loop
    def _run(self) -> None:
        logger.info("VLM worker started (periodic={}s, event_trigger={})",
                    self.config.periodic_interval_sec, self.config.trigger_on_events)
        # Trigger model load up front so the first request isn't a cold start.
        self.vlm.load()
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            req = self._next_request()
            if req is None:
                continue

            # Rate limit between actual model calls.
            wait = self.config.min_seconds_between_calls - (time.time() - self._last_call_ts)
            if wait > 0:
                if self._stop.wait(timeout=wait):
                    break

            self._execute(req)
        logger.info("VLM worker stopped (calls={}, dropped={})",
                    self.total_calls, self.dropped)

    def _next_request(self) -> Optional[VLMRequest]:
        with self._lock:
            if not self._pending:
                return None
            # Prefer the highest-priority event request, else the newest frame.
            events_first = sorted(
                self._pending,
                key=lambda r: (max((e.priority for e in r.events), default=0), r.timestamp),
                reverse=True,
            )
            req = events_first[0]
            self._pending.clear()
            return req

    def _execute(self, req: VLMRequest) -> None:
        self.busy = True
        t0 = time.time()
        try:
            text = self.vlm.scene_summary(
                req.image, req.modality, req.detections,
                [e.description for e in req.events] + list(req.notes),
            )
            self._last_call_ts = time.time()
            self.total_calls += 1
            result = VLMResult(text=text, request=req,
                               latency=time.time() - t0, timestamp=time.time())
            if self.on_result:
                try:
                    self.on_result(result)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("VLM result callback failed: {}", exc)
        finally:
            self.busy = False
