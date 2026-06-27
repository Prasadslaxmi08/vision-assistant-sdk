"""Vision Assistant engine — the live, dual-stream AI Core orchestrator.

Runs one or two **independent** sensor streams (separate EO and IR RTSP feeds or
video files) and, when both are present, performs cross-sensor EO/IR fusion over
an automatically-estimated registration (no calibration — see
:mod:`src.reasoning.registration`).

This is the SDK's real-time engine. It is a pure AI layer: it does **not** record,
persist, score threat, generate reports, or own a UI — host camera software does
that. It produces structured results (JSON / callbacks) and overlay frames.

Threading model — a **single inference thread** that round-robins both streams
through one shared detector. On an 8 GB GPU this avoids a second model and any
concurrent-CUDA contention; each stream keeps its own tracking state. One stream
is the *primary* — EO when present, otherwise IR (an IR-only session is fully
supported): events, scene understanding and summaries anchor to the primary's
identities. When both sensors are present EO is primary and IR contributes thermal
signatures that cross-sensor fusion correlates onto the EO picture (fusion requires
both sensors and is gated separately).

**Real-time contract:** the display path shows the freshest *raw* frame immediately;
detection / tracking / fusion / VLM run on the inference thread and publish *overlay
data* (coordinates, not pixels) by a single atomic reference swap. The render thread
composites it — overlays may lag inference, the live video never does.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np

from src.config.settings import AppConfig
from src.contracts import VlmGateway
from src.detection import Detector, build_detector
from src.events.event_manager import EventManager
from src.ingestion.external_source import ExternalFrameSource
from src.ingestion.frame_buffer import FrameBuffer
from src.ingestion.rtsp_handler import RTSPStreamHandler
from src.ingestion.video_reader import ThreadedVideoSource
from src.intelligence.mission_intelligence import MissionIntelligence
from src.reasoning import (CrossSensorRegistrar, FusionEngine, Registration,
                           SpatialReasoner, blob_keypoints, detection_keypoints,
                           fov_scale_prior)
from src.tracking.reid import Reidentifier
from src.tracking.tracker import ObjectTracker, merge_detections
from src.utils.logger import logger
from src.utils.profiling import StageProfiler
from src.utils.types import Detection, FrameResult, Modality
from src.utils.visualization import compose_overlay
from src.vlm.qwen_vlm import QwenVLM
from src.vlm.vlm_worker import VLMRequest, VLMResult, VLMWorker

# Type of the Output-Layer result callback: receives a JSON-serialisable dict.
ResultCallback = Callable[[dict], None]


@dataclass
class StreamState:
    """Per-sensor capture + tracking state (everything that must NOT be shared)."""
    name: str
    modality: Modality
    buffer: FrameBuffer
    source: object
    tracker: ObjectTracker
    reid: Reidentifier
    latest_raw: Optional[np.ndarray] = None
    latest_overlay: Optional["OverlayResult"] = None
    latest_dets: List[Detection] = field(default_factory=list)
    frame_id: int = -1
    source_ts: float = 0.0
    last_ts: float = 0.0
    fps: float = 0.0
    _fps_t: float = 0.0
    _fps_n: int = 0

    def connected(self) -> bool:
        src = self.source
        return bool(getattr(src, "connected", True)) and self.latest_raw is not None


@dataclass
class OverlayResult:
    """AI output as *data*, not pixels.

    The inference thread fills this and swaps it into ``StreamState.latest_overlay``
    by a single atomic reference assignment; the render thread reads it and
    composites the boxes/HUD onto the live raw frame. Carrying coordinates (not a
    rendered image) means the AI→render handoff copies no pixels and the display
    never waits on drawing."""
    detections: List[Detection] = field(default_factory=list)
    frame_id: int = -1
    fps: float = 0.0
    modality: Modality = Modality.EO


@dataclass
class FrameView:
    """Cheap, high-frequency view for the live feed.

    Built every feed refresh, so it does **no** heavy aggregation: ``eo_frame`` /
    ``ir_frame`` are the freshest *raw* capture frames with the latest AI overlay
    composited on top by the render thread (the overlay may lag inference). GPU
    stats and timeline/intel/stats live in :class:`PanelView` instead."""
    eo_frame: Optional[np.ndarray] = None
    ir_frame: Optional[np.ndarray] = None
    eo_connected: bool = False
    ir_connected: bool = False
    eo_fps: float = 0.0           # AI/detection rate (inferences/sec for this stream)
    ir_fps: float = 0.0
    eo_capture_fps: float = 0.0   # real feed rate — frames captured/sec (decoupled from AI)
    ir_capture_fps: float = 0.0
    eo_tracks: int = 0
    ir_tracks: int = 0
    fusion_available: bool = False
    fusion_enabled: bool = False
    registration: dict = field(default_factory=dict)
    fusion_contacts: List[dict] = field(default_factory=list)
    vlm_busy: bool = False
    running: bool = False


@dataclass
class PanelView:
    """Heavier intelligence view for side panels / dashboards (poll occasionally)."""
    intel_messages: List[str] = field(default_factory=list)
    latest_summary: str = ""
    timeline: List[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    stream_info: dict = field(default_factory=dict)
    gpu: dict = field(default_factory=dict)


class VisionAssistant:
    """The Vision Assistant SDK engine. See module docstring for the contract."""

    def __init__(self, config: AppConfig, shared_vlm: Optional[QwenVLM] = None,
                 detector=None):
        self.config = config
        # ``detector`` injectable for testing; otherwise the config-selected
        # backend (rf_detr | yolo11) is built from config.detection.backend.
        self.detector: Detector = detector or build_detector(config)
        # Primary-stream reasoning + shared scene state (Intelligence Layer).
        self.events = EventManager(config.events)
        self.spatial = SpatialReasoner(config.spatial)
        self.fusion = FusionEngine(config.fusion)
        self.registrar = CrossSensorRegistrar(
            config.registration,
            scale_prior=fov_scale_prior(config.eo_sensor, config.ir_sensor))
        self.mission = MissionIntelligence(
            intel_history=config.intelligence.intel_feed_history,
            timeline_max=config.intelligence.timeline_max_events)
        # The engine only cues the VLM through a gateway (the async worker today;
        # a stream client to the Analyst process tomorrow).
        self.vlm = shared_vlm or QwenVLM(config.vlm)
        self.vlm_worker: VlmGateway = VLMWorker(
            self.vlm, config.vlm, on_result=self._on_vlm)
        # Runtime profiling. Zero-overhead when disabled.
        self.profiler = StageProfiler(enabled=config.profiling.enabled,
                                      window=config.profiling.window)

        self.eo: Optional[StreamState] = None
        self.ir: Optional[StreamState] = None
        self._fusion_enabled = False
        self._registration = Registration()
        self._fusion_contacts: List[dict] = []
        self._fusion_markers: List[dict] = []   # current-frame fusion markers (render data)
        # Cadence counters for sub-rating the heavy AI stages.
        self._fuse_tick = 0     # inference-loop cycles with fusion active
        self._reg_tick = 0      # fuse cycles (gates the RANSAC re-estimate)
        self._eo_tick = 0       # EO frames processed (gates spatial reasoning)
        self._last_periodic = 0.0   # last periodic VLM cue (wall-clock)

        # Output Layer: registered result callbacks (fired off the inference thread).
        self._result_cbs: List[ResultCallback] = []
        # Report filter: surface only these COCO class names (None = all). Defaults
        # from config (detection.report_classes); override at runtime via set_classes.
        from src.detection.coco_classes import normalize
        self.class_filter: Optional[set] = normalize(config.detection.report_classes)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._gpu_cache: dict = {}
        self._gpu_ts: float = 0.0
        self._cap_fps: dict = {}

    # ------------------------------------------------------ Output Layer: callbacks
    def on_result(self, callback: ResultCallback) -> None:
        """Register a callback that receives a JSON-serialisable result dict each
        time the primary stream produces new events / a fresh summary. Multiple
        callbacks are supported; exceptions in one never break the pipeline."""
        self._result_cbs.append(callback)

    def _emit(self, payload: dict) -> None:
        for cb in self._result_cbs:
            try:
                cb(payload)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Result callback failed: {}", exc)

    # ----------------------------------------------------------- source setup
    def _make_state(self, name: str, modality: Modality) -> StreamState:
        buf = FrameBuffer(maxsize=self.config.ingestion.buffer_size,
                          drop_oldest=self.config.ingestion.drop_policy == "newest_keeps")
        return StreamState(name=name, modality=modality, buffer=buf, source=None,
                           tracker=ObjectTracker(self.config.tracking),
                           reid=Reidentifier(self.config.reid))

    def add_source(self, role: str, source: Union[str, int],
                   is_video: bool = False, loop: bool = True) -> None:
        """Attach an EO or IR source. ``role`` is 'EO' or 'IR'.

        ``source`` is an RTSP URL, a webcam index, or — when ``is_video`` — a video
        file path. Call once for a single-sensor session, twice (EO + IR) for a
        dual-sensor session that enables fusion."""
        modality = Modality.EO if role.upper() == "EO" else Modality.IR
        state = self._make_state(role.upper(), modality)
        if is_video:
            state.source = ThreadedVideoSource(
                path=source, buffer=state.buffer,
                target_fps=self.config.ingestion.target_fps,
                modality=modality, loop=loop)
        else:
            src: Union[str, int] = (int(source) if str(source).isdigit() else source)
            state.source = RTSPStreamHandler(
                source=src, buffer=state.buffer,
                transport=self.config.ingestion.rtsp_transport,
                reconnect_delay=self.config.ingestion.rtsp_reconnect_delay,
                max_reconnects=self.config.ingestion.rtsp_max_reconnects,
                read_timeout=self.config.ingestion.read_timeout,
                modality=modality)
        if modality == Modality.EO:
            self.eo = state
        else:
            self.ir = state

    def add_frame_source(self, role: str) -> None:
        """Register a **push** source for a host that already owns capture.

        Use this — instead of :meth:`add_source` — when your application (a GCS,
        VMS, or any surveillance app) already decodes frames itself (its own RTSP,
        camera SDK, capture card, etc.) and just wants the AI layer to look at
        them. After ``start()``, feed each decoded BGR frame in with
        :meth:`submit_frame`. ``role`` is 'EO' or 'IR'; call it twice (EO + IR)
        for a dual-sensor session that can fuse."""
        role_u = role.upper()
        modality = Modality.EO if role_u == "EO" else Modality.IR
        state = self._make_state(role_u, modality)
        state.source = ExternalFrameSource(buffer=state.buffer, modality=modality,
                                           name=role_u)
        if modality == Modality.EO:
            self.eo = state
        else:
            self.ir = state

    def submit_frame(self, role: str, image: np.ndarray,
                     source_ts: Optional[float] = None,
                     copy: bool = True) -> bool:
        """Push one decoded BGR frame into a push source (see
        :meth:`add_frame_source`). ``role`` is 'EO' or 'IR'; ``image`` is an
        HxWx3 BGR uint8 array. Cheap and thread-safe — call it straight from your
        capture/decode thread. Returns False if the frame was dropped (buffer full
        — real-time back-pressure) or no push source is registered for ``role``.

        ``copy`` (default True) makes the engine own an immutable snapshot; pass
        ``copy=False`` only if the host will not mutate the array again."""
        state = self.eo if role.upper() == "EO" else self.ir
        src = getattr(state, "source", None) if state else None
        if not isinstance(src, ExternalFrameSource):
            raise RuntimeError(
                f"No external frame source for role {role!r}; "
                "call add_frame_source({role!r}) before submit_frame.")
        return src.submit(image, source_ts=source_ts, copy=copy)

    @property
    def fusion_available(self) -> bool:
        """True only when both EO and IR streams are present and live — fusion is
        hard-gated on this, so a single-sensor session never attempts it."""
        return (self.eo is not None and self.ir is not None
                and self.eo.connected() and self.ir.connected())

    def set_fusion(self, enabled: bool) -> None:
        self._fusion_enabled = enabled
        if not enabled:
            self.registrar.reset()

    def set_classes(self, names: Optional[set]) -> None:
        """Restrict the surfaced detections to these COCO class names (None = all).

        Applied after detection/tracking, so detection recall is unchanged — only
        which classes are displayed, reasoned over and reported is narrowed."""
        from src.detection.coco_classes import normalize
        self.class_filter = normalize(names)

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self.eo is None and self.ir is None:
            raise RuntimeError("No EO or IR source configured")
        modality = "EO+IR" if (self.eo and self.ir) else (
            "EO" if self.eo else "IR")
        self.detector.warmup()
        self.vlm_worker.start()
        self._stop.clear()
        self._running = True
        for s in self._streams():
            s.source.start()
        self._thread = threading.Thread(target=self._loop, name="va-infer",
                                        daemon=True)
        self._thread.start()
        logger.info("Vision Assistant started ({}, detector={})",
                    modality, self.detector.name)

    def stop(self) -> None:
        logger.info("Stopping Vision Assistant…")
        self._stop.set()
        for s in self._streams():
            try:
                s.source.stop()
            except Exception:  # noqa: BLE001
                pass
            s.buffer.close()
        self.vlm_worker.stop()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._running = False
        logger.info("Vision Assistant stopped")

    def _streams(self) -> List[StreamState]:
        return [s for s in (self.eo, self.ir) if s is not None]

    def _primary(self) -> Optional[StreamState]:
        """The reasoning-anchor stream: EO when present, otherwise IR.

        Events, spatial reasoning and summaries are computed on the primary; a
        *secondary* IR stream (when EO is also present) contributes thermal via
        cross-sensor fusion instead of running its own reasoning. Selecting the
        primary by availability — rather than hardcoding EO — is what lets an
        IR-only session produce full intelligence (cross-sensor fusion still
        requires both sensors and is gated separately)."""
        return self.eo if self.eo is not None else self.ir

    # ------------------------------------------------------------- inference
    def _loop(self) -> None:
        try:
            while not self._stop.is_set():
                worked = False
                for s in self._streams():
                    frame = s.buffer.get_latest(timeout=0.05)
                    if frame is None:
                        continue
                    worked = True
                    self._process_stream(s, frame)

                # Cross-sensor fusion on the latest synchronised pair.
                if self._fusion_enabled and self.eo and self.ir \
                        and self.eo.latest_raw is not None \
                        and self.ir.latest_raw is not None:
                    self._fuse_tick += 1
                    if self._fuse_tick % max(1, self.config.cadence.fusion_interval) == 0:
                        self._fuse()
                if not worked:
                    time.sleep(0.01)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Inference loop crashed: {}", exc)
        finally:
            self._running = False

    def _process_stream(self, s: StreamState, frame) -> None:
        prof = self.profiler
        s.modality = frame.modality
        with prof.time("detect"):
            base, sv = self.detector.detect(frame.image)
        with prof.time("track"):
            tracked = s.tracker.update(sv, base)
        with prof.time("reid"):
            tracked = s.reid.assign(tracked, frame.image, frame.frame_id,
                                    frame.timestamp, frame.modality)
        display = merge_detections(base, tracked)
        if self.class_filter is not None:
            tracked = [d for d in tracked if d.class_name in self.class_filter]
            display = [d for d in display if d.class_name in self.class_filter]

        # The primary stream (EO if present, else IR) drives scene reasoning;
        # a secondary IR stream only contributes thermal via fusion.
        if s is self._primary():
            self._primary_reasoning(s, frame, tracked)

        prof.mark("infer")
        prof.mark("infer_eo" if s.modality == Modality.EO else "infer_ir")

        s._fps_n += 1
        if frame.timestamp - s._fps_t >= 1.0:
            s.fps = s._fps_n / max(1e-6, frame.timestamp - s._fps_t)
            s._fps_n = 0
            s._fps_t = frame.timestamp

        # Publish AI output as *data* (no drawing here — the render thread draws it
        # onto the live raw frame). Single atomic ref swap per field.
        overlay = OverlayResult(detections=display, frame_id=frame.frame_id,
                                fps=s.fps, modality=frame.modality)
        with self._lock:
            s.latest_raw = frame.image
            s.latest_overlay = overlay
            s.latest_dets = tracked
            s.frame_id = frame.frame_id
            s.source_ts = frame.source_ts
            s.last_ts = frame.timestamp

    def _primary_reasoning(self, s: StreamState, frame, tracked) -> None:
        prof = self.profiler
        # --- Every frame: motion events (motion-stateful, never sub-rated). ---
        with prof.time("events"):
            new_events = self.events.process(
                FrameResult(frame.frame_id, frame.timestamp, frame.source_ts,
                            tracked, frame.modality), frame.image)
        self.mission.note_frame(tracked, frame.source_ts, image=frame.image)
        self.mission.add_events(new_events)

        # --- Sub-rated: spatial clustering / scene understanding. -------------
        interactions: list = []
        self._eo_tick += 1
        if self._eo_tick % max(1, self.config.cadence.reasoning_interval) == 0:
            with prof.time("spatial"):
                interactions = self.spatial.process(
                    tracked, frame.frame_id, frame.source_ts, frame.image.shape[:2],
                    now=frame.timestamp)
            self.mission.add_interactions(interactions)

        # VLM cueing: on a significant event OR on the periodic cadence. We only
        # copy the frame when we actually submit (the worker gates the final rate).
        now = frame.timestamp
        periodic_due = (now - self._last_periodic) >= self.config.vlm.periodic_interval_sec
        if new_events or interactions or periodic_due:
            accepted = self.vlm_worker.submit(VLMRequest(
                image=frame.image.copy(), modality=frame.modality.value,
                detections=[f"{d.class_name}#{d.track_id}" for d in tracked],
                events=new_events,
                notes=[it.description for it in interactions],
                frame_id=frame.frame_id, source_ts=frame.source_ts,
                timestamp=now))
            if accepted and periodic_due and not new_events:
                self._last_periodic = now

        # Output Layer: emit a structured result on every fresh signal.
        if new_events or interactions:
            self._emit({
                "frame_id": frame.frame_id,
                "source_ts": frame.source_ts,
                "modality": frame.modality.value,
                "detections": [
                    {"bbox": [round(float(v), 1) for v in d.bbox],
                     "class": d.class_name, "track_id": d.track_id,
                     "confidence": round(float(d.confidence), 3)}
                    for d in tracked],
                "events": [{"type": e.type.value, "description": e.description,
                            "track_ids": e.track_ids} for e in new_events],
                "interactions": [{"type": it.type.value,
                                  "description": it.description}
                                 for it in interactions],
                "summary": self.mission.latest_summary,
            })

    def _fuse(self) -> None:
        with self._lock:
            ir_img = None if self.ir.latest_raw is None else self.ir.latest_raw
            eo_dets = list(self.eo.latest_dets)
            ir_dets = list(self.ir.latest_dets)
            eo_size = (self.eo.latest_raw.shape[1], self.eo.latest_raw.shape[0])
            ir_size = (ir_img.shape[1], ir_img.shape[0])
            frame_id = self.eo.frame_id
            source_ts = self.eo.source_ts
        now = time.time()
        prof = self.profiler

        with prof.time("fusion.hot_blobs"):
            blobs = self.fusion.hot_blobs(ir_img)
        # Re-estimate the IR→EO transform on a sub-rate cadence: it changes slowly
        # once locked, and the WARM/lock hold reuses the last fit in between. Always
        # re-estimate while we have no usable lock yet, so acquisition isn't delayed.
        self._reg_tick += 1
        interval = max(1, self.config.registration.update_interval)
        if self._reg_tick % interval == 0 or not self._registration.usable:
            eo_kp = detection_keypoints(eo_dets)
            ir_kp = detection_keypoints(ir_dets) + blob_keypoints(blobs)
            with prof.time("fusion.registration"):
                registration = self.registrar.update(eo_kp, ir_kp, eo_size, ir_size, now)
        else:
            registration = self._registration
        prof.gauge("reg_inliers", registration.inliers)
        prof.gauge("reg_confidence", round(registration.confidence, 3))
        prof.mark("fuse")

        with prof.time("fusion.fuse_pair"):
            assessments = self.fusion.fuse_pair(eo_dets, blobs, registration,
                                                frame_id, source_ts, now)
        if assessments:
            self.mission.add_fusion(assessments)
            self._emit({
                "frame_id": frame_id, "source_ts": source_ts, "modality": "EO+IR",
                "fusion": [{"type": a.type.value, "description": a.description,
                            "position": a.position, "confidence": a.confidence,
                            "track_id": a.global_id} for a in assessments],
                "summary": self.mission.latest_summary,
            })

        # Publish fusion contacts as render *data* (current frame only).
        markers = [{"pos": a.position, "type": a.type.value} for a in assessments]
        contacts = [{"type": a.type.value, "desc": a.description,
                     "pos": a.position, "conf": a.confidence,
                     "gid": a.global_id} for a in assessments]
        with self._lock:
            self._registration = registration
            self._fusion_markers = markers
            if contacts:
                self._fusion_contacts = (contacts + self._fusion_contacts)[:12]

    def _on_vlm(self, result: VLMResult) -> None:
        kind = "alert" if result.request.reason == "event" else "summary"
        self.mission.add_summary(result.text, source_ts=result.request.source_ts,
                                 timestamp=result.timestamp, kind=kind)
        self._emit({
            "frame_id": result.request.frame_id,
            "source_ts": result.request.source_ts,
            "kind": kind, "summary": result.text,
        })

    # -------------------------------------------------------------- polling
    def _cached_gpu_stats(self, max_age: float = 1.0) -> dict:
        """GPU/VRAM telemetry, refreshed at most every ``max_age`` seconds, to keep
        the NVML query off the high-frequency feed path."""
        from src.utils.system_stats import gpu_stats
        now = time.time()
        if not self._gpu_cache or now - self._gpu_ts >= max_age:
            self._gpu_cache = gpu_stats()
            self._gpu_ts = now
        return self._gpu_cache

    def _capture_fps(self, s: Optional[StreamState]) -> float:
        """Real feed rate of a stream — how fast frames are being *captured*
        (independent of the AI/detection rate)."""
        if s is None:
            return 0.0
        now = time.time()
        rec = s.buffer.received
        prev = self._cap_fps.get(s.name)
        if prev is None:
            self._cap_fps[s.name] = (rec, now, 0.0)
            return 0.0
        prec, pnow, pema = prev
        dt = now - pnow
        if dt < 0.2:
            return pema
        inst = (rec - prec) / dt
        ema = inst if pema <= 0 else 0.6 * pema + 0.4 * inst
        self._cap_fps[s.name] = (rec, now, ema)
        return ema

    def _raw_for_display(self, s: Optional[StreamState]) -> Optional[np.ndarray]:
        """Freshest captured frame for a stream — the newest buffered frame
        (decoupled from inference), falling back to the last processed raw."""
        if s is None:
            return None
        latest = s.buffer.peek_latest()
        if latest is not None:
            return latest.image
        return s.latest_raw

    def _compose(self, raw, overlay, fusion_markers=None, registration=None):
        """Composite overlay *data* onto the live raw frame (render thread).

        Falls back to the bare raw frame when no AI overlay exists yet, so the feed
        shows live video the instant capture starts."""
        if raw is None:
            return None
        if overlay is None:
            return raw
        return compose_overlay(
            raw, overlay.detections, overlay.frame_id, overlay.fps,
            overlay.modality, show_conf=self.config.ui.show_confidence,
            show_track_id=self.config.ui.show_track_ids,
            fusion_markers=fusion_markers, registration=registration)

    def frame_view(self) -> FrameView:
        """Cheap view for the live feed — the freshest *raw* frame with the latest
        AI overlay composited on top (which may lag). No inference on the render
        path; display rate is independent of inference."""
        with self.profiler.time("frame_view"):
            with self._lock:
                eo_overlay = self.eo.latest_overlay if self.eo else None
                ir_overlay = self.ir.latest_overlay if self.ir else None
                reg = self._registration
                markers = list(self._fusion_markers)
                contacts = list(self._fusion_contacts)
                eo_fps = self.eo.fps if self.eo else 0.0
                ir_fps = self.ir.fps if self.ir else 0.0
                eo_tracks = len(self.eo.latest_dets) if self.eo else 0
                ir_tracks = len(self.ir.latest_dets) if self.ir else 0
                eo_last_ts = self.eo.last_ts if self.eo else 0.0
                ir_last_ts = self.ir.last_ts if self.ir else 0.0
            eo_raw = self._raw_for_display(self.eo)
            ir_raw = self._raw_for_display(self.ir)
            eo_cap_fps = self._capture_fps(self.eo)
            ir_cap_fps = self._capture_fps(self.ir)
            reg_dict = {"state": reg.state, "scale": round(reg.scale, 3),
                        "rotation_deg": round(reg.rotation_deg, 2),
                        "confidence": round(reg.confidence, 2),
                        "inliers": reg.inliers, "locked": reg.locked}
            fusion_on = self._fusion_enabled
            with self.profiler.time("render_compose"):
                eo_frame = self._compose(
                    eo_raw, eo_overlay,
                    fusion_markers=markers if fusion_on else None,
                    registration=reg_dict if fusion_on else None)
                ir_frame = self._compose(ir_raw, ir_overlay)
        if self.profiler.enabled:
            now = time.time()
            if eo_last_ts:
                self.profiler.gauge("eo_frame_age_ms", (now - eo_last_ts) * 1000.0)
            if ir_last_ts:
                self.profiler.gauge("ir_frame_age_ms", (now - ir_last_ts) * 1000.0)
        return FrameView(
            eo_frame=eo_frame, ir_frame=ir_frame,
            eo_connected=bool(self.eo and self.eo.connected()),
            ir_connected=bool(self.ir and self.ir.connected()),
            eo_fps=eo_fps, ir_fps=ir_fps,
            eo_capture_fps=eo_cap_fps, ir_capture_fps=ir_cap_fps,
            eo_tracks=eo_tracks, ir_tracks=ir_tracks,
            fusion_available=self.fusion_available,
            fusion_enabled=self._fusion_enabled,
            registration=reg_dict,
            fusion_contacts=contacts,
            vlm_busy=self.vlm_worker.busy, running=self._running)

    def overlay_payload(self) -> dict:
        """Compact JSON overlay state (boxes + fusion contacts) for an external
        renderer, e.g. a browser canvas over a hardware-decoded video.

        Per-stream detections in **source-frame pixel coords** plus the frame
        dimensions (so the host scales boxes to its displayed video). The video is
        the host's; this is only the AI annotation, refreshed on a slow cadence so
        the boxes may lag the smooth video (accepted by design)."""
        payload: dict = {"fusion_enabled": self._fusion_enabled}
        with self._lock:
            for key, s in (("eo", self.eo), ("ir", self.ir)):
                if s is None or s.latest_raw is None:
                    continue
                h, w = s.latest_raw.shape[:2]
                payload[key] = {
                    "w": int(w), "h": int(h),
                    "dets": [{
                        "b": [round(float(v), 1) for v in d.bbox],
                        "c": d.class_name,
                        "id": d.track_id,
                    } for d in s.latest_dets],
                }
            markers = self._fusion_markers if self._fusion_enabled else []
            payload["fusion"] = [{
                "p": [round(float(m["pos"][0]), 1), round(float(m["pos"][1]), 1)],
                "t": m.get("type", ""),
            } for m in markers if m.get("pos")]
        return payload

    def panel_view(self) -> PanelView:
        """Heavier intelligence view (intel feed, timeline, stats, GPU)."""
        stream_info = {}
        for s in self._streams():
            if hasattr(s.source, "stream_info"):
                try:
                    stream_info[s.name] = s.source.stream_info()
                except Exception:  # noqa: BLE001
                    pass
        return PanelView(
            intel_messages=self.mission.intel_messages(),
            latest_summary=self.mission.latest_summary,
            timeline=self.mission.timeline_view(),
            stats=self.mission.stats(),
            stream_info=stream_info, gpu=self._cached_gpu_stats())

    def scene_json(self) -> dict:
        """Full JSON-serialisable snapshot of the current scene (Output Layer):
        stats, timeline, intel feed, latest summary, and live registration."""
        reg = self._registration
        data = self.mission.to_dict()
        data["fusion"] = {
            "available": self.fusion_available, "enabled": self._fusion_enabled,
            "registration": {"state": reg.state, "scale": round(reg.scale, 3),
                             "confidence": round(reg.confidence, 2),
                             "inliers": reg.inliers, "locked": reg.locked},
        }
        data["detector"] = self.detector.name
        data["running"] = self._running
        return data

    def mark_display(self) -> None:
        """Called by a render loop each frame so display FPS is measured separately
        from inference FPS (profiling)."""
        self.profiler.mark("display")

    def metrics(self) -> dict:
        """Measured runtime metrics for a profiling dashboard. Cheap and
        thread-safe; returns ``{"enabled": False}`` when profiling is off."""
        if not self.profiler.enabled:
            return {"enabled": False}
        m = self.profiler.metrics()
        streams = {}
        now = time.time()
        for s in self._streams():
            buf = s.buffer
            streams[s.name] = {
                "fps": round(s.fps, 2),
                "tracks": len(s.latest_dets),
                "frame_age_ms": round((now - s.last_ts) * 1000.0, 1) if s.last_ts else None,
                "buffer_size": buf.size,
                "buffer_received": buf.received,
                "buffer_dropped": buf.dropped,
                "connected": s.connected(),
            }
        reg = self._registration
        m["streams"] = streams
        m["vlm"] = self.vlm_worker.stats()
        m["registration"] = {"state": reg.state, "inliers": reg.inliers,
                             "confidence": round(reg.confidence, 3),
                             "scale": round(reg.scale, 3), "locked": reg.locked}
        m["fusion_enabled"] = self._fusion_enabled
        return m

    @property
    def running(self) -> bool:
        return self._running

    def export_scene(self, name_prefix: str = "scene") -> dict:
        """Write a Markdown + JSON snapshot of the scene to the output dir."""
        return self.mission.export_json(
            self.config.abs_path(self.config.system.output_dir),
            name_prefix=name_prefix)
