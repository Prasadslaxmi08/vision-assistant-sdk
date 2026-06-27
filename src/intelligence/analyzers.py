"""Offline analyzers for the Image and Video input modes.

Unlike :class:`~src.engine.vision_assistant.VisionAssistant` (real-time streams,
polled by a host), these run to completion and return a finished artifact:

  * :class:`ImageAnalyzer` — single EO image, single IR image, or an EO+IR pair
    -> structured :class:`ImageReport` (EO analysis, IR analysis, fused).
  * :class:`VideoAnalyzer` — full-file pass that records every event with a media
    timestamp and produces a timeline + final scene summary.

Both reuse the same AI Core (detection / tracking / events / fusion / VLM), so
behaviour is consistent with the live engine. No persistence, no threat scoring —
those moved to the future AI Mission Analyst repo.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from src.config.settings import AppConfig
from src.detection import Detector, build_detector
from src.events.event_manager import EventManager
from src.ingestion.video_reader import VideoReader
from src.intelligence.mission_intelligence import MissionIntelligence
from src.intelligence.report import IntelReport, SensorSection
from src.reasoning import FusionEngine, ObjectReasoningEngine, SpatialReasoner
from src.tracking.reid import Reidentifier
from src.tracking.tracker import ObjectTracker, merge_detections
from src.utils.logger import logger
from src.utils.types import Detection, FrameResult, Modality
from src.vlm import prompts
from src.vlm.qwen_vlm import QwenVLM

# Sentinel for the ``classes`` argument: distinguishes "caller omitted it (use the
# configured default)" from "caller explicitly passed None (report all classes)".
_USE_CONFIG = object()


# ════════════════════════════════════════════════════════════════ Image
class ImageAnalyzer:
    """Modality-aware, detector-grounded still-image analysis.

    Produces a structured :class:`~src.intelligence.report.IntelReport` whose
    sections adapt to the inputs:

      * EO only  -> EO Sensor Analysis + Detected Objects + Scene Assessment
      * IR only  -> IR Sensor Analysis + Thermal Observations + Scene Assessment
      * EO + IR  -> both + Cross-Modal Correlation + Fusion Assessment

    Detection runs first (RF-DETR, tiled for small-object recall, *all* COCO
    classes so thermal contacts aren't filtered out); the VLM then reasons
    **grounded in those detections** rather than describing the image freely.
    Fusion is produced only when both modalities are present.
    """

    def __init__(self, config: AppConfig, vlm: Optional[QwenVLM] = None):
        self.config = config
        # Stills: tile for small-object recall and keep ALL classes (a COCO
        # detector mislabels thermal blobs, and the ISR class filter would drop
        # the very small contacts we must not miss — reasoning reinterprets them).
        det_cfg = config.model_copy(deep=True)
        det_cfg.detection.classes = None
        self.detector: Detector = build_detector(det_cfg, tiled=True)
        self.ore = ObjectReasoningEngine(config.object_reasoning)
        self.vlm = vlm or QwenVLM(config.vlm)
        from src.detection.coco_classes import normalize
        self.default_classes = normalize(config.detection.report_classes)

    # ---------------------------------------------------------- detection
    def _detect(self, image: np.ndarray):
        """Run the (tiled) detector; return (raw_detections, inference_ms)."""
        t0 = time.time()
        raw, _ = self.detector.detect(image)
        return raw, (time.time() - t0) * 1000.0

    # ------------------------------------------------------ per-sensor VLM
    def _analyze_sensor(self, image: np.ndarray, section: SensorSection) -> None:
        """Fill a section's VLM analysis, grounded on the REFINED (reasoned) objects."""
        if not self.config.vlm.enabled:
            section.analysis = "(VLM reasoning disabled — refined detector output only.)"
            return
        confirmed = section.confirmed[: self.config.intelligence.max_vlm_objects]
        text = self.vlm.infer(
            [image],
            prompts.refined_analysis_prompt(
                section.modality, confirmed, section.interactions,
                section.result.scene.scene_type, len(section.possible)),
            max_new_tokens=640)
        secs = prompts.parse_sections(text, ["ANALYSIS", "THERMAL", "OBJECTS"])
        section.analysis = secs.get("ANALYSIS") or secs.get("", text)
        section.thermal_observations = secs.get("THERMAL", "")
        # Attach grounded per-object observations to the confirmed objects.
        notes = prompts.parse_object_notes(secs.get("OBJECTS", ""))
        for i, o in enumerate(confirmed, 1):
            if notes.get(i):
                o.interaction = (o.interaction + "; " if o.interaction else "") + notes[i]

    # ----------------------------------------------------------- entry pt
    def analyze(self, eo_image: Optional[np.ndarray] = None,
                ir_image: Optional[np.ndarray] = None,
                classes=_USE_CONFIG) -> IntelReport:
        """Analyse whatever modalities are supplied. At least one is required.

        Pipeline per sensor: detect (tiled) -> [filter to ``classes`` by name] ->
        Object Reasoning Engine (group / rank / scene-context / interactions /
        confirmed-vs-possible) -> VLM grounded on the refined objects -> report.

        ``classes``: a set of COCO class names to report. Omit it to use the
        configured default (``detection.report_classes``); pass ``None`` to force
        all classes. The filter is applied *after* detection, so small-object recall
        (tiling) is preserved and only the surfaced classes are narrowed.
        """
        if eo_image is None and ir_image is None:
            raise ValueError("Provide at least an EO or an IR image")
        from src.detection.coco_classes import normalize
        keep = self.default_classes if classes is _USE_CONFIG else normalize(classes)

        report = IntelReport()
        agg = {"inference_ms": 0.0, "object_count": 0, "raw_count": 0,
               "_conf_sum": 0.0, "detector": ""}
        for modality, img in (("EO", eo_image), ("IR", ir_image)):
            if img is None:
                continue
            h, w = img.shape[:2]
            raw, ms = self._detect(img)
            if keep is not None:
                raw = [d for d in raw if d.class_name in keep]
            result = self.ore.reason(raw, w, h)
            section = SensorSection(modality=modality, result=result)
            self._analyze_sensor(img, section)
            report.sensors.append(section)
            report.modalities.append(modality)
            refined = result.all_objects()
            agg["inference_ms"] += ms
            agg["raw_count"] += len(raw)
            agg["object_count"] += len(refined)
            agg["_conf_sum"] += sum(o.confidence for o in refined)
            agg["detector"] = self.detector.name
        n = agg["object_count"]
        report.metrics = {"detector": agg["detector"], "object_count": n,
                          "raw_count": agg["raw_count"],
                          "inference_ms": agg["inference_ms"],
                          "mean_confidence": (agg["_conf_sum"] / n) if n else 0.0}
        self._reason_overall(report)
        return report

    def _reason_overall(self, report: IntelReport) -> None:
        """Scene Assessment, plus Cross-Modal + Fusion only when both present."""
        if not self.config.vlm.enabled:
            report.scene_assessment = "(VLM reasoning disabled.)"
            return
        eo = next((s for s in report.sensors if s.modality == "EO"), None)
        ir = next((s for s in report.sensors if s.modality == "IR"), None)
        if report.both_modalities:
            text = self.vlm.infer(
                [], prompts.cross_modal_fusion_prompt(
                    eo.analysis, ir.analysis, eo.confirmed, ir.confirmed),
                max_new_tokens=768)
            secs = prompts.parse_sections(
                text, ["CROSS_MODAL", "FUSION", "INTEREST", "SCENE", "NOTES"])
            report.cross_modal = secs.get("CROSS_MODAL", "")
            fusion = secs.get("FUSION", "")
            interest = secs.get("INTEREST", "")
            report.fusion_assessment = (f"{fusion}\n\n**Interest level:** {interest}"
                                        if interest else fusion)
            report.scene_assessment = secs.get("SCENE", "")
            report.operator_notes = secs.get("NOTES", "")
        else:
            s = report.sensors[0]
            text = self.vlm.infer(
                [], prompts.scene_and_notes_prompt(s.modality, s.analysis, s.confirmed))
            secs = prompts.parse_sections(text, ["SCENE", "NOTES"])
            report.scene_assessment = secs.get("SCENE") or secs.get("", "")
            report.operator_notes = secs.get("NOTES", "")

    # ----------------------------------------------- backward-compat shims
    def analyze_single(self, image: np.ndarray, modality: Modality) -> IntelReport:
        if modality == Modality.IR:
            return self.analyze(ir_image=image)
        return self.analyze(eo_image=image)

    def analyze_pair(self, eo_image: np.ndarray, ir_image: np.ndarray) -> IntelReport:
        return self.analyze(eo_image=eo_image, ir_image=ir_image)

    def detect_and_reason(self, image: np.ndarray):
        """Detect + reason (no VLM): returns (ReasoningResult, raw_detections, ms)."""
        raw, ms = self._detect(image)
        h, w = image.shape[:2]
        return self.ore.reason(raw, w, h), raw, ms


# ════════════════════════════════════════════════════════════════ Video
class VideoAnalyzer:
    """Synchronous full-file analysis producing a timeline + scene summary.

    Single-stream (EO or IR). For an IR file, intra-frame thermal analysis runs
    (the lone-IR fusion path); cross-sensor EO/IR fusion needs two live streams and
    is handled by :class:`VisionAssistant`, not here.
    """

    def __init__(self, config: AppConfig, vlm: Optional[QwenVLM] = None):
        self.config = config
        self.detector: Detector = build_detector(config)
        self.tracker = ObjectTracker(config.tracking)
        self.reid = Reidentifier(config.reid)
        self.events = EventManager(config.events)
        self.spatial = SpatialReasoner(config.spatial)
        self.fusion = FusionEngine(config.fusion)
        self.mission = MissionIntelligence(
            intel_history=200,
            timeline_max=config.intelligence.timeline_max_events)
        self.vlm = vlm or QwenVLM(config.vlm)
        from src.detection.coco_classes import normalize
        self.default_classes = normalize(config.detection.report_classes)

    def analyze(
        self,
        path: str | Path,
        modality: Optional[Modality] = None,
        sample_stride: int = 1,
        progress_cb: Optional[Callable[[float, str], None]] = None,
        annotate_writer: Optional[Callable[[np.ndarray], None]] = None,
        persist: bool = False,
        classes=_USE_CONFIG,
    ) -> dict:
        """Process a whole video file.

        * ``sample_stride`` — process every Nth frame (speed vs. completeness).
        * ``progress_cb(fraction, message)`` — progress hook.
        * ``annotate_writer(frame)`` — optional sink for annotated frames.
        * ``persist`` — also write a Markdown + JSON snapshot to the output dir.

        Returns a dict with the report (markdown), timeline, stats, summary.
        """
        from src.detection.coco_classes import normalize
        from src.utils.visualization import draw_detections
        keep = self.default_classes if classes is _USE_CONFIG else normalize(classes)

        reader = VideoReader(path, modality)
        self.tracker.reset()
        self.reid.reset()
        self.events.reset()
        self.spatial.reset()
        self.fusion.reset()
        self.mission.reset()

        fps = reader.fps or 30.0
        total = reader.frame_count or 0
        vlm_interval_frames = max(1, int(self.config.vlm.periodic_interval_sec * fps))
        last_vlm_frame = -vlm_interval_frames

        processed = 0
        for frame in reader:
            if frame.frame_id % sample_stride != 0:
                continue

            base_dets, sv_dets = self.detector.detect(frame.image)
            tracked = self.tracker.update(sv_dets, base_dets)
            tracked = self.reid.assign(
                tracked, frame.image, frame.frame_id, frame.timestamp,
                frame.modality)
            if keep is not None:
                tracked = [d for d in tracked if d.class_name in keep]
                base_dets = [d for d in base_dets if d.class_name in keep]
            result = FrameResult(
                frame_id=frame.frame_id, timestamp=frame.timestamp,
                source_ts=frame.source_ts, detections=tracked,
                modality=frame.modality,
            )
            new_events = self.events.process(result, frame.image)
            new_interactions = self.spatial.process(
                tracked, frame.frame_id, frame.source_ts, frame.image.shape[:2],
                now=frame.timestamp)
            # Lone-IR intra-frame thermal analysis (no EO to cross-fuse against).
            new_fusions = self.fusion.process(
                frame.image, tracked, frame.frame_id, frame.source_ts,
                frame.modality, now=frame.timestamp)
            self.mission.note_frame(tracked, frame.source_ts, image=frame.image)
            self.mission.add_events(new_events)
            self.mission.add_interactions(new_interactions)
            self.mission.add_fusion(new_fusions)

            # VLM: periodic OR on a significant event (synchronous here).
            trigger = ((frame.frame_id - last_vlm_frame >= vlm_interval_frames)
                       or bool(new_events) or bool(new_interactions)
                       or bool(new_fusions))
            if self.config.vlm.enabled and trigger:
                summary = self.vlm.scene_summary(
                    frame.image, frame.modality.value,
                    [f"{d.class_name}#{d.track_id}" for d in tracked],
                    [e.description for e in new_events]
                    + [it.description for it in new_interactions]
                    + [fa.description for fa in new_fusions],
                )
                self.mission.add_summary(summary, source_ts=frame.source_ts,
                                         timestamp=frame.timestamp,
                                         kind="alert" if new_events else "summary")
                last_vlm_frame = frame.frame_id

            if annotate_writer is not None:
                annotate_writer(draw_detections(
                    frame.image, merge_detections(base_dets, tracked)))

            processed += 1
            if progress_cb and total:
                progress_cb(min(1.0, frame.frame_id / total),
                            f"Frame {frame.frame_id}/{total} — "
                            f"{len(tracked)} tracks")

        reader.release()

        # Final overall scene summary from the accumulated timeline.
        final_summary = ""
        if self.config.vlm.enabled:
            timeline_texts = [f"{e['time']} {e['detail']}"
                              for e in self.mission.timeline_view()
                              if e["kind"] == "event"]
            final_summary = self.vlm.mission_summary(
                None, timeline_texts[: self.config.intelligence.timeline_max_events],
                self.mission.detections_summary(),
            )

        report_md = self.mission.build_mission_report(
            title="Video Scene Intelligence Report", final_summary=final_summary)

        paths = {}
        if persist:
            paths = self.mission.export_json(
                self.config.abs_path(self.config.system.output_dir),
                name_prefix=Path(str(path)).stem, final_summary=final_summary)

        logger.info("Video analysis complete ({} frames processed)", processed)
        return {
            "report_markdown": report_md,
            "final_summary": final_summary,
            "timeline": self.mission.timeline_view(),
            "stats": self.mission.stats(),
            "frames_processed": processed,
            "report_paths": paths,
        }
