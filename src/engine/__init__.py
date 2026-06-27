"""Vision Assistant SDK — the engine surface external apps integrate against.

    from src.engine import VisionAssistant
    va = VisionAssistant(get_config())
    va.add_source("EO", "rtsp://cam/eo")
    va.add_source("IR", "rtsp://cam/ir")
    va.on_result(lambda r: print(r["summary"]))   # Output Layer: callbacks
    va.start()
    frame = va.frame_view().eo_frame              # overlay-composited live frame
    va.stop()

The engine wraps the AI Core (detector → ByteTrack → EO/IR fusion → VLM) and the
Intelligence Layer (events, scene understanding, summaries), and exposes results
through JSON (:meth:`VisionAssistant.scene_json`), callbacks (:meth:`on_result`),
and overlay rendering (:meth:`frame_view` / :meth:`overlay_payload`). Offline image
and video analysis use the same AI Core via :class:`ImageAnalyzer` /
:class:`VideoAnalyzer`.
"""
from src.engine.vision_assistant import (FrameView, OverlayResult, PanelView,
                                          VisionAssistant)
from src.intelligence.analyzers import ImageAnalyzer, VideoAnalyzer

__all__ = ["VisionAssistant", "FrameView", "PanelView", "OverlayResult",
           "ImageAnalyzer", "VideoAnalyzer"]
