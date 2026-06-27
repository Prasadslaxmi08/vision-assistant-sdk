"""Headless CLI runner for the Vision Assistant SDK.

Drives the engine without a GUI — useful for testing the AI Core on the target
box (RTSP/video -> RF-DETR -> ByteTrack -> EO/IR fusion -> Qwen2.5-VL) or running
it as a background service that emits results over a callback.

Examples:
    # Single EO RTSP stream, 60 s, print the JSON result stream
    python run_cli.py --eo rtsp://user:pass@host:554/eo --duration 60 --json

    # Dual EO + IR streams with fusion enabled
    python run_cli.py --eo rtsp://host/eo --ir rtsp://host/ir --fusion

    # A local video treated as IR
    python run_cli.py --eo clip.mp4 --video --modality IR

    # Default webcam, no preview window
    python run_cli.py --eo 0 --no-display
"""
from __future__ import annotations

import argparse
import json
import time

import cv2

from src.config.settings import get_config
from src.engine import VisionAssistant
from src.utils.logger import logger, setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vision Assistant — CLI")
    p.add_argument("--eo", help="EO source: RTSP URL, video path, or webcam index")
    p.add_argument("--ir", help="IR source (enables EO+IR fusion when both given)")
    p.add_argument("--modality", choices=["EO", "IR"], default="EO",
                   help="Modality of a single --eo source (default EO)")
    p.add_argument("--video", action="store_true",
                   help="Treat sources as video files (not live streams)")
    p.add_argument("--fusion", action="store_true",
                   help="Enable EO/IR fusion (requires both --eo and --ir)")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Seconds to run (0 = until source ends / Ctrl-C)")
    p.add_argument("--no-display", action="store_true",
                   help="Disable the OpenCV preview window")
    p.add_argument("--json", action="store_true",
                   help="Print the JSON result stream to stdout")
    p.add_argument("--classes",
                   help="Comma-separated COCO class names to report (e.g. "
                        "'person,car,boat'). Overrides detection.report_classes; "
                        "omit to use the config default.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.eo and not args.ir:
        raise SystemExit("Provide at least --eo or --ir")
    cfg = get_config()
    setup_logging(cfg.abs_path(cfg.system.log_dir), cfg.system.log_level)

    va = VisionAssistant(cfg)   # class filter defaults from detection.report_classes
    if args.classes:            # explicit override
        va.set_classes({c.strip() for c in args.classes.split(",") if c.strip()})
    if args.json:
        va.on_result(lambda r: print(json.dumps(r, default=str)))

    # A lone --eo with --modality IR is an IR-only session.
    if args.eo and args.modality == "IR" and not args.ir:
        va.add_source("IR", args.eo, is_video=args.video)
    else:
        if args.eo:
            va.add_source("EO", args.eo, is_video=args.video)
        if args.ir:
            va.add_source("IR", args.ir, is_video=args.video)
    if args.fusion:
        va.set_fusion(True)

    va.start()
    logger.info("Running. Press Ctrl-C (or 'q' in the window) to stop.")
    start = time.time()
    try:
        while va.running:
            view = va.frame_view()
            frame = view.eo_frame if view.eo_frame is not None else view.ir_frame
            if not args.no_display and frame is not None:
                cv2.imshow("Vision Assistant", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if args.duration and (time.time() - start) >= args.duration:
                break
            time.sleep(0.03)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        va.stop()
        if not args.no_display:
            cv2.destroyAllWindows()
        summary = va.panel_view().latest_summary
        if summary:
            logger.info("Latest scene summary: {}", summary)


if __name__ == "__main__":
    main()
