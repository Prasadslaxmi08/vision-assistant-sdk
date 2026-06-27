"""Minimal Tkinter "GCS" that embeds the Vision Assistant SDK.

This is a *reference integration*: a tiny stand-in for a real Ground Control
Station / VMS / surveillance app. It owns its own camera (OpenCV capture) exactly
like a real host would, and embeds the SDK as a pure AI layer behind **one button**.

It mirrors, in ~150 lines, the same pattern used to embed the SDK into the real
LUMIRA GCS:

    capture thread  ──►  _ingest(frame)        # host owns capture
                          • (AI on?) va.submit_frame("EO", frame)   ◄── push frame in
                          • build a PIL display image
                          • (AI on?) draw va.overlay_payload() boxes ◄── draw result
    Tk main thread  ──►  _render_tick()         # paint the latest image on a label

Everything SDK-specific is fenced by ``# ── SDK ──`` comments below. Note how few
lines they are — initialise, push frames, read overlays, stop. The host never
touches the detector, tracker, fusion engine, or VLM.

Run:
    python examples/tkinter_gcs/mini_gcs.py            # default webcam (source 0)
    python examples/tkinter_gcs/mini_gcs.py rtsp://cam/eo
    python examples/tkinter_gcs/mini_gcs.py path/to/clip.mp4

Tune for your GPU (optional):
    set VISION_ASSISTANT_BACKEND=yolo11   # default here; ships with the SDK
    set VISION_ASSISTANT_VLM=0            # skip the VLM for a lighter demo
"""
from __future__ import annotations

import os
import sys
import threading
import time
import tkinter as tk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

# Make the SDK importable when running straight from the repo. A real host would
# instead `pip install -e .` the SDK (see pyproject.toml) and skip this.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ── SDK ── the only imports a host needs ────────────────────────────────────
from src.config.settings import get_config          # noqa: E402
from src.engine import VisionAssistant               # noqa: E402
# ────────────────────────────────────────────────────────────────────────────

_BOX = (0, 229, 200)
_TXT = (4, 12, 14)
_INTEL = (120, 235, 215)


class MiniGCS(tk.Tk):
    def __init__(self, source: str):
        super().__init__()
        self.title("Mini-GCS — Vision Assistant SDK integration demo")
        self.configure(bg="#040c0e")
        self.geometry("1000x640")
        self.source = source

        self.video = tk.Label(self, bg="#040c0e", fg=_INTEL,
                              text="Press  Start feed", font=("Segoe UI", 14))
        self.video.pack(fill="both", expand=True)

        bar = tk.Frame(self, bg="#0a181b")
        bar.pack(fill="x")
        self.btn_feed = tk.Button(bar, text="▶ Start feed", command=self._toggle_feed,
                                  width=14, relief="flat", bg="#16323a", fg="#cfeee9")
        self.btn_feed.pack(side="left", padx=8, pady=8)
        # ── SDK ── the one AI button the host adds ──────────────────────────
        self.btn_ai = tk.Button(bar, text="◬ Vision AI: OFF", command=self._toggle_ai,
                                width=18, relief="flat", bg="#16323a", fg="#cfeee9")
        self.btn_ai.pack(side="left", padx=8, pady=8)
        # ────────────────────────────────────────────────────────────────────
        self.status = tk.Label(bar, text="idle", bg="#0a181b", fg="#7be1d2",
                               font=("Segoe UI", 10))
        self.status.pack(side="right", padx=12)

        self._cap = None
        self._cap_thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None            # latest PIL image for the render tick
        self._geom = None              # (off_x, off_y, draw_w, draw_h)
        self._native = (0, 0)

        # ── SDK ── engine handle + the intel it streams back ────────────────
        self.va = None                 # the VisionAssistant engine (lazy)
        self._ai_state = "off"         # off | loading | on
        self._summary = ""
        # ────────────────────────────────────────────────────────────────────

        try:
            self._font = ImageFont.truetype("segoeui.ttf", 15)
            self._font_sm = ImageFont.truetype("segoeui.ttf", 12)
        except Exception:
            self._font = self._font_sm = ImageFont.load_default()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(33, self._render_tick)

    # ----------------------------------------------------------- feed (host)
    def _toggle_feed(self) -> None:
        if self._cap_thread is None:
            self._stop.clear()
            self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._cap_thread.start()
            self.btn_feed.config(text="■ Stop feed")
        else:
            self._shutdown_feed()
            self.btn_feed.config(text="▶ Start feed")

    def _capture_loop(self) -> None:
        src = int(self.source) if self.source.isdigit() else self.source
        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            self._set_status(f"cannot open source: {self.source}")
            self._cap_thread = None
            return
        self._set_status("feed live")
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if isinstance(src, str) and not str(src).isdigit():
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # loop video files
                    continue
                break
            self._ingest(frame)
        self._cap.release()

    def _ingest(self, frame: np.ndarray) -> None:
        """Capture thread: this is the integration seam (mirrors LUMIRA)."""
        # ── SDK ── push the frame the host already decoded into the engine ──
        if self._ai_state == "on" and self.va is not None:
            self.va.submit_frame("EO", frame)
        # ────────────────────────────────────────────────────────────────────
        pil = self._build_display(frame)
        if pil is not None:
            with self._lock:
                self._latest = pil

    def _build_display(self, frame: np.ndarray):
        w = self.video.winfo_width()
        h = self.video.winfo_height()
        if w < 16 or h < 16:
            return None
        fh, fw = frame.shape[:2]
        self._native = (fw, fh)
        scale = min(w / fw, h / fh)
        dw, dh = max(1, int(fw * scale)), max(1, int(fh * scale))
        ox, oy = (w - dw) // 2, (h - dh) // 2
        boxed = np.zeros((h, w, 3), np.uint8)
        boxed[oy:oy + dh, ox:ox + dw] = cv2.resize(frame, (dw, dh))
        self._geom = (ox, oy, dw, dh)
        img = Image.fromarray(cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB))
        # ── SDK ── read the AI result (data, not pixels) and draw it ────────
        if self._ai_state == "on" and self.va is not None:
            self._draw_ai(img, (ox, oy, dw, dh))
        # ────────────────────────────────────────────────────────────────────
        return img

    def _draw_ai(self, img: Image.Image, geom) -> None:
        ox, oy, dw, dh = geom
        try:
            payload = self.va.overlay_payload()
        except Exception:
            return
        draw = ImageDraw.Draw(img, "RGBA")
        eo = payload.get("eo")
        if eo and eo.get("dets"):
            sx = dw / max(1, eo["w"])
            sy = dh / max(1, eo["h"])
            for d in eo["dets"]:
                b = d["b"]
                x1, y1 = ox + b[0] * sx, oy + b[1] * sy
                x2, y2 = ox + b[2] * sx, oy + b[3] * sy
                draw.rectangle((x1, y1, x2, y2), outline=_BOX, width=2)
                label = d.get("c", "")
                if d.get("id") is not None:
                    label += f"#{d['id']}"
                tw = draw.textlength(label, font=self._font_sm)
                draw.rectangle((x1, y1 - 16, x1 + tw + 8, y1), fill=_BOX)
                draw.text((x1 + 4, y1 - 15), label, fill=_TXT, font=self._font_sm)
        draw.text((ox + 12, oy + 10), "◬ VISION AI", fill=_BOX, font=self._font_sm)
        line = (self._summary or "").replace("\n", " ").strip()
        if line:
            if len(line) > 110:
                line = line[:107] + "…"
            draw.rectangle((ox, oy + dh - 30, ox + dw, oy + dh), fill=(8, 16, 18, 180))
            draw.text((ox + 12, oy + dh - 23), line, fill=_INTEL, font=self._font)

    # ----------------------------------------------------------- AI (SDK)
    def _toggle_ai(self) -> None:
        if self._cap_thread is None:
            self._set_status("start the feed first")
            return
        if self._ai_state == "off":
            self._ai_state = "loading"
            self.btn_ai.config(text="◬ Vision AI: loading…")
            threading.Thread(target=self._start_ai, daemon=True).start()
        else:
            self._stop_ai()

    def _start_ai(self) -> None:
        try:
            # ── SDK ── initialise the engine for an externally-fed stream ───
            cfg = get_config()
            cfg.detection.backend = os.environ.get("VISION_ASSISTANT_BACKEND", "yolo11")
            if os.environ.get("VISION_ASSISTANT_VLM") in ("0", "false"):
                cfg.vlm.enabled = False
            va = VisionAssistant(cfg)
            va.add_frame_source("EO")               # a *push* source — host feeds it
            va.on_result(self._on_result)           # Output-Layer callback
            va.start()
            # ────────────────────────────────────────────────────────────────
            self.va = va
            self._ai_state = "on"
            self.after(0, lambda: self.btn_ai.config(text="◬ Vision AI: ON"))
            self._set_status("Vision AI online")
        except Exception as exc:
            self._ai_state = "off"
            self.va = None
            self.after(0, lambda: self.btn_ai.config(text="◬ Vision AI: OFF"))
            self._set_status(f"AI failed: {exc}")

    def _on_result(self, r: dict) -> None:
        # ── SDK ── structured AI output (runs on the engine's thread) ───────
        if r.get("summary"):
            self._summary = r["summary"]
        # ────────────────────────────────────────────────────────────────────

    def _stop_ai(self) -> None:
        va, self.va = self.va, None
        self._ai_state = "off"
        self._summary = ""
        self.btn_ai.config(text="◬ Vision AI: OFF")
        self._set_status("Vision AI off")
        if va is not None:
            threading.Thread(target=lambda: self._safe(va.stop), daemon=True).start()

    # ----------------------------------------------------------- render (Tk)
    def _render_tick(self) -> None:
        with self._lock:
            pil = self._latest
            self._latest = None
        if pil is not None:
            tkimg = ImageTk.PhotoImage(pil)
            self.video.imgtk = tkimg
            self.video.config(image=tkimg, text="")
        self.after(33, self._render_tick)

    # ----------------------------------------------------------- teardown
    def _shutdown_feed(self) -> None:
        self._stop.set()
        if self._cap_thread is not None:
            self._cap_thread.join(timeout=2.0)
        self._cap_thread = None

    def _on_close(self) -> None:
        self._stop_ai()
        self._shutdown_feed()
        self.destroy()

    @staticmethod
    def _safe(fn) -> None:
        try:
            fn()
        except Exception:
            pass

    def _set_status(self, msg: str) -> None:
        self.after(0, lambda: self.status.config(text=msg))


if __name__ == "__main__":
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    MiniGCS(source).mainloop()
