"""Vision Assistant — PySide6 demo (single window).

A lightweight demonstration interface for the Vision Assistant SDK. It shows the
three input modes the engine supports:

  * **Live**  — one or two independent RTSP streams (EO and/or IR); fusion is
    enabled only when both are present.
  * **Video** — an EO or IR video file (full-file analysis).
  * **Image** — a single EO/IR image, or an EO+IR pair (fused assessment).

The window never blocks on AI: in Live mode a fast timer pulls the latest
overlay-composited frame (decoupled from inference), and a slow timer pulls the
scene summary. Heavy video/image analysis runs on a worker thread.

Run with:  ``python -m desktop``   (or ``python desktop/app.py``)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project root importable when run as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import Qt, QThread, QTimer, Signal  # noqa: E402
from PySide6.QtGui import QImage, QPixmap  # noqa: E402
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox,  # noqa: E402
                               QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMainWindow, QProgressBar,
                               QPushButton, QStackedWidget, QTextEdit,
                               QVBoxLayout, QWidget)

from desktop.class_selector import ClassSelector  # noqa: E402
from src.config.settings import get_config  # noqa: E402
from src.utils.logger import setup_logging  # noqa: E402
from src.utils.types import Modality  # noqa: E402


# ───────────────────────────────────────────────────────── helpers
def bgr_to_qimage(frame: np.ndarray) -> QImage:
    """Convert a BGR uint8 OpenCV frame to a (copied) QImage."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def set_frame(label: QLabel, frame: np.ndarray | None) -> None:
    if frame is None:
        return
    pix = QPixmap.fromImage(bgr_to_qimage(frame))
    label.setPixmap(pix.scaled(label.width(), label.height(),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation))


def _shared_vlm(cfg):
    """One Qwen VLM per process (the 7B/3B weights load once)."""
    global _VLM
    try:
        return _VLM
    except NameError:
        from src.vlm.qwen_vlm import QwenVLM
        globals()["_VLM"] = QwenVLM(cfg.vlm)
        return globals()["_VLM"]


# ───────────────────────────────────────────────────────── worker threads
class StartEngineWorker(QThread):
    """Builds the engine (detector + VLM — which may download weights on first run)
    and starts it, entirely off the UI thread so the window never freezes."""
    started_ok = Signal(object)   # carries the built, running engine
    failed = Signal(str)

    def __init__(self, factory):
        super().__init__()
        self._factory = factory   # () -> a started VisionAssistant

    def run(self) -> None:
        try:
            self.started_ok.emit(self._factory())
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class AnalysisWorker(QThread):
    """Runs offline image/video analysis off the UI thread.

    ``done`` carries either a dict (video result) or a str (image markdown), so it
    is typed ``object``. ``fn`` receives a ``progress(fraction, message)`` callable
    it may call to drive the progress bar (image jobs just ignore it)."""
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self) -> None:
        try:
            self.done.emit(self._fn(
                lambda frac, msg="": self.progress.emit(int(frac * 100), msg)))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


# ───────────────────────────────────────────────────────── main window
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = get_config()
        setup_logging(self.cfg.abs_path(self.cfg.system.log_dir),
                      self.cfg.system.log_level)
        self.setWindowTitle("Vision Assistant — SDK Demo")
        self.resize(1180, 720)

        self.engine = None          # live VisionAssistant when running
        self._start_worker = None
        self._analysis_worker = None

        self._build_ui()

        # Timers: fast feed (decoupled from inference) + slow panel refresh.
        self._feed_timer = QTimer(self)
        self._feed_timer.timeout.connect(self._refresh_feed)
        self._panel_timer = QTimer(self)
        self._panel_timer.timeout.connect(self._refresh_panel)

    # -- layout ---------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left: controls.
        controls = QVBoxLayout()
        mode_box = QGroupBox("Input mode")
        mb = QVBoxLayout(mode_box)
        self.mode = QComboBox()
        self.mode.addItems(["Live (RTSP)", "Video file", "Image"])
        self.mode.currentIndexChanged.connect(self._on_mode_changed)
        mb.addWidget(self.mode)
        # VLM is opt-in: the language model is large and its FIRST load takes a
        # few minutes (and holds Python's GIL, briefly pausing the UI). Leaving it
        # off keeps detection / tracking / fusion instant and the window snappy.
        self.vlm_chk = QCheckBox("Run VLM reasoning (large model; first load is slow)")
        mb.addWidget(self.vlm_chk)
        controls.addWidget(mode_box)

        cls_box = QGroupBox("Detection classes — what to detect")
        cbl = QVBoxLayout(cls_box)
        from src.detection.coco_classes import normalize
        self.class_selector = ClassSelector(
            initial=normalize(self.cfg.detection.report_classes))
        self.class_selector.changed.connect(self._on_classes_changed)
        cbl.addWidget(self.class_selector)
        controls.addWidget(cls_box)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._live_panel())
        self.stack.addWidget(self._video_panel())
        self.stack.addWidget(self._image_panel())
        controls.addWidget(self.stack)
        controls.addStretch(1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        controls.addWidget(self.progress)
        self.status = QLabel("Ready.")
        self.status.setWordWrap(True)
        controls.addWidget(self.status)
        left = QWidget()
        left.setLayout(controls)
        left.setFixedWidth(330)
        root.addWidget(left)

        # Center/right: displays + summary.
        right = QVBoxLayout()
        feeds = QHBoxLayout()
        self.eo_view = QLabel("EO")
        self.eo_view.setAlignment(Qt.AlignCenter)
        self.eo_view.setMinimumSize(640, 380)
        self.eo_view.setStyleSheet("background:#111;color:#888;border:1px solid #333")
        self.ir_view = QLabel("IR")
        self.ir_view.setAlignment(Qt.AlignCenter)
        self.ir_view.setMinimumSize(300, 380)
        self.ir_view.setStyleSheet("background:#111;color:#888;border:1px solid #333")
        feeds.addWidget(self.eo_view, 2)
        feeds.addWidget(self.ir_view, 1)
        right.addLayout(feeds, 3)

        self.info = QLabel("—")
        right.addWidget(self.info)
        right.addWidget(QLabel("Scene summary / detections"))
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        right.addWidget(self.summary, 2)
        root.addLayout(right, 1)

    def _live_panel(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.eo_url = QLineEdit("rtsp://CAMERA_IP:8554/eo")
        self.ir_url = QLineEdit("rtsp://CAMERA_IP:8554/ir")
        self.fusion_chk = QCheckBox("Enable EO/IR fusion (needs both streams)")
        f.addRow("EO RTSP:", self.eo_url)
        f.addRow("IR RTSP:", self.ir_url)
        f.addRow(self.fusion_chk)
        self.live_btn = QPushButton("Start")
        self.live_btn.clicked.connect(self._toggle_live)
        f.addRow(self.live_btn)
        return w

    def _video_panel(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.video_path = QLineEdit()
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self._pick_file(self.video_path,
                               "Videos (*.mp4 *.avi *.mov *.mkv)"))
        row = QHBoxLayout()
        row.addWidget(self.video_path)
        row.addWidget(browse)
        roww = QWidget()
        roww.setLayout(row)
        self.video_modality = QComboBox()
        self.video_modality.addItems(["EO", "IR"])
        f.addRow("Video:", roww)
        f.addRow("Modality:", self.video_modality)
        self.video_btn = QPushButton("Analyze video")
        self.video_btn.clicked.connect(self._analyze_video)
        f.addRow(self.video_btn)
        return w

    def _image_panel(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.eo_img = QLineEdit()
        self.ir_img = QLineEdit()
        eo_b = QPushButton("Browse…")
        eo_b.clicked.connect(lambda: self._pick_file(self.eo_img,
                             "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"))
        ir_b = QPushButton("Browse…")
        ir_b.clicked.connect(lambda: self._pick_file(self.ir_img,
                             "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"))
        for lbl, le, b in (("EO image:", self.eo_img, eo_b),
                           ("IR image:", self.ir_img, ir_b)):
            row = QHBoxLayout()
            row.addWidget(le)
            row.addWidget(b)
            rw = QWidget()
            rw.setLayout(row)
            f.addRow(lbl, rw)
        self.image_modality = QComboBox()
        self.image_modality.addItems(["EO", "IR"])
        f.addRow("Single modality:", self.image_modality)
        f.addRow(QLabel("EO + IR → fused report; one image → that sensor only."))
        self.boxes_chk = QCheckBox("Show detection boxes")
        self.boxes_chk.setChecked(True)
        self.boxes_chk.stateChanged.connect(self._redraw_image)
        f.addRow(self.boxes_chk)
        self.devmode_chk = QCheckBox("Developer mode: raw detector output")
        self.devmode_chk.stateChanged.connect(self._redraw_image)
        f.addRow(self.devmode_chk)
        self.image_btn = QPushButton("Analyze image(s)")
        self.image_btn.clicked.connect(self._analyze_image)
        f.addRow(self.image_btn)
        return w

    # -- helpers --------------------------------------------------------
    def _on_mode_changed(self, idx: int) -> None:
        self.stack.setCurrentIndex(idx)

    def _on_classes_changed(self) -> None:
        # Live re-filtering: apply the new class selection to a running engine.
        if self.engine is not None:
            self.engine.set_classes(self.class_selector.selected_classes())

    def _pick_file(self, target: QLineEdit, filt: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select file", "", filt)
        if path:
            target.setText(path)

    def _set_status(self, msg: str) -> None:
        self.status.setText(msg)

    def _run_cfg(self):
        """A per-run config copy whose VLM is enabled only if the box is ticked."""
        cfg = self.cfg.model_copy(deep=True)
        cfg.vlm.enabled = self.vlm_chk.isChecked()
        return cfg

    def _on_progress(self, pct: int, msg: str) -> None:
        self.progress.setVisible(True)
        self.progress.setValue(pct)
        if msg:
            self._set_status(msg)

    # -- live mode ------------------------------------------------------
    def _toggle_live(self) -> None:
        if self.engine is not None:
            self._stop_live()
            return
        if self._start_worker is not None and self._start_worker.isRunning():
            return                               # build already in progress
        eo, ir = self.eo_url.text().strip(), self.ir_url.text().strip()
        if not eo and not ir:
            self._set_status("Enter at least one RTSP URL.")
            return
        fusion = self.fusion_chk.isChecked() and bool(eo) and bool(ir)
        classes = self.class_selector.selected_classes()
        cfg = self._run_cfg()

        def build_and_start():
            # Runs on the worker thread — building the detector may download
            # weights on first use, so this must not touch the UI thread.
            from src.engine import VisionAssistant
            eng = VisionAssistant(cfg, shared_vlm=_shared_vlm(cfg))
            if eo:
                eng.add_source("EO", eo)
            if ir:
                eng.add_source("IR", ir)
            if fusion:
                eng.set_fusion(True)
            eng.set_classes(classes)
            eng.on_result(self._on_engine_result)
            eng.start()
            return eng

        self.live_btn.setEnabled(False)
        self._set_status("Loading detector…" + (" + VLM (first load is slow)"
                         if cfg.vlm.enabled else " (VLM off — detection/tracking/fusion)"))
        self._start_worker = StartEngineWorker(build_and_start)
        self._start_worker.started_ok.connect(self._on_live_started)
        self._start_worker.failed.connect(self._on_live_failed)
        self._start_worker.start()

    def _on_live_started(self, engine) -> None:
        self.engine = engine
        self.live_btn.setText("Stop")
        self.live_btn.setEnabled(True)
        self._set_status("Live. Feed is decoupled from AI inference.")
        self._feed_timer.start(int(self.cfg.ui.feed_refresh_sec * 1000) or 33)
        self._panel_timer.start(700)

    def _on_live_failed(self, msg: str) -> None:
        self._set_status(f"Failed to start: {msg}")
        self.live_btn.setEnabled(True)
        self.live_btn.setText("Start")
        self.engine = None

    def _stop_live(self) -> None:
        self._feed_timer.stop()
        self._panel_timer.stop()
        try:
            self.engine.stop()
        except Exception:  # noqa: BLE001
            pass
        self.engine = None
        self.live_btn.setText("Start")
        self._set_status("Stopped.")

    def _refresh_feed(self) -> None:
        if self.engine is None:
            return
        view = self.engine.frame_view()
        set_frame(self.eo_view, view.eo_frame)
        set_frame(self.ir_view, view.ir_frame)
        self.engine.mark_display()
        reg = view.registration
        fus = ("fusion " + ("ON" if view.fusion_enabled else "off")
               + (f" · {reg.get('state', '')} conf={reg.get('confidence', 0)}"
                  if view.fusion_enabled else ""))
        self.info.setText(
            f"EO {view.eo_capture_fps:.0f} fps cap / {view.eo_fps:.0f} fps AI · "
            f"{view.eo_tracks} tracks   |   IR {view.ir_tracks} tracks   |   {fus}"
            + ("   · VLM…" if view.vlm_busy else ""))

    def _refresh_panel(self) -> None:
        if self.engine is None:
            return
        panel = self.engine.panel_view()
        lines = [panel.latest_summary, ""] if panel.latest_summary else []
        lines += panel.intel_messages[-10:]
        self.summary.setPlainText("\n".join(lines))

    def _on_engine_result(self, result: dict) -> None:
        # Output Layer callback — here we just surface a one-line status.
        if result.get("summary"):
            self._set_status("New scene summary received.")

    # -- offline analysis ----------------------------------------------
    def _analyze_video(self) -> None:
        path = self.video_path.text().strip()
        if not path:
            self._set_status("Pick a video file.")
            return
        mod = Modality(self.video_modality.currentText())
        cfg = self._run_cfg()
        vlm_on = cfg.vlm.enabled
        from src.engine import VideoAnalyzer
        analyzer = VideoAnalyzer(cfg, vlm=_shared_vlm(cfg) if vlm_on else None)

        classes = self.class_selector.selected_classes()

        def job(progress):
            # Stride 3 keeps the demo responsive; full-rate is available via the SDK.
            return analyzer.analyze(path, modality=mod, sample_stride=3,
                                    progress_cb=progress, classes=classes)

        self.video_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self._set_status("Analyzing video…" + ("" if vlm_on else " (detection only — "
                         "tick 'Run VLM reasoning' for scene summaries)"))
        self._analysis_worker = AnalysisWorker(job)
        self._analysis_worker.progress.connect(self._on_progress)
        self._analysis_worker.done.connect(self._on_video_done)
        self._analysis_worker.failed.connect(self._on_analysis_failed)
        self._analysis_worker.start()

    def _on_video_done(self, out: dict) -> None:
        self.video_btn.setEnabled(True)
        self.progress.setVisible(False)
        self._set_status(f"Done — {out.get('frames_processed', 0)} frames.")
        self.summary.setPlainText(out.get("report_markdown", ""))

    def _analyze_image(self) -> None:
        eo_p, ir_p = self.eo_img.text().strip(), self.ir_img.text().strip()
        if not eo_p and not ir_p:
            self._set_status("Pick an EO and/or IR image.")
            return
        cfg = self._run_cfg()
        vlm_on = cfg.vlm.enabled
        single_mod = self.image_modality.currentText()
        classes = self.class_selector.selected_classes()
        from src.engine import ImageAnalyzer
        analyzer = ImageAnalyzer(cfg, vlm=_shared_vlm(cfg) if vlm_on else None)

        def job(progress):
            eo_img = cv2.imread(eo_p) if eo_p else None
            ir_img = cv2.imread(ir_p) if ir_p else None
            # A lone image tagged IR via the single-modality selector → IR slot.
            if eo_img is not None and ir_img is None and single_mod == "IR":
                eo_img, ir_img = None, eo_img
            report = analyzer.analyze(eo_image=eo_img, ir_image=ir_img, classes=classes)
            return {"report": report, "EO": eo_img, "IR": ir_img}

        set_frame(self.eo_view, cv2.imread(eo_p) if eo_p else cv2.imread(ir_p))
        self.image_btn.setEnabled(False)
        self._set_status("Detecting (RF-DETR, tiled)…"
                         + ("  then VLM reasoning…" if vlm_on else " — detection only"))
        self._analysis_worker = AnalysisWorker(job)
        self._analysis_worker.done.connect(self._on_image_done)
        self._analysis_worker.failed.connect(self._on_analysis_failed)
        self._analysis_worker.start()

    def _on_image_done(self, out: dict) -> None:
        self.image_btn.setEnabled(True)
        report = out["report"]
        # Per modality keep (image, refined objects, raw detections) so the
        # box / developer toggles can redraw instantly without re-running.
        self._img_views = []
        for s in report.sensors:
            img = out.get(s.modality)
            if img is None:
                continue
            refined = list(s.confirmed) + list(s.possible)
            raw = list(s.result.raw) if s.result else []
            self._img_views.append((s.modality, img, refined, raw))
        self._refined_md = report.to_markdown()
        self._raw_md = self._build_raw_md()
        m = report.metrics
        self.info.setText(
            f"Detector: {m.get('detector', '?')}   |   raw: {m.get('raw_count', 0)} "
            f"-> refined: {m.get('object_count', 0)}   |   "
            f"inference: {m.get('inference_ms', 0):.0f} ms   |   "
            f"mean conf: {m.get('mean_confidence', 0):.2f}")
        self._set_status(f"Analysis complete — {m.get('raw_count', 0)} raw -> "
                         f"{m.get('object_count', 0)} refined objects, "
                         f"{' + '.join(report.modalities)}.")
        self._redraw_image()

    def _build_raw_md(self) -> str:
        lines = ["# Developer — Raw Detector Output (pre-reasoning)", ""]
        for mod, _img, _refined, raw in getattr(self, "_img_views", []):
            lines.append(f"## {mod} — {len(raw)} raw detections")
            lines += [f"- {d.class_name} ({d.confidence:.2f})" for d in raw] or ["- none"]
            lines.append("")
        return "\n".join(lines)

    def _draw_objs(self, img: np.ndarray, objs, raw: bool = False) -> np.ndarray:
        """Boxes: raw=grey echo; refined green=confident, red=low-confidence."""
        out = img.copy()
        for o in objs:
            x1, y1, x2, y2 = (int(v) for v in o.bbox)
            if raw:
                col = (150, 150, 150)
            else:
                col = (0, 0, 255) if getattr(o, "uncertain", False) else (0, 200, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            cv2.putText(out, f"{o.class_name} {o.confidence:.2f}",
                        (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        col, 1, cv2.LINE_AA)
        return out

    def _redraw_image(self) -> None:
        """Render validation views; developer mode shows the raw detector echo."""
        views = getattr(self, "_img_views", [])
        if not views:
            return
        show = self.boxes_chk.isChecked()
        dev = self.devmode_chk.isChecked()
        self.summary.setPlainText(self._raw_md if dev else self._refined_md)

        def overlay(img, refined, raw):
            if not show:
                return img
            return self._draw_objs(img, raw, raw=True) if dev else self._draw_objs(img, refined)

        if len(views) == 1:                       # single sensor: original | overlay
            _mod, img, refined, raw = views[0]
            set_frame(self.eo_view, img)
            set_frame(self.ir_view, overlay(img, refined, raw))
        else:                                     # pair: EO overlay | IR overlay
            for label, (_mod, img, refined, raw) in zip((self.eo_view, self.ir_view), views):
                set_frame(label, overlay(img, refined, raw))

    def _on_analysis_failed(self, msg: str) -> None:
        self.video_btn.setEnabled(True)
        self.image_btn.setEnabled(True)
        self.progress.setVisible(False)
        self._set_status(f"Analysis failed: {msg}")

    # -- shutdown -------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        if self.engine is not None:
            self._stop_live()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
