# examples/tkinter_gcs — embed the SDK in a Tkinter GCS

A minimal, self‑contained **mini Ground Control Station** that shows how to embed the
Vision Assistant SDK into an existing camera app behind **one button**. It owns its own
OpenCV camera (exactly like a real host) and uses the SDK only as an AI layer.

It mirrors, in ~150 lines, the integration used in the real **LUMIRA** GCS. See the full
walk‑through in [`docs/integration/INTEGRATION.md`](../../docs/integration/INTEGRATION.md).

## Run

```bash
# from the SDK repo root, in the SDK's venv
python examples/tkinter_gcs/mini_gcs.py            # default webcam (source 0)
python examples/tkinter_gcs/mini_gcs.py rtsp://user:pass@host:554/eo
python examples/tkinter_gcs/mini_gcs.py path/to/clip.mp4   # video file, looped
```

1. **▶ Start feed** — opens the camera/source (host‑owned capture).
2. **◬ Vision AI** — toggles the SDK. First press loads the models on a worker thread
   (the UI never freezes); then detection boxes, track IDs and a rolling intel line
   appear on the video. Press again to stop.

## What to read in the code

Every SDK‑specific line in [`mini_gcs.py`](mini_gcs.py) is fenced with `# ── SDK ──`.
There are only four touch‑points:

```python
va = VisionAssistant(get_config()); va.add_frame_source("EO"); va.on_result(cb); va.start()  # init
va.submit_frame("EO", frame)        # feed   — in the capture loop
va.overlay_payload()                # draw   — in the display builder
va.stop()                           # stop   — on toggle‑off / close
```

Everything else is ordinary host code (Tk window, OpenCV capture, a render tick). The
host never touches the detector, tracker, fusion engine, or VLM.

## Tuning (env vars)

| Env var | Default | Effect |
|---------|---------|--------|
| `VISION_ASSISTANT_BACKEND` | `yolo11` | detector backend (`yolo11` ships with the SDK) |
| `VISION_ASSISTANT_VLM` | on | set `0` to skip the VLM for a lighter demo |
| `VISION_ASSISTANT_SDK` | repo root | where to import the SDK from (if not `pip install`ed) |
