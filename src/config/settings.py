"""Typed configuration loader.

Loads ``config/config.yaml`` (or the path in the ``EOIR_CONFIG`` env var) into
validated Pydantic models. Import :func:`get_config` anywhere to access a cached,
process-wide singleton::

    from src.config.settings import get_config
    cfg = get_config()
    print(cfg.detection.conf_threshold)
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field

# Project root = two levels up from this file (src/config/settings.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


class SystemConfig(BaseModel):
    device: str = "cuda:0"
    half_precision: bool = True
    log_level: str = "INFO"
    log_dir: str = "logs"
    output_dir: str = "outputs"


class IngestionConfig(BaseModel):
    target_fps: float = 0.0
    buffer_size: int = 64
    drop_policy: str = "newest_keeps"
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    rtsp_reconnect_delay: float = 3.0
    rtsp_max_reconnects: int = -1
    read_timeout: float = 10.0


class RFDetrConfig(BaseModel):
    """RF-DETR backend options (read only when ``detection.backend == 'rf_detr'``).

    RF-DETR is a swappable *replacement* for YOLO (one detector loaded at a time),
    so peak VRAM stays one detector + the VLM — the earlier OOM concern about a
    third concurrent model does not apply. The chosen variant's VRAM alongside
    Qwen-3B(4-bit) must still be confirmed on the box (detector-benchmark)."""
    variant: Literal["nano", "small", "medium", "base", "large"] = "base"
    weights: str = ""          # path or hub id; empty = the variant's pretrained COCO weights
    img_size: int = 560        # RF-DETR native resolution (multiple of 56)


class TilingConfig(BaseModel):
    """Sliced (tiled) inference for small / distant / low-contrast objects.

    High-resolution frames are downsampled to the detector's native resolution,
    which erases small targets (e.g. a distant boat in a thermal frame). Tiling
    runs the detector over overlapping crops at near-native resolution and merges
    the results with NMS, recovering small objects the full-frame pass misses.
    Used for still-image analysis; kept off the real-time path by default (it
    multiplies inference cost per frame)."""
    enabled: bool = True
    tile: int = 640            # tile size in px (≈ the detector's native resolution)
    overlap: float = 0.2       # fraction of overlap between adjacent tiles
    min_side: int = 900        # only tile when the longest image side exceeds this
    iou_merge: float = 0.55    # IoU threshold to merge full-frame + tile detections
    max_tiles: int = 24        # safety cap on tiles per image


class DetectionConfig(BaseModel):
    # RF-DETR is the primary (transformer) detector; YOLOv11 stays as a one-line
    # fallback. Swap the whole detector via config only — downstream is untouched.
    backend: Literal["rf_detr", "yolo11"] = "rf_detr"
    model_path: str = "yolo11m.pt"                     # used when backend == yolo11
    conf_threshold: float = 0.25   # lowered for small-object recall (was 0.30)
    iou_threshold: float = 0.50
    img_size: int = 640
    max_detections: int = 300
    classes: Optional[List[int]] = None
    # Report filter (the "class selector"): COCO class NAMES to surface. null/empty
    # = report every class. Applied AFTER detection, so small-object recall is
    # unchanged — only which classes reach reasoning/report/overlay is narrowed.
    # The demo's class selector edits this; headless consumers (engine, CLI) read it.
    report_classes: Optional[List[str]] = None
    rf_detr: RFDetrConfig = Field(default_factory=RFDetrConfig)
    tiling: TilingConfig = Field(default_factory=TilingConfig)


class TrackingConfig(BaseModel):
    track_activation_threshold: float = 0.25
    lost_track_buffer: int = 30
    minimum_matching_threshold: float = 0.80
    frame_rate: int = 30
    minimum_consecutive_frames: int = 3


class ReIDConfig(BaseModel):
    """Appearance re-identification — stable ids across ByteTrack id churn."""
    enabled: bool = True
    # Classes that attempt cross-break appearance matching. Others still get a
    # stable global id, but a fresh one each time their raw track is recreated.
    classes: List[str] = Field(default_factory=lambda: ["person"])
    similarity_threshold: float = 0.55   # cosine sim on the appearance signature
    max_age_sec: float = 30.0            # how long a departed identity stays matchable
    max_distance_px: float = 0.0         # spatial gate on re-entry (0 = disabled)
    embedding_refresh_frames: int = 12   # EMA-refresh a live track's signature every N frames
    gallery_max_size: int = 96           # hard cap on remembered identities
    hist_bins: int = 32                  # histogram resolution per channel
    min_crop_px: int = 16                # ignore crops smaller than this (too small to ReID)
    track_grace_frames: int = 30         # keep a raw->global map this long after a track
                                         # vanishes (match to tracking.lost_track_buffer)


class VLMConfig(BaseModel):
    enabled: bool = True
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    quantization: Literal["4bit", "8bit", "none"] = "4bit"
    device: str = "cuda:0"
    max_new_tokens: int = 384
    temperature: float = 0.2
    min_pixels: int = 200704
    max_pixels: int = 1003520
    periodic_interval_sec: float = 5.0
    trigger_on_events: bool = True
    min_seconds_between_calls: float = 1.5
    max_pending_requests: int = 2


class EventsConfig(BaseModel):
    stop_speed_px: float = 1.5
    stop_min_frames: int = 15
    direction_change_deg: float = 45.0
    direction_min_speed_px: float = 3.0
    motion_window: int = 8
    enter_vehicle_iou: float = 0.15
    enter_vehicle_distance_px: float = 80.0
    thermal_hotspot_threshold: int = 200
    thermal_min_area_px: int = 150
    cooldown_sec: float = 2.0


class SpatialConfig(BaseModel):
    """Spatial Reasoning Engine — object-to-object relationships (V2 phase 2)."""
    enabled: bool = True
    history_len: int = 16                 # per-track center history window
    proximity_distance_px: float = 80.0   # centers nearer than this = "close"
    proximity_min_frames: int = 8         # sustained frames before PROXIMITY fires
    following_min_frames: int = 15
    following_max_lag_px: float = 140.0    # max follower→leader gap
    following_min_speed_px: float = 1.5    # both must be moving at least this fast
    group_min_size: int = 3
    group_radius_px: float = 120.0
    group_min_frames: int = 10
    # Restricted keep-out zones as normalised [x1,y1,x2,y2] rectangles (0..1).
    restricted_zones: List[List[float]] = Field(default_factory=list)
    restricted_min_frames: int = 3
    cooldown_sec: float = 8.0             # min seconds between re-firing one relation


class SensorProfile(BaseModel):
    """Optics description for one camera. Every field is OPTIONAL — the system
    runs cross-sensor fusion with none of it (auto-registration estimates the
    mapping from shared targets). When HFOV + width are known for both sensors,
    their ratio seeds a scale prior that makes the registration lock faster."""
    name: str = "generic"
    width: int = 0                       # delivered frame width (px), 0 = unknown
    height: int = 0
    pixel_pitch_um: float = 0.0          # detector pitch (e.g. IR 12.0)
    focal_length_mm: float = 0.0         # e.g. IR 50.0
    hfov_deg: float = 0.0                # horizontal FOV if known (EO wide 66, IR ~12)
    radiometric: bool = False            # True = per-pixel temperature; False = 8-bit AGC


class RegistrationConfig(BaseModel):
    """Automatic EO/IR cross-sensor registration — NO calibration required.

    The two sensors are unaligned with different FOV/zoom and provide no
    boresight or telemetry, so the transform mapping IR→EO is estimated online
    from the geometry of targets both sensors can see (RANSAC over object/blob
    correspondences), smoothed over time, and gated by a confidence 'lock'."""
    enabled: bool = True
    similarity_only: bool = True         # 4-DOF scale+rotation+translation (vs full affine)
    inlier_tol_frac: float = 0.05        # match tolerance as a fraction of EO width
    min_points: int = 2                  # min shared points needed to hypothesise a fit
    min_lock_inliers: int = 2            # inliers (matched targets) required to hold a lock
    lock_persist_cycles: int = 3         # consecutive good cycles before declaring LOCKED
    unlock_grace_cycles: int = 6         # cycles of no fit tolerated before dropping lock
    smoothing: float = 0.6               # EMA on transform params (0=snap, →1=sluggish)
    max_warm_sec: float = 3.0            # reuse last transform up to this long without a new fit
    class_constrained: bool = True       # only hypothesise same-class correspondences
    ransac_iters: int = 300
    scale_prior_tol: float = 3.0         # accept fits within this factor of the FOV-ratio prior
    # -- Spike-proofing the RANSAC (latency roadmap M5/B5) -----------------
    max_points: int = 8                  # cap control points per sensor (strongest by weight)
                                         # — keeps the 2x2 hypothesis enumeration bounded
    early_exit_inliers: int = 4          # stop searching once a fit reaches this many inliers
    update_interval: int = 2             # re-estimate the transform every Nth fuse cycle
                                         # (1 = every cycle); the lock FSM/WARM hold reuses
                                         # the last transform in between — it changes slowly


class FusionConfig(BaseModel):
    """EO/IR Fusion Engine — cross-sensor thermal↔visual correlation (V2).

    Operates on a *registered pair* of independent EO and IR streams. On a lone
    IR stream it degrades to intra-frame thermal analysis. Thermal thresholding
    is adaptive by default because an 8-bit AGC IR stream has no fixed radiometric
    scale (brightness drifts frame to frame)."""
    enabled: bool = True
    # -- Thermal hotspot extraction (8-bit AGC friendly) -------------------
    adaptive: bool = True                # relative threshold from per-frame stats
    white_hot: bool = True               # True = hot is bright; False = black-hot (hot is dark)
    hot_percentile: float = 99.0         # adaptive: pixels above this percentile = "hot"
    hot_threshold: int = 200             # fixed fallback (radiometric / adaptive off)
    min_hot_threshold: int = 70          # floor so a flat/cold frame yields no blobs
    min_blob_area_px: int = 120          # ignore hot blobs smaller than this (full-res px)
    max_blobs: int = 20                  # cap blobs processed per frame
    blob_downscale_width: int = 640      # downscale IR to at most this width before the
                                         # percentile + connected-components pass, then scale
                                         # blob coords back up (latency roadmap M5/B6).
                                         # 0 disables; no-op when the frame is already smaller.
    # -- Cross-sensor correlation -----------------------------------------
    correlate_overlap: float = 0.30      # min blob-covered fraction to bind to a detection
    correlate_radius_frac: float = 0.04  # point-match radius (frac of EO width) when boxes unavailable
    concealed_min_area_px: int = 150     # hot blob this big with no detection = concealed target
    concealed_cell_px: int = 48          # spatial cell size for de-duping concealed-heat alerts
    min_frames: int = 4                  # sustained frames before an assessment fires
    cooldown_sec: float = 8.0


class ObjectReasoningConfig(BaseModel):
    """Object Reasoning Engine — refine raw detections before reporting / VLM.

    Treats detector output as raw observations: clusters overlapping boxes into a
    single physical object, ranks the most-probable class (refined by scene
    context), splits confirmed vs low-confidence, and infers human-object
    interactions. Deterministic and explainable — it never invents objects."""
    enabled: bool = True
    cluster_iou: float = 0.45        # group detections as one object above this IoU
    confirm_threshold: float = 0.45  # >= confirmed; below = a 'possible' observation
    alt_margin: float = 0.55         # keep an alternative class if adj-conf >= margin*primary
    interaction_pad_frac: float = 0.35  # person↔object proximity gate (frac of person size)
    context_boost: float = 1.30      # plausible-class multiplier in the matching scene
    context_penalty: float = 0.80    # implausible-class multiplier in the matching scene


class IntelligenceConfig(BaseModel):
    """Intelligence Layer — live aggregation of the scene (no persistence).

    Holds the in-memory intel feed, timeline and stats; report generation and
    long-term archives moved to the future AI Mission Analyst repo."""
    summary_interval_sec: float = 15.0
    timeline_max_events: int = 500
    intel_feed_history: int = 12         # natural-language messages kept on screen
    max_vlm_objects: int = 12            # cap detections sent to the VLM for per-object
                                         # notes (all still appear in the report)


class UIConfig(BaseModel):
    """Demo-app rendering knobs (overlay compositing)."""
    title: str = "Vision Assistant"
    max_display_width: int = 1280
    show_confidence: bool = True
    show_track_ids: bool = True
    feed_refresh_sec: float = 0.03       # demo feed refresh period (~30 Hz target)


class CadenceConfig(BaseModel):
    """Sub-rate the heavy AI/reasoning stages relative to capture (M5/B7).

    Detection, tracking, ReID and per-frame mission-memory recording always run
    every frame — only the heavier *synthesis* stages are throttled, because they
    do not need per-frame frequency for a smooth feed (the live display is already
    decoupled from inference, see M2). Defaults of 1 reproduce the every-frame
    behaviour exactly; raise them on the box if the dashboard shows reasoning/
    fusion spikes. No output is removed — only its frequency drops (events emitted
    on skipped frames are accumulated and folded into the next reasoning pass)."""
    reasoning_interval: int = 1   # run spatial clustering + threat re-scoring every Nth EO frame
    fusion_interval: int = 1      # run the EO/IR fuse step every Nth inference loop cycle


class ProfilingConfig(BaseModel):
    """Runtime profiling instrumentation (latency roadmap Milestone 0).

    Off by default — it adds nothing to the hot path when disabled. Turn
    ``enabled`` on to collect measured per-stage latencies; turn ``dashboard``
    on to surface the dev-only profiling panel in the operator UI."""
    enabled: bool = False        # collect per-stage timings / rates / gauges
    window: int = 240            # samples retained per stage for p50/p95
    dashboard: bool = False      # show the dev profiling panel in the UI


class AppConfig(BaseModel):
    system: SystemConfig = Field(default_factory=SystemConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    reid: ReIDConfig = Field(default_factory=ReIDConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    events: EventsConfig = Field(default_factory=EventsConfig)
    spatial: SpatialConfig = Field(default_factory=SpatialConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    registration: RegistrationConfig = Field(default_factory=RegistrationConfig)
    eo_sensor: SensorProfile = Field(default_factory=lambda: SensorProfile(
        name="EO", hfov_deg=66.0))
    ir_sensor: SensorProfile = Field(default_factory=lambda: SensorProfile(
        name="IR", pixel_pitch_um=12.0, focal_length_mm=50.0, hfov_deg=12.0))
    intelligence: IntelligenceConfig = Field(default_factory=IntelligenceConfig)
    object_reasoning: ObjectReasoningConfig = Field(default_factory=ObjectReasoningConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    cadence: CadenceConfig = Field(default_factory=CadenceConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)

    # Resolved absolute paths (filled in post-load).
    project_root: Path = PROJECT_ROOT

    def abs_path(self, relative: str) -> Path:
        """Resolve a config-relative path against the project root."""
        p = Path(relative)
        return p if p.is_absolute() else (self.project_root / p)


def _config_path() -> Path:
    env = os.environ.get("EOIR_CONFIG")
    return Path(env).expanduser().resolve() if env else DEFAULT_CONFIG_PATH


def load_config(path: Optional[Path] = None) -> AppConfig:
    """Load and validate the YAML config. Missing file -> sane defaults."""
    cfg_path = path or _config_path()
    data: dict = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    cfg = AppConfig(**data)

    # Ensure output / log directories exist.
    cfg.abs_path(cfg.system.log_dir).mkdir(parents=True, exist_ok=True)
    cfg.abs_path(cfg.system.output_dir).mkdir(parents=True, exist_ok=True)
    return cfg


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    """Process-wide cached config singleton."""
    return load_config()


def reload_config() -> AppConfig:
    """Force a re-read from disk (clears the cache)."""
    get_config.cache_clear()
    return get_config()
