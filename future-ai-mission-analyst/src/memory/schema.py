"""SQLite schema for persistent mission memory.

One file = one rolling archive of missions. Every table is mission-scoped via
``mission_id`` so multiple missions coexist in a single database. The schema is
created idempotently on first connection.

Tables
  missions          one row per analysis session (live stream or video file)
  objects           per-identity lifecycle (the heart of temporal memory)
  detections        per-frame detection rows (sampled, for path reconstruction)
  events            significant events (the EventManager output)
  vlm_observations  Qwen2.5-VL scene summaries / alerts
  interactions      object-to-object relations (spatial reasoning, V2 phase 2)
  thermal_events    IR hotspots + correlation to detected objects (fusion)
"""
from __future__ import annotations

SCHEMA_VERSION = 1

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS missions (
    mission_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT,
    source      TEXT,
    modality    TEXT,
    started_at  REAL,          -- wall-clock epoch
    ended_at    REAL,
    meta        TEXT           -- json
);

-- Per-identity lifecycle. global_id is the stable ReID identity.
CREATE TABLE IF NOT EXISTS objects (
    mission_id        INTEGER NOT NULL,
    global_id         INTEGER NOT NULL,
    class_name        TEXT,
    first_seen_ts     REAL,    -- media-relative seconds
    last_seen_ts      REAL,
    first_frame       INTEGER,
    last_frame        INTEGER,
    frames_seen       INTEGER DEFAULT 0,
    max_confidence    REAL DEFAULT 0,
    reid_count        INTEGER DEFAULT 0,   -- times re-identified after leaving
    stop_count        INTEGER DEFAULT 0,
    direction_changes INTEGER DEFAULT 0,
    interaction_count INTEGER DEFAULT 0,
    distance_px       REAL DEFAULT 0,      -- accumulated path length
    last_x            REAL,
    last_y            REAL,
    last_bbox         TEXT,                -- json [x1,y1,x2,y2]
    threat_score      REAL DEFAULT 0,
    threat_level      TEXT DEFAULT 'NONE',
    attributes        TEXT,                -- json (VLM-derived)
    snapshot          BLOB,                -- jpeg thumbnail
    PRIMARY KEY (mission_id, global_id)
);

CREATE TABLE IF NOT EXISTS detections (
    mission_id  INTEGER NOT NULL,
    frame_id    INTEGER,
    source_ts   REAL,
    global_id   INTEGER,
    class_name  TEXT,
    confidence  REAL,
    x1 REAL, y1 REAL, x2 REAL, y2 REAL,
    cx REAL, cy REAL
);
CREATE INDEX IF NOT EXISTS idx_det_obj ON detections(mission_id, global_id);
CREATE INDEX IF NOT EXISTS idx_det_ts  ON detections(mission_id, source_ts);

CREATE TABLE IF NOT EXISTS events (
    mission_id   INTEGER NOT NULL,
    frame_id     INTEGER,
    source_ts    REAL,
    timestamp    REAL,
    type         TEXT,
    description  TEXT,
    track_ids    TEXT,        -- json
    classes      TEXT,        -- json
    x REAL, y REAL,
    threat_delta REAL DEFAULT 0,
    metadata     TEXT         -- json
);
CREATE INDEX IF NOT EXISTS idx_ev_ts   ON events(mission_id, source_ts);
CREATE INDEX IF NOT EXISTS idx_ev_type ON events(mission_id, type);

CREATE TABLE IF NOT EXISTS vlm_observations (
    mission_id INTEGER NOT NULL,
    frame_id   INTEGER,
    source_ts  REAL,
    timestamp  REAL,
    reason     TEXT,          -- 'periodic' | 'event'
    kind       TEXT,          -- 'summary' | 'alert'
    text       TEXT
);
CREATE INDEX IF NOT EXISTS idx_vlm_ts ON vlm_observations(mission_id, source_ts);

CREATE TABLE IF NOT EXISTS interactions (
    mission_id  INTEGER NOT NULL,
    frame_id    INTEGER,
    source_ts   REAL,
    type        TEXT,         -- PROXIMITY | FOLLOWING | GROUP | ENTER_VEHICLE | ...
    subject_id  INTEGER,
    object_id   INTEGER,
    description TEXT,
    metadata    TEXT
);
CREATE INDEX IF NOT EXISTS idx_int_ts  ON interactions(mission_id, source_ts);
CREATE INDEX IF NOT EXISTS idx_int_sub ON interactions(mission_id, subject_id);

CREATE TABLE IF NOT EXISTS thermal_events (
    mission_id           INTEGER NOT NULL,
    frame_id             INTEGER,
    source_ts            REAL,
    area_px              INTEGER,
    x REAL, y REAL,
    correlated_global_id INTEGER,
    description          TEXT
);
CREATE INDEX IF NOT EXISTS idx_therm_ts ON thermal_events(mission_id, source_ts);
"""
