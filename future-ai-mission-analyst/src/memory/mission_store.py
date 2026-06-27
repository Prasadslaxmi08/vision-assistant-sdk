"""Asynchronous, thread-safe SQLite mission store.

Design goals
  * The real-time inference thread must NEVER block on disk I/O — all writes are
    enqueued and drained by a dedicated background writer thread that batches
    them into transactions.
  * Per-identity lifecycle (objects table) is aggregated in memory and flushed
    on each commit cycle, so the high-frequency detection stream doesn't thrash
    the objects table.
  * Reads (UI timeline, Object Explorer, Query Assistant) run on other threads;
    WAL mode lets them read concurrently with the writer. Each read opens a
    short-lived connection, keeping thread-affinity simple.

The store is intentionally a thin, well-indexed persistence layer with a small
query surface. Higher-level reasoning (spatial, fusion, threat, NL query) lives
in dedicated modules that call into this store.
"""
from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config.settings import MemoryConfig
from src.memory.schema import SCHEMA_SQL, SCHEMA_VERSION
from src.utils.logger import logger

# Event types that advance lifecycle counters.
_COUNTER_EVENTS = {
    "DIRECTION_CHANGE": "direction_changes",
    "STOPPED": "stop_count",
    "OBJECT_REIDENTIFIED": "reid_count",
}


@dataclass
class _ObjAgg:
    """In-memory running lifecycle for one identity (flushed to the DB)."""
    global_id: int
    class_name: str = ""
    first_seen_ts: float = 0.0
    last_seen_ts: float = 0.0
    first_frame: int = 0
    last_frame: int = 0
    frames_seen: int = 0
    max_confidence: float = 0.0
    reid_count: int = 0
    stop_count: int = 0
    direction_changes: int = 0
    interaction_count: int = 0
    distance_px: float = 0.0
    last_x: float = 0.0
    last_y: float = 0.0
    last_bbox: tuple = (0.0, 0.0, 0.0, 0.0)
    threat_score: float = 0.0
    threat_level: str = "NONE"
    attributes: dict = field(default_factory=dict)
    snapshot: Optional[bytes] = None
    dirty: bool = True


class MissionStore:
    """Persistent, async mission memory backed by SQLite."""

    def __init__(self, config: MemoryConfig, project_root: Optional[Path] = None):
        self.config = config
        self.enabled = config.enabled
        root = project_root or Path.cwd()
        p = Path(config.db_path)
        self.db_path = p if p.is_absolute() else (root / p)

        self._q: "queue.Queue[tuple]" = queue.Queue(maxsize=config.queue_maxsize)
        self._writer: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._mission_id: Optional[int] = None
        self._agg: Dict[int, _ObjAgg] = {}
        self._dropped = 0

    # ------------------------------------------------------------ lifecycle
    def open_mission(self, name: str, source: str, modality: str,
                     meta: Optional[dict] = None) -> Optional[int]:
        """Create a mission row and start the writer thread. Returns mission_id."""
        if not self.enabled:
            return None
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(SCHEMA_SQL)
            conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('version', ?)",
                (str(SCHEMA_VERSION),))
            cur = conn.execute(
                "INSERT INTO missions(name, source, modality, started_at, meta) "
                "VALUES(?,?,?,?,?)",
                (name, source, modality, time.time(),
                 json.dumps(meta or {})))
            self._mission_id = int(cur.lastrowid)
            conn.commit()
        finally:
            conn.close()

        self._agg.clear()
        self._stop.clear()
        self._writer = threading.Thread(target=self._run, name="mission-writer",
                                        daemon=True)
        self._writer.start()
        logger.info("Mission memory opened: id={} db={}", self._mission_id,
                    self.db_path.name)
        return self._mission_id

    def close_mission(self) -> None:
        """Flush remaining writes and stop the writer thread."""
        if not self.enabled or self._mission_id is None:
            return
        self._enqueue(("close", time.time()))
        self._stop.set()
        if self._writer:
            self._writer.join(timeout=5.0)
        if self._dropped:
            logger.warning("Mission memory dropped {} write ops under load",
                           self._dropped)
        logger.info("Mission memory closed: id={}", self._mission_id)

    @property
    def mission_id(self) -> Optional[int]:
        return self._mission_id

    def queue_stats(self) -> dict:
        """Write-queue health for profiling (depth / dropped / capacity)."""
        return {"depth": self._q.qsize(), "dropped": self._dropped,
                "maxsize": self.config.queue_maxsize}

    # --------------------------------------------------------------- writes
    def record_frame(self, frame_id: int, source_ts: float, detections) -> None:
        """Record one processed frame's tracked detections (non-blocking)."""
        if not self.enabled or self._mission_id is None:
            return
        rows = []
        for d in detections:
            if d.track_id is None:
                continue
            cx, cy = d.center
            x1, y1, x2, y2 = d.bbox
            rows.append((int(d.track_id), d.class_name, float(d.confidence),
                         float(x1), float(y1), float(x2), float(y2),
                         float(cx), float(cy)))
        if rows:
            self._enqueue(("frame", frame_id, source_ts, rows))

    def record_events(self, events) -> None:
        if not self.enabled or self._mission_id is None or not events:
            return
        payload = [(
            e.frame_id, e.source_ts, e.timestamp, e.type.value, e.description,
            json.dumps(e.track_ids), json.dumps(e.classes),
            (e.position[0] if e.position else None),
            (e.position[1] if e.position else None),
            float(e.metadata.get("threat_delta", 0.0)) if e.metadata else 0.0,
            json.dumps(e.metadata or {}),
            list(e.track_ids), e.type.value,
        ) for e in events]
        self._enqueue(("events", payload))

    def record_vlm(self, frame_id: int, source_ts: float, timestamp: float,
                   reason: str, kind: str, text: str) -> None:
        if not self.enabled or self._mission_id is None:
            return
        self._enqueue(("vlm", (frame_id, source_ts, timestamp, reason, kind, text)))

    def update_object_media(self, snapshots: Dict[int, Any]) -> None:
        """Attach best thumbnails to identities. ``snapshots`` maps global_id ->
        object with ``.jpeg/.class_name/.source_ts/.confidence`` (or a dict)."""
        if not self.enabled or self._mission_id is None or not snapshots:
            return
        media = {}
        for gid, s in snapshots.items():
            if isinstance(s, dict):
                media[int(gid)] = (s.get("jpeg"), s.get("class_name", ""),
                                   s.get("source_ts", 0.0), s.get("confidence", 0.0))
            else:
                media[int(gid)] = (s.jpeg, s.class_name, s.source_ts, s.confidence)
        self._enqueue(("media", media))

    def update_threat(self, updates: Dict[int, tuple]) -> None:
        """Persist per-object threat. ``updates`` maps global_id -> (score, level)."""
        if not self.enabled or self._mission_id is None or not updates:
            return
        self._enqueue(("threat", dict(updates)))

    def record_interaction(self, frame_id: int, source_ts: float, itype: str,
                           subject_id: int, object_id: Optional[int],
                           description: str, metadata: Optional[dict] = None) -> None:
        if not self.enabled or self._mission_id is None:
            return
        self._enqueue(("interaction", (frame_id, source_ts, itype, subject_id,
                                       object_id, description,
                                       json.dumps(metadata or {}))))

    def record_thermal(self, frame_id: int, source_ts: float, area_px: int,
                       x: Optional[float], y: Optional[float],
                       correlated_global_id: Optional[int], description: str) -> None:
        if not self.enabled or self._mission_id is None:
            return
        self._enqueue(("thermal", (frame_id, source_ts, area_px, x, y,
                                   correlated_global_id, description)))

    # ----------------------------------------------------------- writer loop
    def _enqueue(self, item: tuple) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self._dropped += 1  # shed load rather than block the inference thread

    def _run(self) -> None:
        conn = self._connect()
        mid = self._mission_id
        stride = max(1, self.config.detection_sample_stride)
        last_commit = time.time()
        try:
            while True:
                try:
                    item = self._q.get(timeout=0.5)
                except queue.Empty:
                    if self._stop.is_set() and self._q.empty():
                        break
                    self._flush(conn, mid)
                    continue
                self._apply(conn, mid, item, stride)
                # Drain whatever else is queued, then commit as one transaction.
                drained = 0
                while drained < 512:
                    try:
                        nxt = self._q.get_nowait()
                    except queue.Empty:
                        break
                    self._apply(conn, mid, nxt, stride)
                    drained += 1
                if time.time() - last_commit >= self.config.flush_interval_sec or drained:
                    self._flush(conn, mid)
                    last_commit = time.time()
            self._flush(conn, mid)
            conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Mission writer crashed: {}", exc)
        finally:
            conn.close()

    def _apply(self, conn: sqlite3.Connection, mid: int, item: tuple,
               stride: int) -> None:
        kind = item[0]
        if kind == "frame":
            _, frame_id, source_ts, rows = item
            store_rows = (frame_id % stride == 0)
            for (gid, cname, conf, x1, y1, x2, y2, cx, cy) in rows:
                self._update_agg(gid, cname, conf, frame_id, source_ts,
                                 (x1, y1, x2, y2), cx, cy)
                if store_rows:
                    conn.execute(
                        "INSERT INTO detections(mission_id,frame_id,source_ts,"
                        "global_id,class_name,confidence,x1,y1,x2,y2,cx,cy) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (mid, frame_id, source_ts, gid, cname, conf,
                         x1, y1, x2, y2, cx, cy))
        elif kind == "events":
            for row in item[1]:
                *cols, track_ids, etype = row
                conn.execute(
                    "INSERT INTO events(mission_id,frame_id,source_ts,timestamp,"
                    "type,description,track_ids,classes,x,y,threat_delta,metadata)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (mid, *cols))
                counter = _COUNTER_EVENTS.get(etype)
                if counter and track_ids:
                    agg = self._agg.get(int(track_ids[0]))
                    if agg is not None:
                        setattr(agg, counter, getattr(agg, counter) + 1)
                        agg.dirty = True
        elif kind == "vlm":
            conn.execute(
                "INSERT INTO vlm_observations(mission_id,frame_id,source_ts,"
                "timestamp,reason,kind,text) VALUES(?,?,?,?,?,?,?)",
                (mid, *item[1]))
        elif kind == "media":
            for gid, (jpeg, cname, ts, conf) in item[1].items():
                agg = self._agg.get(gid)
                if agg is None:
                    agg = self._agg.setdefault(gid, _ObjAgg(global_id=gid))
                if jpeg is not None:
                    agg.snapshot = jpeg
                if cname and not agg.class_name:
                    agg.class_name = cname
                agg.dirty = True
        elif kind == "interaction":
            conn.execute(
                "INSERT INTO interactions(mission_id,frame_id,source_ts,type,"
                "subject_id,object_id,description,metadata) VALUES(?,?,?,?,?,?,?,?)",
                (mid, *item[1]))
            agg = self._agg.get(int(item[1][3]))
            if agg is not None:
                agg.interaction_count += 1
                agg.dirty = True
        elif kind == "threat":
            for gid, (score, level) in item[1].items():
                agg = self._agg.get(gid)
                if agg is not None:
                    agg.threat_score = score
                    agg.threat_level = level
                    agg.dirty = True
        elif kind == "thermal":
            conn.execute(
                "INSERT INTO thermal_events(mission_id,frame_id,source_ts,area_px,"
                "x,y,correlated_global_id,description) VALUES(?,?,?,?,?,?,?,?)",
                (mid, *item[1]))
        elif kind == "close":
            conn.execute("UPDATE missions SET ended_at=? WHERE mission_id=?",
                         (item[1], mid))

    def _update_agg(self, gid: int, cname: str, conf: float, frame_id: int,
                    source_ts: float, bbox: tuple, cx: float, cy: float) -> None:
        agg = self._agg.get(gid)
        if agg is None:
            agg = _ObjAgg(global_id=gid, class_name=cname,
                          first_seen_ts=source_ts, first_frame=frame_id,
                          last_x=cx, last_y=cy)
            self._agg[gid] = agg
        else:
            # Accumulate path length between consecutive sightings.
            agg.distance_px += ((cx - agg.last_x) ** 2 + (cy - agg.last_y) ** 2) ** 0.5
        agg.class_name = cname or agg.class_name
        agg.last_seen_ts = source_ts
        agg.last_frame = frame_id
        agg.frames_seen += 1
        agg.max_confidence = max(agg.max_confidence, conf)
        agg.last_x, agg.last_y = cx, cy
        agg.last_bbox = bbox
        agg.dirty = True

    def _flush(self, conn: sqlite3.Connection, mid: int) -> None:
        dirty = [a for a in self._agg.values() if a.dirty]
        for a in dirty:
            conn.execute(
                "INSERT OR REPLACE INTO objects(mission_id,global_id,class_name,"
                "first_seen_ts,last_seen_ts,first_frame,last_frame,frames_seen,"
                "max_confidence,reid_count,stop_count,direction_changes,"
                "interaction_count,distance_px,last_x,last_y,last_bbox,threat_score,"
                "threat_level,attributes,snapshot) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (mid, a.global_id, a.class_name, a.first_seen_ts, a.last_seen_ts,
                 a.first_frame, a.last_frame, a.frames_seen, a.max_confidence,
                 a.reid_count, a.stop_count, a.direction_changes,
                 a.interaction_count, a.distance_px, a.last_x, a.last_y,
                 json.dumps(a.last_bbox), a.threat_score, a.threat_level,
                 json.dumps(a.attributes), a.snapshot))
            a.dirty = False
        try:
            conn.commit()
        except sqlite3.Error as exc:  # noqa: BLE001
            logger.warning("Mission memory commit failed: {}", exc)

    # --------------------------------------------------------------- reads
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False,
                               timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def read(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Run a read-only query and return a list of dict rows."""
        if not self.enabled or not self.db_path.exists():
            return []
        conn = self._connect()
        try:
            cur = conn.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
        except sqlite3.Error as exc:  # noqa: BLE001
            logger.warning("Mission memory read failed: {}", exc)
            return []
        finally:
            conn.close()

    def list_missions(self) -> List[Dict[str, Any]]:
        """All archived missions, newest first (for the UI mission picker)."""
        return self.read(
            "SELECT mission_id, name, source, modality, started_at, ended_at, meta "
            "FROM missions ORDER BY started_at DESC")

    def list_objects(self, mission_id: Optional[int] = None) -> List[Dict[str, Any]]:
        mid = mission_id or self._mission_id
        return self.read(
            "SELECT * FROM objects WHERE mission_id=? ORDER BY first_seen_ts", (mid,))

    def object_track(self, global_id: int, mission_id: Optional[int] = None
                     ) -> List[Dict[str, Any]]:
        """Sampled (cx, cy) path for one identity, oldest first."""
        mid = mission_id or self._mission_id
        return self.read(
            "SELECT source_ts, cx, cy FROM detections "
            "WHERE mission_id=? AND global_id=? ORDER BY source_ts", (mid, global_id))

    def interactions(self, mission_id: Optional[int] = None, limit: int = 1000
                     ) -> List[Dict[str, Any]]:
        mid = mission_id or self._mission_id
        return self.read(
            "SELECT source_ts, type, subject_id, object_id, description "
            "FROM interactions WHERE mission_id=? ORDER BY source_ts LIMIT ?",
            (mid, limit))

    def thermal_events(self, mission_id: Optional[int] = None, limit: int = 1000
                       ) -> List[Dict[str, Any]]:
        mid = mission_id or self._mission_id
        return self.read(
            "SELECT source_ts, area_px, x, y, correlated_global_id, description "
            "FROM thermal_events WHERE mission_id=? ORDER BY source_ts LIMIT ?",
            (mid, limit))

    def timeline(self, mission_id: Optional[int] = None, limit: int = 1000
                 ) -> List[Dict[str, Any]]:
        mid = mission_id or self._mission_id
        return self.read(
            "SELECT source_ts, type, description, 'event' AS kind FROM events "
            "WHERE mission_id=? "
            "UNION ALL "
            "SELECT source_ts, 'VLM_'||UPPER(kind) AS type, text AS description, "
            "'vlm' AS kind FROM vlm_observations WHERE mission_id=? "
            "ORDER BY source_ts LIMIT ?", (mid, mid, limit))

    def mission_stats(self, mission_id: Optional[int] = None) -> Dict[str, Any]:
        mid = mission_id or self._mission_id
        rows = self.read(
            "SELECT COUNT(*) AS n, "
            "SUM(CASE WHEN class_name='person' THEN 1 ELSE 0 END) AS people "
            "FROM objects WHERE mission_id=?", (mid,))
        ev = self.read(
            "SELECT type, COUNT(*) AS n FROM events WHERE mission_id=? GROUP BY type",
            (mid,))
        return {"mission_id": mid,
                "unique_objects": (rows[0]["n"] if rows else 0),
                "unique_people": (rows[0]["people"] if rows and rows[0]["people"] else 0),
                "events": {r["type"]: r["n"] for r in ev}}
