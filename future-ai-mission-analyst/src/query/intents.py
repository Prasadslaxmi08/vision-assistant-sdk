"""Whitelisted query intents for the Mission Query Assistant (V2 phase 5).

Natural-language questions are **never** turned into free-form SQL. Instead each
question is routed to one of a fixed catalogue of *intents* defined here, and
every intent owns a single hard-coded, read-only, parameterised SQL statement.
The only user-derived values that ever reach the database are bound parameters
(integers, floats, or class names whitelisted against a fixed set) — so the
query surface is closed and SQL injection is impossible by construction.

Each intent returns an :class:`Evidence` whose ``answer`` is computed *in code*
straight from the returned rows. The VLM (see :mod:`src.query.assistant`) only
rephrases that grounded answer for readability; it can never alter the
underlying numbers, which is what keeps answers faithful to mission memory.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

# COCO vehicle classes (YOLOv11 names). Used for "vehicle" grouped queries.
VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle", "bicycle", "boat",
                   "train", "airplane")
# Object classes a question may name explicitly (whitelist for the class slot).
KNOWN_CLASSES = VEHICLE_CLASSES + ("person", "backpack", "handbag", "suitcase",
                                   "dog", "cat", "bird")


# --------------------------------------------------------------------------- #
#  Result containers
# --------------------------------------------------------------------------- #
@dataclass
class Evidence:
    """The grounded result of running one intent against mission memory.

    ``answer`` is derived deterministically from ``rows`` so the headline fact
    is never left to the language model. ``citations`` are the human-readable
    supporting records, and ``sql`` is kept for transparency / debugging.
    """
    answer: str
    citations: List[str] = field(default_factory=list)
    rows: List[Dict[str, Any]] = field(default_factory=list)
    sql: str = ""


@dataclass
class Intent:
    name: str
    description: str                 # one line, shown to the VLM classifier
    examples: List[str]              # few-shot question examples
    handler: Callable[..., Evidence] # (store, mission_id, slots) -> Evidence


# --------------------------------------------------------------------------- #
#  Formatting helpers
# --------------------------------------------------------------------------- #
def _fmt_ts(t: Optional[float]) -> str:
    if t is None:
        return "?"
    return f"{float(t):.1f}s"


def _fmt_dur(seconds: Optional[float]) -> str:
    s = max(0.0, float(seconds or 0.0))
    if s < 90:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    return f"{m}m{sec:02d}s"


def _pretty(token: str) -> str:
    return token.replace("_", " ").lower()


# --------------------------------------------------------------------------- #
#  Intent handlers — each owns ONE hard-coded, parameterised, read-only query.
# --------------------------------------------------------------------------- #
def _h_count_people(store, mid: int, slots: dict) -> Evidence:
    sql = ("SELECT global_id, first_seen_ts, last_seen_ts FROM objects "
           "WHERE mission_id=? AND class_name='person' ORDER BY first_seen_ts")
    rows = store.read(sql, (mid,))
    n = len(rows)
    verb = "person was" if n == 1 else "people were"
    answer = f"{n} unique {verb} tracked during the mission."
    cites = [f"person #{r['global_id']} — seen {_fmt_ts(r['first_seen_ts'])}"
             f"–{_fmt_ts(r['last_seen_ts'])}" for r in rows[:12]]
    return Evidence(answer, cites, rows, sql)


def _h_count_objects(store, mid: int, slots: dict) -> Evidence:
    cls = slots.get("class_name")
    if cls in ("vehicle", "vehicles"):
        ph = ",".join("?" * len(VEHICLE_CLASSES))
        sql = (f"SELECT class_name, COUNT(*) AS n FROM objects WHERE mission_id=? "
               f"AND class_name IN ({ph}) GROUP BY class_name ORDER BY n DESC")
        rows = store.read(sql, (mid, *VEHICLE_CLASSES))
        total = sum(r["n"] for r in rows)
        if not total:
            return Evidence("No vehicles were tracked during the mission.", [], rows, sql)
        breakdown = ", ".join(f"{r['n']}× {r['class_name']}" for r in rows)
        return Evidence(f"{total} unique vehicle(s) tracked: {breakdown}.",
                        [f"{r['n']}× {r['class_name']}" for r in rows], rows, sql)
    if cls and cls in KNOWN_CLASSES:
        sql = "SELECT COUNT(*) AS n FROM objects WHERE mission_id=? AND class_name=?"
        rows = store.read(sql, (mid, cls))
        n = rows[0]["n"] if rows else 0
        return Evidence(f"{n} unique {cls}(s) tracked during the mission.", [], rows, sql)
    sql = ("SELECT class_name, COUNT(*) AS n FROM objects WHERE mission_id=? "
           "GROUP BY class_name ORDER BY n DESC")
    rows = store.read(sql, (mid,))
    total = sum(r["n"] for r in rows)
    breakdown = ", ".join(f"{r['n']}× {r['class_name']}" for r in rows) or "none"
    return Evidence(f"{total} unique object(s) tracked: {breakdown}.",
                    [f"{r['n']}× {r['class_name']}" for r in rows], rows, sql)


def _h_longest_present(store, mid: int, slots: dict) -> Evidence:
    cls = slots.get("class_name")
    base = ("SELECT global_id, class_name, first_seen_ts, last_seen_ts, "
            "(last_seen_ts - first_seen_ts) AS dwell FROM objects WHERE mission_id=?")
    if cls in ("vehicle", "vehicles"):
        ph = ",".join("?" * len(VEHICLE_CLASSES))
        sql = base + f" AND class_name IN ({ph}) ORDER BY dwell DESC LIMIT 5"
        params, label = (mid, *VEHICLE_CLASSES), "vehicle"
    elif cls and cls in KNOWN_CLASSES:
        sql = base + " AND class_name=? ORDER BY dwell DESC LIMIT 5"
        params, label = (mid, cls), cls
    else:
        sql = base + " ORDER BY dwell DESC LIMIT 5"
        params, label = (mid,), "object"
    rows = store.read(sql, params)
    if not rows:
        return Evidence(f"No {label} was recorded during the mission.", [], rows, sql)
    top = rows[0]
    answer = (f"The longest-present {label} was {top['class_name']} "
              f"#{top['global_id']}, present for {_fmt_dur(top['dwell'])} "
              f"({_fmt_ts(top['first_seen_ts'])}–{_fmt_ts(top['last_seen_ts'])}).")
    cites = [f"{r['class_name']} #{r['global_id']} — {_fmt_dur(r['dwell'])}"
             for r in rows]
    return Evidence(answer, cites, rows, sql)


def _h_recent_activity(store, mid: int, slots: dict) -> Evidence:
    window = float(slots.get("window_sec") or 300.0)
    mx = store.read("SELECT MAX(last_seen_ts) AS t FROM objects WHERE mission_id=?",
                    (mid,))
    end = mx[0]["t"] if mx and mx[0]["t"] is not None else 0.0
    start = max(0.0, end - window)
    sql = ("SELECT source_ts, type, description FROM events WHERE mission_id=? "
           "AND source_ts>=? ORDER BY source_ts")
    rows = store.read(sql, (mid, start))
    win = _fmt_dur(window)
    if not rows:
        return Evidence(f"No significant events were recorded in the last {win}.",
                        [], rows, sql)
    answer = (f"{len(rows)} event(s) in the last {win} "
              f"(from {_fmt_ts(start)} to {_fmt_ts(end)}).")
    cites = [f"{_fmt_ts(r['source_ts'])} — {r['description']}" for r in rows[:20]]
    return Evidence(answer, cites, rows, sql)


def _h_thermal_events(store, mid: int, slots: dict) -> Evidence:
    sql = ("SELECT source_ts, area_px, correlated_global_id, description "
           "FROM thermal_events WHERE mission_id=? ORDER BY source_ts")
    rows = store.read(sql, (mid,))
    if not rows:
        return Evidence("No thermal (IR) events were recorded during the mission.",
                        [], rows, sql)
    answer = f"{len(rows)} thermal/IR event(s) were recorded during the mission."
    cites = []
    for r in rows[:20]:
        tag = (f" (object #{r['correlated_global_id']})"
               if r["correlated_global_id"] else "")
        cites.append(f"{_fmt_ts(r['source_ts'])} — {r['description']}{tag}")
    return Evidence(answer, cites, rows, sql)


def _h_most_direction_changes(store, mid: int, slots: dict) -> Evidence:
    sql = ("SELECT global_id, class_name, direction_changes, stop_count "
           "FROM objects WHERE mission_id=? AND direction_changes>0 "
           "ORDER BY direction_changes DESC LIMIT 5")
    rows = store.read(sql, (mid,))
    if not rows:
        return Evidence("No object exhibited notable direction changes.", [], rows, sql)
    top = rows[0]
    answer = (f"{top['class_name']} #{top['global_id']} changed direction the most "
              f"— {top['direction_changes']} times.")
    cites = [f"{r['class_name']} #{r['global_id']} — "
             f"{r['direction_changes']} direction changes, {r['stop_count']} stops"
             for r in rows]
    return Evidence(answer, cites, rows, sql)


def _h_suspicious(store, mid: int, slots: dict) -> Evidence:
    objs = store.read(
        "SELECT global_id, class_name, threat_score, threat_level FROM objects "
        "WHERE mission_id=? AND threat_score>0 ORDER BY threat_score DESC LIMIT 8",
        (mid,))
    evs = store.read(
        "SELECT source_ts, description FROM events WHERE mission_id=? "
        "AND type='THREAT_ESCALATION' ORDER BY source_ts", (mid,))
    ints = store.read(
        "SELECT type, description FROM interactions WHERE mission_id=? "
        "AND type IN ('FOLLOWING','RESTRICTED_AREA') ORDER BY source_ts", (mid,))
    total = len(objs) + len(evs) + len(ints)
    if not total:
        return Evidence("No suspicious activity was flagged during the mission.",
                        [], [], "")
    cites: List[str] = []
    for o in objs:
        cites.append(f"{o['class_name']} #{o['global_id']} — threat "
                     f"{o['threat_level']} ({o['threat_score']:.1f})")
    for e in evs:
        cites.append(f"{_fmt_ts(e['source_ts'])} — {e['description']}")
    for it in ints:
        cites.append(f"{_pretty(it['type'])}: {it['description']}")
    answer = (f"{total} suspicious indicator(s) flagged: "
              f"{len(objs)} elevated-threat object(s), "
              f"{len(evs)} threat escalation(s), "
              f"{len(ints)} suspicious interaction(s).")
    return Evidence(answer, cites[:24], objs + evs + ints, "(suspicious composite)")


def _h_threat_assessment(store, mid: int, slots: dict) -> Evidence:
    sql = ("SELECT global_id, class_name, threat_score, threat_level FROM objects "
           "WHERE mission_id=? ORDER BY threat_score DESC LIMIT 6")
    rows = store.read(sql, (mid,))
    elevated = [r for r in rows if r["threat_score"] > 0]
    if not elevated:
        return Evidence("Mission threat level is NONE — no elevated-threat objects "
                        "were recorded.", [], rows, sql)
    top = elevated[0]
    answer = (f"Highest threat: {top['class_name']} #{top['global_id']} at "
              f"{top['threat_level']} ({top['threat_score']:.1f}). "
              f"{len(elevated)} object(s) carried a non-zero threat score.")
    cites = [f"{r['class_name']} #{r['global_id']} — "
             f"{r['threat_level']} ({r['threat_score']:.1f})" for r in elevated]
    return Evidence(answer, cites, rows, sql)


def _h_object_detail(store, mid: int, slots: dict) -> Evidence:
    gid = slots.get("global_id")
    if gid is None:
        return Evidence("Please specify which object/track id to describe "
                        "(e.g. 'tell me about person 3').", [], [], "")
    sql = ("SELECT global_id, class_name, first_seen_ts, last_seen_ts, frames_seen, "
           "distance_px, stop_count, direction_changes, reid_count, "
           "interaction_count, threat_score, threat_level "
           "FROM objects WHERE mission_id=? AND global_id=?")
    rows = store.read(sql, (mid, gid))
    if not rows:
        return Evidence(f"No object with id #{gid} was found in this mission.",
                        [], rows, sql)
    o = rows[0]
    dwell = (o["last_seen_ts"] or 0.0) - (o["first_seen_ts"] or 0.0)
    answer = (f"{o['class_name']} #{gid}: present {_fmt_dur(dwell)} "
              f"({_fmt_ts(o['first_seen_ts'])}–{_fmt_ts(o['last_seen_ts'])}), "
              f"{o['frames_seen']} frames, threat {o['threat_level']} "
              f"({o['threat_score']:.1f}).")
    cites = [
        f"path length {o['distance_px']:.0f}px",
        f"{o['stop_count']} stops, {o['direction_changes']} direction changes",
        f"re-identified {o['reid_count']}×, "
        f"{o['interaction_count']} interactions",
    ]
    return Evidence(answer, cites, rows, sql)


def _h_mission_summary(store, mid: int, slots: dict) -> Evidence:
    stats = store.mission_stats(mid)
    timeline = store.timeline(mid, limit=40)
    events = stats.get("events", {}) or {}
    ev_str = ", ".join(f"{_pretty(k)}×{v}" for k, v in events.items()) \
        or "no significant events"
    answer = (f"Mission tracked {stats.get('unique_objects', 0)} unique object(s) "
              f"({stats.get('unique_people', 0)} people). Events: {ev_str}.")
    cites = [f"{_fmt_ts(r['source_ts'])} — {r['description']}"
             for r in timeline[:20]]
    return Evidence(answer, cites, timeline, "(mission_stats + timeline)")


# --------------------------------------------------------------------------- #
#  Intent registry
# --------------------------------------------------------------------------- #
INTENTS: Dict[str, Intent] = {
    i.name: i for i in [
        Intent("count_people",
               "how many unique/distinct people or persons were seen",
               ["how many people were there?",
                "count the distinct individuals"],
               _h_count_people),
        Intent("count_objects",
               "how many unique objects (optionally of a class, e.g. vehicles)",
               ["how many vehicles?", "number of cars detected"],
               _h_count_objects),
        Intent("longest_present",
               "which object/vehicle/person was present the longest",
               ["which vehicle stayed the longest?",
                "longest-present person"],
               _h_longest_present),
        Intent("recent_activity",
               "what happened recently / in the last N minutes or seconds",
               ["what happened in the last 5 minutes?",
                "any activity in the last 30 seconds?"],
               _h_recent_activity),
        Intent("thermal_events",
               "thermal / infrared / IR / heat-signature events",
               ["show all thermal events", "any IR heat detections?"],
               _h_thermal_events),
        Intent("most_direction_changes",
               "which object changed direction the most (erratic/evasive movement)",
               ["who changed direction the most?",
                "most erratic mover"],
               _h_most_direction_changes),
        Intent("suspicious_activities",
               "list suspicious activity, alerts, or flagged behaviour",
               ["list suspicious activities", "anything suspicious happen?"],
               _h_suspicious),
        Intent("threat_assessment",
               "overall threat level or the highest-threat object",
               ["what is the threat level?", "biggest threat in the scene"],
               _h_threat_assessment),
        Intent("object_detail",
               "details about one specific object/track by id",
               ["tell me about person 3", "describe track 12"],
               _h_object_detail),
        Intent("mission_summary",
               "a general summary / overview of the whole mission",
               ["summarize the mission", "give me an overview"],
               _h_mission_summary),
    ]
}


# --------------------------------------------------------------------------- #
#  Deterministic slot parsers + keyword fallback classifier
# --------------------------------------------------------------------------- #
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_TIME_RE = re.compile(
    r"(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s*"
    r"(seconds?|secs?|minutes?|mins?|hours?|hrs?|[smh])\b")
_ID_RE = re.compile(
    r"(?:#|\bid\b|\btrack\b|\bobject\b|\bperson\b|\bvehicle\b|\bcar\b)\s*#?\s*"
    r"(\d{1,6})\b")


def parse_time_window(question: str) -> Optional[float]:
    """Extract a time window in seconds from phrases like 'last 5 minutes'."""
    m = _TIME_RE.search(question.lower())
    if not m:
        return None
    raw, unit = m.group(1), m.group(2)
    val = float(_WORD_NUM[raw]) if raw in _WORD_NUM else float(raw)
    if unit in ("s", "sec", "secs", "second", "seconds"):
        return val
    if unit in ("m", "min", "mins", "minute", "minutes"):
        return val * 60.0
    return val * 3600.0  # hours


def parse_global_id(question: str) -> Optional[int]:
    """Extract an explicit object/track id, e.g. 'person 3' or '#12'."""
    m = _ID_RE.search(question.lower())
    return int(m.group(1)) if m else None


_PERSON_WORDS = ("people", "persons", "person", "pedestrians", "pedestrian",
                 "individuals", "individual", "men", "man", "women", "woman")


def _has_word(text: str, word: str) -> bool:
    # Word-boundary match so e.g. "man" doesn't fire inside "many".
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def parse_class(question: str) -> Optional[str]:
    """Extract a class slot; returns the 'vehicle' sentinel for the vehicle group."""
    s = question.lower()
    if _has_word(s, "vehicle") or _has_word(s, "vehicles"):
        return "vehicle"
    if any(_has_word(s, w) for w in _PERSON_WORDS):
        return "person"
    for c in KNOWN_CLASSES:
        if _has_word(s, c) or _has_word(s, c + "s"):
            return "vehicle" if c in VEHICLE_CLASSES else c
    return None


def classify_keywords(question: str) -> str:
    """Rule-based intent routing — the fallback when the VLM is unavailable."""
    s = question.lower()
    if "suspicious" in s:
        return "suspicious_activities"
    if any(w in s for w in ("threat", "danger", "hostile", "risk level")):
        return "threat_assessment"
    if any(w in s for w in ("thermal", "infrared", "ir ", "heat signature",
                            "heat ")):
        return "thermal_events"
    if any(w in s for w in ("direction", "erratic", "evasive", "zigzag",
                            "zig-zag")):
        return "most_direction_changes"
    if any(w in s for w in ("longest", "stayed the most", "most time",
                            "longest-present", "present the longest")):
        return "longest_present"
    if "last" in s and parse_time_window(question) is not None:
        return "recent_activity"
    if any(w in s for w in ("how many", "count", "number of", "how much")):
        if parse_class(question) == "person" or not parse_class(question):
            if any(w in s for w in ("people", "person", "individual", "men")):
                return "count_people"
        return "count_objects"
    if (parse_global_id(question) is not None
            and any(w in s for w in ("about", "detail", "describe", "track",
                                     "object", "tell me", "id"))):
        return "object_detail"
    if any(w in s for w in ("summar", "overview", "what happened", "report",
                            "recap", "brief")):
        return "mission_summary"
    return "mission_summary"
