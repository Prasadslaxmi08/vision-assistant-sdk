"""Object Explorer tab — per-identity lifecycle from persisted memory.

Lists every tracked identity (stable ReID global id) for the selected mission
with its lifecycle aggregates, threat level, and best thumbnail, and drills into
one identity's movement path and interactions.
"""
from __future__ import annotations

import streamlit as st

from ui.common import fmt_clock, require_mission, safe_image
from ui.model_manager import get_memory_store

_LEVEL_COLOR = {"NONE": "⚪", "LOW": "🟢", "MEDIUM": "🟡",
                "HIGH": "🟠", "CRITICAL": "🔴"}


def render(mission_id: int | None) -> None:
    st.header("🧭 Object Explorer")
    if not require_mission(mission_id):
        return
    store = get_memory_store()
    objects = store.list_objects(mission_id)
    if not objects:
        st.info("No tracked objects recorded for this mission yet.")
        return

    classes = sorted({o["class_name"] for o in objects if o["class_name"]})
    c1, c2 = st.columns([2, 1])
    with c1:
        pick = st.multiselect("Class filter", classes, default=classes,
                              key="obj_classes")
    with c2:
        only_threat = st.toggle("Threat > 0 only", value=False, key="obj_threat")

    shown = [o for o in objects if o["class_name"] in pick
             and (not only_threat or (o.get("threat_score") or 0) > 0)]

    table = [{
        "id": o["global_id"],
        "class": o["class_name"],
        "threat": f"{_LEVEL_COLOR.get(o['threat_level'], '')} {o['threat_level']}"
                  f" ({o['threat_score']:.1f})",
        "first": fmt_clock(o["first_seen_ts"]),
        "last": fmt_clock(o["last_seen_ts"]),
        "dwell": fmt_clock((o["last_seen_ts"] or 0) - (o["first_seen_ts"] or 0)),
        "frames": o["frames_seen"],
        "dir_chg": o["direction_changes"],
        "stops": o["stop_count"],
        "reid": o["reid_count"],
        "path_px": round(o["distance_px"] or 0.0),
    } for o in sorted(shown, key=lambda o: o["threat_score"] or 0, reverse=True)]

    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption(f"{len(shown)} of {len(objects)} identities.")

    st.divider()
    _detail(store, mission_id, shown)


def _detail(store, mission_id: int, objects: list) -> None:
    if not objects:
        return
    ids = [o["global_id"] for o in objects]
    by_id = {o["global_id"]: o for o in objects}
    gid = st.selectbox(
        "Inspect identity", ids, key="obj_detail_id",
        format_func=lambda i: f"#{i} · {by_id[i]['class_name']} "
                              f"[{by_id[i]['threat_level']}]")
    o = by_id[gid]

    head, thumb = st.columns([3, 1])
    with head:
        st.subheader(f"{_LEVEL_COLOR.get(o['threat_level'], '')} "
                     f"{o['class_name']} #{gid}")
        a, b, c = st.columns(3)
        a.metric("Threat", o["threat_level"], f"{o['threat_score']:.1f}")
        b.metric("Frames seen", o["frames_seen"])
        c.metric("Path length", f"{round(o['distance_px'] or 0)} px")
        a.metric("Direction changes", o["direction_changes"])
        b.metric("Stops", o["stop_count"])
        c.metric("Re-identified", f"{o['reid_count']}×")
    with thumb:
        if o.get("snapshot"):
            safe_image(thumb, o["snapshot"], caption=f"#{gid}")
        else:
            st.caption("No thumbnail captured.")

    track = store.object_track(gid, mission_id)
    if track:
        st.markdown("**Movement path** (sampled detection centres)")
        st.scatter_chart({"x": [t["cx"] for t in track],
                          "y": [t["cy"] for t in track]}, x="x", y="y",
                         height=240)

    inter = store.read(
        "SELECT source_ts, type, object_id, description FROM interactions "
        "WHERE mission_id=? AND (subject_id=? OR object_id=?) ORDER BY source_ts",
        (mission_id, gid, gid))
    if inter:
        st.markdown("**Interactions**")
        st.dataframe(
            [{"time": fmt_clock(i["source_ts"]), "type": i["type"],
              "detail": i["description"]} for i in inter],
            use_container_width=True, hide_index=True)
