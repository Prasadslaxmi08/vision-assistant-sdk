"""Timeline tab — chronological mission record from persisted memory.

Merges events, spatial interactions, EO/IR thermal/fusion, and VLM observations
into one filterable, time-ordered table for the selected mission.
"""
from __future__ import annotations

import streamlit as st

from ui.common import fmt_clock, require_mission
from ui.model_manager import get_memory_store

_KINDS = {"event": "🟦 Events", "interaction": "🟪 Interactions",
          "fusion": "🌡️ Thermal", "summary": "🧠 Intelligence"}


def render(mission_id: int | None) -> None:
    st.header("🕒 Mission Timeline")
    if not require_mission(mission_id):
        return
    store = get_memory_store()

    events = store.read(
        "SELECT source_ts, type, description FROM events WHERE mission_id=? "
        "ORDER BY source_ts", (mission_id,))
    interactions = store.interactions(mission_id)
    thermal = store.thermal_events(mission_id)
    vlm = store.read(
        "SELECT source_ts, kind, text FROM vlm_observations WHERE mission_id=? "
        "ORDER BY source_ts", (mission_id,))

    rows = []
    rows += [{"time": fmt_clock(e["source_ts"]), "kind": "event",
              "type": e["type"], "detail": e["description"]} for e in events]
    rows += [{"time": fmt_clock(it["source_ts"]), "kind": "interaction",
              "type": it["type"], "detail": it["description"]}
             for it in interactions]
    rows += [{"time": fmt_clock(t["source_ts"]), "kind": "fusion",
              "type": "THERMAL", "detail": t["description"]} for t in thermal]
    rows += [{"time": fmt_clock(v["source_ts"]), "kind": "summary",
              "type": "INTEL_" + str(v["kind"]).upper(), "detail": v["text"]}
             for v in vlm]

    if not rows:
        st.info("No timeline entries recorded for this mission yet.")
        return

    c1, c2 = st.columns([3, 1])
    with c1:
        shown = st.multiselect(
            "Show", options=list(_KINDS), default=list(_KINDS),
            format_func=lambda k: _KINDS[k], key="tl_kinds")
    with c2:
        search = st.text_input("Filter text", key="tl_search").strip().lower()

    rows = [r for r in rows if r["kind"] in shown]
    if search:
        rows = [r for r in rows if search in r["detail"].lower()
                or search in r["type"].lower()]
    rows.sort(key=lambda r: r["time"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Events", len(events))
    m2.metric("Interactions", len(interactions))
    m3.metric("Thermal", len(thermal))
    m4.metric("Intelligence notes", len(vlm))

    st.dataframe(rows, use_container_width=True, hide_index=True,
                 column_config={
                     "time": st.column_config.TextColumn("Time", width="small"),
                     "kind": st.column_config.TextColumn("Kind", width="small"),
                     "type": st.column_config.TextColumn("Type", width="medium"),
                     "detail": st.column_config.TextColumn("Detail", width="large"),
                 })
    st.caption(f"{len(rows)} of {len(events)+len(interactions)+len(thermal)+len(vlm)} "
               "entries shown.")
