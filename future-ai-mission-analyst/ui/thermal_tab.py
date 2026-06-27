"""Thermal tab — EO/IR fusion / thermal events from persisted memory.

Surfaces the FusionEngine output (running engines, thermal-confirmed targets,
concealed heat) recorded for the selected mission, with a spatial scatter of
hotspot locations.
"""
from __future__ import annotations

import streamlit as st

from ui.common import fmt_clock, require_mission
from ui.model_manager import get_memory_store


def render(mission_id: int | None) -> None:
    st.header("🌡️ Thermal / EO-IR Fusion")
    st.caption("Heat-signature correlations: running engines, warm/active "
               "targets, and concealed heat sources beyond the visual picture.")
    if not require_mission(mission_id):
        return
    store = get_memory_store()
    thermal = store.thermal_events(mission_id)
    if not thermal:
        st.info("No thermal/IR fusion events recorded for this mission. "
                "(Thermal correlation requires a live IR feed.)")
        return

    correlated = sum(1 for t in thermal if t["correlated_global_id"])
    m1, m2, m3 = st.columns(3)
    m1.metric("Thermal events", len(thermal))
    m2.metric("Correlated to objects", correlated)
    m3.metric("Concealed / uncorrelated", len(thermal) - correlated)

    table = [{
        "time": fmt_clock(t["source_ts"]),
        "area_px": t["area_px"],
        "object": (f"#{t['correlated_global_id']}"
                   if t["correlated_global_id"] else "—"),
        "detail": t["description"],
    } for t in thermal]
    st.dataframe(table, use_container_width=True, hide_index=True)

    pts = [t for t in thermal if t["x"] is not None and t["y"] is not None]
    if pts:
        st.markdown("**Hotspot locations** (frame coordinates)")
        st.scatter_chart(
            {"x": [t["x"] for t in pts], "y": [t["y"] for t in pts],
             "area": [t["area_px"] or 1 for t in pts]},
            x="x", y="y", size="area", height=260)
