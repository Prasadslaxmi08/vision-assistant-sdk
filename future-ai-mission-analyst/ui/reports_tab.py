"""Reports tab — enriched mission reports from persisted memory (phase 6).

Reconstructs a full report for any archived mission (events, interactions,
thermal/fusion, per-identity threat, object gallery) and optionally a grounded
natural-language Q&A section from the Query Assistant, then offers Markdown /
JSON / Excel / PDF downloads and writes them to the output directory.
"""
from __future__ import annotations

import streamlit as st

from src.intelligence import mission_report
from ui.common import require_mission, safe_image
from ui.model_manager import (get_app_config, get_memory_store,
                              get_query_assistant)

_SPECS = [
    ("pdf", "⬇️ PDF", "pdf", "application/pdf"),
    ("excel", "⬇️ Excel (.xlsx)", "xlsx",
     "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ("markdown", "⬇️ Markdown", "md", "text/markdown"),
    ("json", "⬇️ JSON", "json", "application/json"),
]


def render(mission_id: int | None) -> None:
    st.header("📤 Mission Reports")
    st.caption("Generate an analyst-ready report from the archived mission — "
               "timeline, statistics, threat overview, object gallery, and an "
               "optional grounded Q&A brief.")
    if not require_mission(mission_id):
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        include_qa = st.toggle("Include Q&A brief", value=True, key="rep_qa",
                               help="Auto-answer a few standard questions and "
                                    "embed them in the report.")
    with c2:
        qa_vlm = st.toggle("Narrative phrasing", value=False, key="rep_qa_vlm",
                           disabled=not include_qa,
                           help="Phrase the Q&A brief in natural language (slower).")

    if st.button("🛠️ Build report", type="primary", key="rep_build"):
        with st.spinner("Reconstructing mission report from memory…"):
            store = get_memory_store()
            qa = get_query_assistant(qa_vlm) if include_qa else None
            data = mission_report.gather(store, mission_id, qa_assistant=qa)
            blobs = mission_report.to_all_bytes(data)
            cfg = get_app_config()
            out_dir = cfg.abs_path(cfg.system.output_dir)
            paths = mission_report.write_all(data, out_dir,
                                             name_prefix=f"mission{mission_id}",
                                             blobs=blobs)
        st.session_state["report_data"] = data
        st.session_state["report_blobs"] = blobs
        st.session_state["report_paths"] = paths

    data = st.session_state.get("report_data")
    blobs = st.session_state.get("report_blobs")
    if not data or data.mission_id != mission_id:
        st.info("Press **Build report** to generate downloadable files.")
        return

    _preview(data)
    _downloads(blobs, mission_id)
    paths = st.session_state.get("report_paths") or {}
    if paths:
        st.caption("Saved to outputs/: "
                   + ", ".join(p.split("/")[-1].split("\\")[-1]
                               for p in paths.values()))


def _preview(data) -> None:
    s = data.stats
    st.success(data.final_summary)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Unique objects", s.get("unique_objects", 0))
    m2.metric("People", s.get("unique_people", 0))
    m3.metric("Interactions", s.get("interactions", 0))
    m4.metric("Thermal events", s.get("thermal_events", 0))

    if data.qa:
        with st.expander("🧠 Q&A brief", expanded=True):
            for q, a in data.qa:
                st.markdown(f"**Q: {q}**")
                st.markdown(a)
                st.divider()
    if data.snapshots:
        with st.expander(f"🖼️ Object gallery ({len(data.snapshots)})",
                         expanded=False):
            cols = st.columns(6)
            for i, snap in enumerate(data.snapshots[:36]):
                safe_image(cols[i % 6], snap["jpeg"],
                           caption=f"#{snap['identity']} {snap['class_name']}")
    with st.expander(f"🕒 Timeline ({len(data.timeline)})", expanded=False):
        st.dataframe(data.timeline, use_container_width=True, hide_index=True)


def _downloads(blobs: dict, mission_id: int) -> None:
    st.markdown("**Download**")
    cols = st.columns(len(_SPECS))
    for col, (kind, label, ext, mime) in zip(cols, _SPECS):
        blob = (blobs or {}).get(kind)
        fname = f"mission{mission_id}_report.{ext}"
        if blob:
            col.download_button(label, blob, file_name=fname, mime=mime,
                                key=f"dl_rep_{kind}", use_container_width=True)
        else:
            col.button(label, disabled=True, key=f"dl_rep_{kind}_x",
                       use_container_width=True,
                       help="Format unavailable (optional dependency missing).")
