"""Mission Query tab — natural-language Q&A over mission memory (phase 5 → UI).

Wires :class:`~src.query.assistant.QueryAssistant` to the selected archived
mission. The VLM toggle lets an operator choose between instant deterministic
answers (no model load) and Qwen2.5-VL-reworded answers (reuses the shared 7B
model — no second model is ever loaded).
"""
from __future__ import annotations

import streamlit as st

from ui.common import require_mission
from ui.model_manager import get_query_assistant

_SUGGESTIONS = [
    "How many unique people were there?",
    "Which vehicle was present the longest?",
    "What happened in the last 5 minutes?",
    "List suspicious activities",
    "Who changed direction the most?",
    "Summarize the mission",
]


def render(mission_id: int | None) -> None:
    st.header("💬 Mission Query Assistant")
    st.caption("Ask questions about the selected mission. Answers are grounded "
               "in mission memory — every figure is computed from the database, "
               "never invented.")
    if not require_mission(mission_id):
        return

    use_vlm = st.toggle(
        "Phrase answers in natural language (slower)",
        value=False, key="query_use_vlm",
        help="Off = instant deterministic answers with citations. "
             "On = the intelligence assistant reclassifies and rewords the "
             "grounded answer (slower).")

    with st.form("query_form", clear_on_submit=False):
        question = st.text_input("Your question", key="query_text",
                                 placeholder="e.g. How many vehicles were seen?")
        submitted = st.form_submit_button("Ask", type="primary")

    st.caption("Try:")
    cols = st.columns(3)
    for i, sug in enumerate(_SUGGESTIONS):
        if cols[i % 3].button(sug, key=f"sug_{i}", use_container_width=True):
            question, submitted = sug, True

    if submitted and question:
        spinner = ("Generating narrative answer…" if use_vlm
                   else "Querying mission memory…")
        with st.spinner(spinner):
            qa = get_query_assistant(use_vlm)
            result = qa.ask(question, mission_id)
        _render_result(result)

    _render_history(question if submitted else None)


def _render_result(result) -> None:
    hist = st.session_state.setdefault("query_history", [])
    hist.insert(0, result)
    del hist[8:]  # keep the last few

    if result.grounded:
        st.success(result.answer)
    else:
        st.warning(result.answer)

    meta = f"intent: `{result.intent}`"
    active = {k: v for k, v in (result.slots or {}).items() if v is not None}
    if active:
        meta += "  ·  slots: " + ", ".join(f"`{k}={v}`" for k, v in active.items())
    st.caption(meta)

    if result.evidence.citations:
        with st.expander(f"📎 Evidence ({len(result.evidence.citations)})",
                         expanded=True):
            for c in result.evidence.citations:
                st.markdown(f"- {c}")
    if result.evidence.sql:
        with st.expander("🛢️ Query (read-only, parameterised)", expanded=False):
            st.code(result.evidence.sql, language="sql")


def _render_history(_just_asked) -> None:
    hist = st.session_state.get("query_history", [])
    if len(hist) <= 1:
        return
    with st.expander("🕘 Recent questions", expanded=False):
        for r in hist[1:]:
            st.markdown(f"**Q:** {r.question}")
            st.markdown(f"**A:** {r.answer.splitlines()[0]}")
            st.divider()
