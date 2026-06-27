# future-ai-mission-analyst (staging)

This folder holds code **moved out** of the Vision Assistant SDK during the
2026-06-25 refactor. It is **not imported** by the SDK and does not run from here —
it is preserved verbatim to seed a separate **AI Mission Analyst** repository later.

The Vision Assistant SDK is now scoped to Computer Vision + Multimodal AI only
(detection, tracking, EO/IR fusion, vision-language reasoning). Everything that is
"mission management / long-term intelligence" lives here:

| Path | What it is |
|---|---|
| `src/memory/` | SQLite persistent mission memory (`mission_store.py`, `schema.py`) |
| `src/query/` | Natural-language query assistant over mission memory |
| `src/reasoning/threat.py` | Threat & anomaly scoring |
| `src/intelligence/mission_report.py` | Report reconstruction from the archive |
| `src/intelligence/exporters.py` | PDF / Excel report rendering |
| `src/contracts/store.py` | `StoreWriter` / `StoreReader` DB seam |
| `ui/*.py` | Streamlit panels: timeline, objects, thermal, query, reports |

### How these reconnect in the future repo
The boundary is the **mission record**, not Python imports. The SDK produces
structured per-frame results, events, fusion assessments and VLM summaries (JSON /
callbacks). AI Mission Analyst consumes that stream (or a recording of it) and owns
persistence, threat synthesis, RAG, querying, agentic workflows and reporting.

See `docs/architecture-review/06-vision-assistant-sdk-refactor.md` in the parent
repo for the full plan, and `04-repository-split-plan.md` for the original seam map.
