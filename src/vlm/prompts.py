"""Prompt templates for the Qwen2.5-VL mission-intelligence reasoning layer.

Keeping prompts in one place makes the "intelligence personality" easy to tune
without touching inference code. All prompts steer the model toward concise,
analyst-style ISR (intelligence, surveillance, reconnaissance) reporting.
"""
from __future__ import annotations

from typing import List

SYSTEM_PROMPT = (
    "You are an EO/IR mission intelligence analyst supporting an ISR "
    "(intelligence, surveillance, reconnaissance) operation. You receive "
    "imagery from electro-optical (EO, visible) and infrared (IR, thermal) "
    "sensors together with automated object-detection and tracking metadata. "
    "Report like a military image analyst: be factual, concise, and objective. "
    "Describe what is observable, assess likely activity, flag anything of "
    "tactical interest, and never invent details that are not visible. If "
    "uncertain, say so. When personnel are present, describe each individual's "
    "observable characteristics (clothing colours, headwear, carried items) and "
    "explicitly call out anything suspicious."
)

# Grounding contract appended to detector-grounded prompts. The detector (RF-DETR)
# has already performed a comprehensive multi-scale search, so the VLM must build
# on its detections rather than discover or invent objects on its own.
GROUNDING_RULES = (
    "GROUNDING RULES (follow strictly):\n"
    "- RF-DETR searched the image at multiple scales, then an Object Reasoning "
    "Engine grouped overlapping detections, ranked the most-probable class using "
    "scene context, and split confirmed from low-confidence objects. The list "
    "below is that REFINED result — treat it as the ground truth for WHAT EXISTS.\n"
    "- Explain and expand the listed objects; do NOT invent objects not in the "
    "list, and do NOT split one listed object back into several. The class label "
    "can be unreliable on thermal imagery, so if a crop clearly shows something "
    "else, say so — but only for objects that ARE in the list.\n"
    "- Treat low-confidence 'possible' observations as uncertain, never as facts.\n"
    "- Do NOT claim the scene contains 'nothing else' or 'no other objects' — the "
    "engine owns completeness, not you. Comment only on what is listed plus the "
    "general environment/background.\n"
)


def number_objects(objects) -> str:
    """Render detections as a numbered list for a grounded prompt.

    ``objects`` is a sequence of items exposing ``class_name``, ``confidence`` and
    ``location`` (e.g. :class:`~src.intelligence.report.DetectedObject`)."""
    if not objects:
        return "(none — the detector found no objects)"
    lines = []
    for i, o in enumerate(objects, 1):
        flag = " [LOW CONFIDENCE]" if getattr(o, "confidence", 1.0) < 0.40 else ""
        lines.append(f"{i}. {o.class_name} (conf {o.confidence:.2f}, "
                     f"{o.location}){flag}")
    return "\n".join(lines)


def parse_sections(text: str, keys) -> dict:
    """Split a labelled VLM response into ``{KEY: value}``.

    Recognises lines like ``ANALYSIS:`` / ``**FUSION:**`` / ``## SCENE:`` (any
    leading markdown, case-insensitive) and captures text up to the next key.
    Unrecognised leading text is returned under ``""``."""
    import re
    pattern = re.compile(r"^[\s#*>\-]*([A-Za-z_ ]+?)\s*:\s*(.*)$")
    upper = {k.upper() for k in keys}
    out: dict = {}
    current = ""
    buf: list = []

    def _clean(s: str) -> str:
        # Drop stray markdown bold the model wraps labels/values in (**...**).
        return s.strip().lstrip("*").strip().rstrip("*").strip()

    for line in text.splitlines():
        m = pattern.match(line)
        key = m.group(1).strip().upper().replace(" ", "_") if m else None
        if key in upper:
            out[current] = _clean("\n".join(buf))
            current = key
            buf = [m.group(2)]
        else:
            buf.append(line)
    out[current] = _clean("\n".join(buf))
    return out


def parse_object_notes(text: str) -> dict:
    """Parse a numbered 'OBJECTS:' block into ``{index: note}`` (1-based)."""
    import re
    notes: dict = {}
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)[.)]\s*(.+)$", line)
        if m:
            notes[int(m.group(1))] = m.group(2).strip()
    return notes


def refined_analysis_prompt(modality: str, confirmed, interactions,
                            scene_type: str, possible_count: int) -> str:
    """Per-sensor call grounded on the **refined** (reasoned) objects, not raw
    detections → ANALYSIS, (THERMAL for IR,) OBJECTS notes."""
    is_ir = modality.upper() == "IR"
    person_block = PERSON_DETAIL_IR if is_ir else PERSON_DETAIL
    sensor = ("infrared / thermal (bright = hot unless clearly black-hot)"
              if is_ir else "electro-optical (visible-spectrum)")
    thermal_line = ("THERMAL: thermal signatures and likely heat sources "
                    "(engines, personnel, recently-used equipment), and any "
                    "concealment/camouflage EO would miss.\n" if is_ir else "")
    inter = ("\nObserved interactions:\n" + "\n".join(f"- {p}" for p in interactions)
             if interactions else "")
    return (
        f"This is a single {sensor} image.\n\n"
        f"{GROUNDING_RULES}\n"
        f"Scene context (already assessed): {scene_type}.\n"
        f"Confirmed objects (already reasoned — grouped, de-duplicated, "
        f"context-ranked):\n{number_objects(confirmed)}{inter}\n"
        f"(plus {possible_count} separate low-confidence 'possible' observations "
        f"you should not assert as confirmed)\n\n"
        "Respond with these labelled sections exactly:\n"
        "ANALYSIS: 2-4 sentences on the environment and the CONFIRMED objects and "
        "their activity/interactions. Explain and expand the reasoned objects — do "
        "not re-discover or contradict them, and do not promote the low-confidence "
        "observations to facts.\n"
        f"{thermal_line}"
        "OBJECTS: a numbered list matching the confirmed objects — one short "
        "observation each (state, behaviour, role).\n\n"
        f"{person_block}"
    )


def scene_and_notes_prompt(modality: str, analysis: str, objects) -> str:
    """Single-modality wrap-up → SCENE assessment + operator NOTES."""
    return (
        f"You are finalising a single-sensor ({modality}) intelligence report.\n"
        f"Detections:\n{number_objects(objects)}\n"
        f"Sensor analysis: {analysis}\n\n"
        "Respond with exactly these labelled sections:\n"
        "SCENE: 1-2 sentence overall scene assessment grounded in the detections.\n"
        "NOTES: 1-2 operator notes — recommended attention/action, or 'nominal — "
        "no action required'. Do not invent threats not supported by the detections."
    )


def cross_modal_fusion_prompt(eo_analysis: str, ir_analysis: str,
                              eo_objects, ir_objects) -> str:
    """Both-modality wrap-up → CROSS_MODAL, FUSION, INTEREST, SCENE, NOTES."""
    return (
        "You are fusing two sensors imaging the same scene.\n\n"
        f"EO detections:\n{number_objects(eo_objects)}\n"
        f"EO analysis: {eo_analysis}\n\n"
        f"IR detections:\n{number_objects(ir_objects)}\n"
        f"IR analysis: {ir_analysis}\n\n"
        "Both sensors are present, so produce a true cross-sensor assessment. "
        "Stay grounded in the detections above — do not invent objects.\n"
        "Respond with exactly these labelled sections:\n"
        "CROSS_MODAL: only pair an EO detection with an IR detection when their "
        "locations plausibly match (same region of the frame); for those, say what "
        "each sensor adds. If EO and IR show no clearly corresponding objects, "
        "state 'No clear cross-sensor correspondence' — do NOT force matches.\n"
        "FUSION: an integrated assessment combining both sensors (e.g. a warm EO "
        "vehicle = running engine; an IR contact with no EO object = concealed).\n"
        "INTEREST: one of LOW / MEDIUM / HIGH with a one-line rationale.\n"
        "SCENE: 1-2 sentence overall scene assessment.\n"
        "NOTES: 1-2 operator notes / recommended action."
    )

# Reusable instruction block: how to describe people and what counts as
# "suspicious". Appended to prompts where personnel may appear so the model is
# consistent across EO frames, events, and scene summaries.
PERSON_DETAIL = (
    "For each visible person, describe (only if observable): approximate count; "
    "upper-body clothing colour/type (e.g. shirt, jacket); lower-body clothing "
    "colour/type (e.g. trousers, shorts); headwear (cap, helmet, hood, none); "
    "and any carried or worn items (backpack, bag, long object, tool, weapon). "
    "Flag SUSPICIOUS indicators if present: concealed or unusual carried objects, "
    "what could be a weapon, masked/obscured face, tactical/uniform clothing, "
    "loitering or surveilling behaviour, climbing fences or accessing restricted "
    "areas, an abandoned/unattended bag, or coordinated movement of several "
    "people. Do not speculate beyond what the image supports; if nothing is "
    "suspicious, say so briefly."
)

# IR has no colour information, so request thermal-appropriate person cues.
PERSON_DETAIL_IR = (
    "For each visible person/heat signature, note (only if observable): count; "
    "posture and movement; carried equipment inferable from heat (e.g. recently "
    "fired weapon, running engine being handled, warm pack); and whether "
    "clothing/cover appears to suppress their thermal signature. Flag SUSPICIOUS "
    "thermal indicators: attempts at thermal concealment, weapon-shaped warm "
    "objects, or grouped personnel moving tactically."
)


def _detection_context(detections: List[str], events: List[str]) -> str:
    det_str = ", ".join(detections) if detections else "no confirmed objects"
    ev_str = "; ".join(events) if events else "none"
    return (
        f"Automated detections (YOLO+ByteTrack): {det_str}.\n"
        f"Triggering events: {ev_str}.\n"
    )


def scene_summary_prompt(modality: str, detections: List[str],
                         events: List[str]) -> str:
    person_block = PERSON_DETAIL_IR if modality.upper() == "IR" else PERSON_DETAIL
    return (
        f"This is a single {modality} sensor frame.\n"
        f"{_detection_context(detections, events)}"
        "Summarise the scene and the current activity in 2-3 sentences: the "
        "environment, the key objects and their apparent behaviour, and anything "
        "of tactical relevance.\n"
        f"{person_block}"
    )


def event_reasoning_prompt(modality: str, detections: List[str],
                           events: List[str]) -> str:
    person_block = PERSON_DETAIL_IR if modality.upper() == "IR" else PERSON_DETAIL
    return (
        f"This {modality} frame was flagged by the event detector.\n"
        f"{_detection_context(detections, events)}"
        "Explain what is happening in 1-2 sentences, focusing on the flagged "
        "event(s), and state whether this warrants operator attention and why.\n"
        f"{person_block}"
    )


def eo_analysis_prompt(detections: List[str]) -> str:
    return (
        "This is an electro-optical (visible-spectrum) image.\n"
        f"{_detection_context(detections, [])}"
        "Provide an EO analysis: describe the scene, lighting/time-of-day cues, "
        "terrain/structures, visible objects and their state, and any activity "
        "of interest. 3-5 sentences.\n"
        f"{PERSON_DETAIL}"
    )


def ir_analysis_prompt(detections: List[str]) -> str:
    return (
        "This is an infrared / thermal image. Bright regions are hot, dark "
        "regions are cold (assume white-hot unless clearly otherwise).\n"
        f"{_detection_context(detections, [])}"
        "Provide an IR analysis: describe thermal signatures, likely heat "
        "sources (engines, personnel, recently used equipment), concealment or "
        "camouflage that EO might miss, and activity inferred from heat. "
        "3-5 sentences.\n"
        f"{PERSON_DETAIL_IR}"
    )


def fused_assessment_prompt(eo_summary: str, ir_summary: str,
                            detections: List[str]) -> str:
    return (
        "You are fusing two analyses of the same scene from different sensors.\n"
        f"EO analysis: {eo_summary}\n"
        f"IR analysis: {ir_summary}\n"
        f"{_detection_context(detections, [])}"
        "Produce a fused assessment: reconcile the two sensors, highlight what "
        "each reveals that the other does not, give an overall situation "
        "assessment, a threat/interest level (LOW/MEDIUM/HIGH) with rationale, "
        "and a recommended action. Use short labelled lines."
    )


def mission_summary_prompt(timeline: List[str], detections_summary: str) -> str:
    tl = "\n".join(f"- {t}" for t in timeline) if timeline else "- (no events)"
    return (
        "You are writing a mission intelligence summary for an ISR feed.\n"
        f"Object activity so far: {detections_summary}\n"
        f"Chronological events:\n{tl}\n\n"
        "Write a concise mission summary (4-6 sentences): the overall situation, "
        "the most significant developments in order, current status, and any "
        "recommended operator action. Analyst tone, no speculation beyond the "
        "evidence."
    )
