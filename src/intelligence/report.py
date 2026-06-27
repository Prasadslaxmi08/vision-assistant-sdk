"""Structured operational intelligence report (refined, not raw).

The report presents the **Object Reasoning Engine's** refined conclusions, not the
raw detector echo. Sections adapt to the available sensors and follow an operator
layout:

  * Sensor Analysis (per available modality)
  * Confirmed Objects          (grouped, deduplicated, context-ranked)
  * Contextual Interpretation  (scene type + how it refined the ranking)
  * Human-Object Interactions  ("one person appears to be holding a …")
  * Possible Objects (Low Confidence)   (uncertain observations, not asserted)
  * Thermal Observations       (IR present)
  * Scene Assessment
  * Cross-Modal Correlation / Fusion Assessment   (only when EO **and** IR present)
  * Confidence Levels
  * Operator Notes

Raw detector outputs are retained on each :class:`SensorSection` (``result.raw``)
and surfaced via :meth:`IntelReport.to_dict` for developer mode / debugging — they
are not part of the default operational narrative.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.reasoning.object_reasoning import ReasonedObject, ReasoningResult

# Detector confidence below which an object is 'possible' rather than confirmed.
UNCERTAIN_BELOW = 0.45


def describe_location(bbox: Tuple[float, float, float, float],
                      img_w: int, img_h: int) -> str:
    """Map a bbox centre to a human 3x3 grid cell, e.g. 'lower-right'."""
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    col = 0 if cx < img_w / 3 else (1 if cx < 2 * img_w / 3 else 2)
    row = 0 if cy < img_h / 3 else (1 if cy < 2 * img_h / 3 else 2)
    vert = ["upper", "middle", "lower"][row]
    horiz = ["left", "centre", "right"][col]
    if vert == "middle" and horiz == "centre":
        return "centre"
    return f"{vert}-{horiz}"


@dataclass
class SensorSection:
    modality: str                     # "EO" | "IR"
    analysis: str = ""
    result: Optional[ReasoningResult] = None   # refined ORE output for this sensor
    thermal_observations: str = ""    # IR only

    @property
    def confirmed(self) -> List[ReasonedObject]:
        return self.result.confirmed if self.result else []

    @property
    def possible(self) -> List[ReasonedObject]:
        return self.result.possible if self.result else []

    @property
    def interactions(self) -> List[str]:
        return self.result.interactions if self.result else []

    def to_dict(self) -> dict:
        r = self.result
        return {
            "modality": self.modality, "analysis": self.analysis,
            "thermal_observations": self.thermal_observations,
            "scene": (r.scene.scene_type if r else "general"),
            "confirmed_objects": [o.to_dict() for o in self.confirmed],
            "possible_objects": [o.to_dict() for o in self.possible],
            "interactions": self.interactions,
            # Developer mode: the raw detector output, pre-reasoning.
            "raw_detections": [
                {"class": d.class_name, "confidence": round(float(d.confidence), 3),
                 "bbox": [round(float(v), 1) for v in d.bbox]}
                for d in (r.raw if r else [])],
        }


@dataclass
class IntelReport:
    modalities: List[str] = field(default_factory=list)
    sensors: List[SensorSection] = field(default_factory=list)
    scene_assessment: str = ""
    cross_modal: str = ""            # populated only when both modalities present
    fusion_assessment: str = ""      # populated only when both modalities present
    operator_notes: str = ""
    metrics: dict = field(default_factory=dict)

    @property
    def both_modalities(self) -> bool:
        return len(self.modalities) == 2

    def all_confirmed(self) -> List[ReasonedObject]:
        return [o for s in self.sensors for o in s.confirmed]

    def all_possible(self) -> List[ReasonedObject]:
        return [o for s in self.sensors for o in s.possible]

    def all_objects(self) -> List[ReasonedObject]:
        return self.all_confirmed() + self.all_possible()

    # ---------------------------------------------------------- confidence
    def confidence_levels(self) -> str:
        objs = self.all_objects()
        if not objs:
            return "No objects detected — nothing to assess."
        conf = [o.confidence for o in objs]
        avg = sum(conf) / len(conf)
        return ("; ".join([
            f"{len(self.all_confirmed())} confirmed + {len(self.all_possible())} "
            f"possible (low-confidence)",
            f"mean confidence {avg:.2f}",
            f"high (>=0.60): {sum(c >= 0.6 for c in conf)}",
            f"medium (0.45-0.59): {sum(0.45 <= c < 0.6 for c in conf)}",
            f"low (<0.45): {sum(c < 0.45 for c in conf)}"]))

    # ------------------------------------------------------------- render
    def _confirmed_line(self, o: ReasonedObject) -> str:
        line = (f"- **{o.primary_class}** | conf {o.confidence:.2f} | "
                f"{o.location} | {o.size_label}")
        if o.alternatives:
            line += f"\n    - alternative hypotheses: {o.alt_text}"
        if o.interaction:
            line += f"\n    - interaction: {o.interaction}"
        return line

    def _possible_line(self, o: ReasonedObject) -> str:
        alt = f" (or possibly {o.alt_text})" if o.alternatives else ""
        return (f"- A possible **{o.primary_class}**{alt} in the {o.location} "
                f"(conf {o.confidence:.2f}) — confidence insufficient to confirm.")

    def to_markdown(self) -> str:
        L: List[str] = ["# Vision Assistant — Intelligence Report", ""]
        L.append(f"**Sensors:** {' + '.join(self.modalities) or 'none'}")
        if self.metrics:
            m = self.metrics
            L.append(f"**Detector:** {m.get('detector', '?')} | "
                     f"raw detections: {m.get('raw_count', 0)} -> "
                     f"refined: {m.get('object_count', 0)} | "
                     f"inference: {m.get('inference_ms', 0):.0f} ms | "
                     f"mean conf: {m.get('mean_confidence', 0):.2f}")
        L.append("")

        # Sensor Analysis.
        L.append("## Sensor Analysis")
        for s in self.sensors:
            L.append(f"### {s.modality}")
            L.append(s.analysis or "_no analysis_")
            L.append("")

        # Confirmed Objects.
        L.append("## Confirmed Objects")
        if not self.all_confirmed():
            L.append("_No objects met the confirmation threshold._")
        else:
            for s in self.sensors:
                if not s.confirmed:
                    continue
                if self.both_modalities:
                    L.append(f"**{s.modality}:**")
                L += [self._confirmed_line(o) for o in s.confirmed]
        L.append("")

        # Contextual Interpretation.
        L.append("## Contextual Interpretation")
        ctx = []
        for s in self.sensors:
            if s.result:
                sc = s.result.scene
                ctx.append(f"**{s.modality}:** scene assessed as *{sc.scene_type}* "
                           f"({sc.rationale}); class ranking refined accordingly "
                           f"(context refines confidence, it does not add objects).")
        L.append("\n".join(ctx) if ctx else "_no scene cue_")
        L.append("")

        # Human-Object Interactions.
        L.append("## Human-Object Interactions")
        inter = [p for s in self.sensors for p in s.interactions]
        L.append("\n".join(f"- {p}" for p in inter) if inter
                 else "_None identified._")
        L.append("")

        # Possible Objects (Low Confidence).
        L.append("## Possible Objects (Low Confidence)")
        if not self.all_possible():
            L.append("_None._")
        else:
            for s in self.sensors:
                if not s.possible:
                    continue
                if self.both_modalities:
                    L.append(f"**{s.modality}:**")
                L += [self._possible_line(o) for o in s.possible]
        L.append("")

        # Thermal Observations (IR only).
        ir = next((s for s in self.sensors if s.modality == "IR"), None)
        if ir is not None:
            L.append("## Thermal Observations")
            L.append(ir.thermal_observations or ir.analysis or "_none_")
            L.append("")

        # Scene Assessment.
        L.append("## Scene Assessment")
        L.append(self.scene_assessment or "_none_")
        L.append("")

        # Cross-Modal + Fusion — both-modality only.
        if self.both_modalities:
            L.append("## Cross-Modal Correlation")
            L.append(self.cross_modal or "_none_")
            L.append("")
            L.append("## Fusion Assessment")
            L.append(self.fusion_assessment or "_none_")
            L.append("")

        # Confidence Levels.
        L.append("## Confidence Levels")
        L.append(self.confidence_levels())
        L.append("")

        # Operator Notes.
        L.append("## Operator Notes")
        L.append(self.operator_notes or "_none_")
        return "\n".join(L)

    def to_dict(self) -> dict:
        d = {
            "modalities": self.modalities,
            "sensors": [s.to_dict() for s in self.sensors],
            "confirmed_objects": [o.to_dict() for o in self.all_confirmed()],
            "possible_objects": [o.to_dict() for o in self.all_possible()],
            "interactions": [p for s in self.sensors for p in s.interactions],
            "scene_assessment": self.scene_assessment,
            "confidence_levels": self.confidence_levels(),
            "operator_notes": self.operator_notes,
            "metrics": self.metrics,
        }
        if self.both_modalities:
            d["cross_modal_correlation"] = self.cross_modal
            d["fusion_assessment"] = self.fusion_assessment
        return d
