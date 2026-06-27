"""Object Reasoning Engine (ORE) — reason about detections before reporting.

Sits **between RF-DETR and the VLM**. The detector's output is treated as raw
*observations*, not conclusions. The ORE:

  1. **Groups overlapping detections** that likely belong to one physical object
     (high IoU) and picks the most-probable class, keeping the rest as *alternative
     hypotheses* when their confidence is close — so a single object detected as
     "tennis racket / baseball bat / tennis racket" is reported once, not thrice.
  2. **Applies scene context** to refine (never override) the class ranking — a
     racket is more plausible than a bat on a court; a boat more than a car at a
     harbour. Context reweights confidence; it never fabricates objects.
  3. **Splits confirmed vs possible** — low-confidence groups are reported as
     uncertain "possible observations", not asserted objects.
  4. **Infers human-object interactions** — "one person appears to be holding a
     tennis racket" instead of separate "person" and "tennis racket" lines.

Everything here is deterministic and explainable. The refined output (not the raw
detections) is what the report and the VLM consume; the raw detections are kept on
the result for developer mode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.config.settings import ObjectReasoningConfig
from src.utils.logger import logger
from src.utils.types import Detection

# --- class groupings (COCO names) --------------------------------------------
HOLD_CLASSES = {
    "tennis racket", "baseball bat", "baseball glove", "sports ball", "frisbee",
    "bottle", "cup", "wine glass", "cell phone", "laptop", "book", "remote",
    "knife", "fork", "spoon", "scissors", "umbrella", "skateboard", "surfboard",
    "skis", "snowboard", "kite",
}
CARRY_CLASSES = {"backpack", "handbag", "suitcase"}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train"}
RIDE_CLASSES = {"bicycle", "motorcycle", "horse"}
BOAT_CLASSES = {"boat"}
SIT_CLASSES = {"chair", "couch", "bench", "dining table", "bed", "toilet"}


def _verb_for(cls: str) -> str:
    if cls in CARRY_CLASSES:
        return "carrying"
    if cls in BOAT_CLASSES:
        return "aboard"
    if cls in RIDE_CLASSES:
        return "with"
    if cls in VEHICLE_CLASSES:
        return "next to"
    if cls in SIT_CLASSES:
        return "at"
    if cls in HOLD_CLASSES:
        return "holding"
    return "near"


# --- scene context rules (priority order) ------------------------------------
@dataclass
class SceneRule:
    name: str
    triggers: set
    boosts: Dict[str, float]   # class -> multiplier (relative to context_boost/penalty)


def _scene_rules(boost: float, penalty: float) -> List[SceneRule]:
    return [
        SceneRule("maritime / harbour", {"boat", "surfboard"},
                  {"boat": boost, "surfboard": boost, "truck": penalty,
                   "car": penalty}),
        SceneRule("airfield / airport", {"airplane"},
                  {"airplane": boost, "bird": penalty, "kite": penalty}),
        SceneRule("sports / court", {"tennis racket", "sports ball", "frisbee",
                                     "baseball glove", "skis", "snowboard"},
                  {"tennis racket": boost, "sports ball": boost, "frisbee": boost,
                   "baseball bat": penalty}),
        SceneRule("road / urban", {"car", "truck", "bus", "traffic light",
                                   "stop sign"},
                  {"car": boost, "truck": boost, "bus": boost}),
        SceneRule("indoor / workspace", {"laptop", "tv", "keyboard", "mouse",
                                         "dining table", "couch", "bed", "toilet",
                                         "microwave", "oven", "sink"},
                  {"laptop": boost, "keyboard": boost, "tv": boost,
                   "dining table": boost, "kite": penalty, "teddy bear": penalty}),
    ]


@dataclass
class SceneContext:
    scene_type: str = "general"
    rationale: str = ""
    boosts: Dict[str, float] = field(default_factory=dict)


# --- refined object ----------------------------------------------------------
@dataclass
class ReasonedObject:
    primary_class: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    location: str
    size_label: str
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    confirmed: bool = True
    member_count: int = 1            # raw detections grouped into this object
    interaction: str = ""            # filled when this object interacts with a person

    # Aliases so overlay/report code can treat a reasoned object like a detection.
    @property
    def class_name(self) -> str:
        return self.primary_class

    @property
    def uncertain(self) -> bool:
        return not self.confirmed

    @property
    def alt_text(self) -> str:
        return ", ".join(f"{c} ({p:.2f})" for c, p in self.alternatives)

    def to_dict(self) -> dict:
        return {"class": self.primary_class, "confidence": round(self.confidence, 3),
                "bbox": [round(float(v), 1) for v in self.bbox],
                "location": self.location, "size": self.size_label,
                "alternatives": [{"class": c, "confidence": round(p, 3)}
                                 for c, p in self.alternatives],
                "confirmed": self.confirmed, "grouped_detections": self.member_count,
                "interaction": self.interaction}


@dataclass
class ReasoningResult:
    confirmed: List[ReasonedObject] = field(default_factory=list)
    possible: List[ReasonedObject] = field(default_factory=list)
    interactions: List[str] = field(default_factory=list)
    scene: SceneContext = field(default_factory=SceneContext)
    raw: List[Detection] = field(default_factory=list)   # developer mode

    def all_objects(self) -> List[ReasonedObject]:
        return self.confirmed + self.possible


# --- helpers -----------------------------------------------------------------
def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _size_label(bbox, img_w, img_h) -> str:
    frac = (max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])) / float(img_w * img_h)
    if frac < 0.005:
        return "very small"
    if frac < 0.02:
        return "small"
    if frac < 0.10:
        return "medium"
    return "large"


def _location(bbox, img_w, img_h) -> str:
    from src.intelligence.report import describe_location
    return describe_location(bbox, img_w, img_h)


class ObjectReasoningEngine:
    def __init__(self, config: ObjectReasoningConfig):
        self.cfg = config

    # -- scene -----------------------------------------------------------
    def _infer_scene(self, dets: List[Detection]) -> SceneContext:
        present = {d.class_name for d in dets if d.confidence >= 0.30}
        for rule in _scene_rules(self.cfg.context_boost, self.cfg.context_penalty):
            hit = present & rule.triggers
            if hit:
                return SceneContext(scene_type=rule.name, boosts=rule.boosts,
                                    rationale=f"cued by detected {', '.join(sorted(hit))}")
        return SceneContext(scene_type="general",
                            rationale="no strong scene cue among detections")

    # -- clustering ------------------------------------------------------
    def _cluster(self, dets: List[Detection]) -> List[List[Detection]]:
        """Union-find groups of detections overlapping above ``cluster_iou``."""
        n = len(dets)
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            parent[find(i)] = find(j)

        for i in range(n):
            for j in range(i + 1, n):
                if _iou(dets[i].bbox, dets[j].bbox) >= self.cfg.cluster_iou:
                    union(i, j)
        groups: Dict[int, List[Detection]] = {}
        for i, d in enumerate(dets):
            groups.setdefault(find(i), []).append(d)
        return list(groups.values())

    def _resolve(self, group: List[Detection], scene: SceneContext,
                 img_w: int, img_h: int) -> ReasonedObject:
        """Pick the most-probable class for a group; keep close ones as alternatives."""
        # Best raw confidence per class, and a context-adjusted score for ranking.
        best: Dict[str, float] = {}
        for d in group:
            best[d.class_name] = max(best.get(d.class_name, 0.0), d.confidence)
        adj = {c: p * scene.boosts.get(c, 1.0) for c, p in best.items()}
        ranked = sorted(adj.items(), key=lambda kv: kv[1], reverse=True)
        primary_cls, primary_adj = ranked[0]
        primary_conf = best[primary_cls]
        alts = [(c, best[c]) for c, a in ranked[1:]
                if a >= self.cfg.alt_margin * primary_adj]
        # The merged box = the highest-raw-confidence detection of the primary class.
        prim_det = max((d for d in group if d.class_name == primary_cls),
                       key=lambda d: d.confidence)
        bbox = prim_det.bbox
        return ReasonedObject(
            primary_class=primary_cls, confidence=primary_conf, bbox=bbox,
            location=_location(bbox, img_w, img_h),
            size_label=_size_label(bbox, img_w, img_h),
            alternatives=alts[:3], member_count=len(group),
            confirmed=primary_conf >= self.cfg.confirm_threshold)

    # -- interactions ----------------------------------------------------
    def _interactions(self, objs: List[ReasonedObject],
                      img_w: int, img_h: int) -> List[str]:
        people = [o for o in objs if o.primary_class == "person" and o.confirmed]
        things = [o for o in objs if o.primary_class != "person"]
        phrases: List[str] = []
        for p in people:
            px1, py1, px2, py2 = p.bbox
            pw, ph = px2 - px1, py2 - py1
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            pad = self.cfg.interaction_pad_frac * max(pw, ph)
            for t in things:
                tcx = (t.bbox[0] + t.bbox[2]) / 2
                tcy = (t.bbox[1] + t.bbox[3]) / 2
                near = (_iou(p.bbox, t.bbox) > 0.0
                        or (px1 - pad <= tcx <= px2 + pad
                            and py1 - pad <= tcy <= py2 + pad))
                if not near:
                    continue
                verb = _verb_for(t.primary_class)
                conf_word = "" if t.confirmed else " (low confidence)"
                phrase = (f"One person appears to be {verb} a "
                          f"{t.primary_class}{conf_word}.")
                t.interaction = f"{verb} by a person"
                if phrase not in phrases:
                    phrases.append(phrase)
        return phrases

    # -- entry -----------------------------------------------------------
    def reason(self, detections: List[Detection],
               img_w: int, img_h: int) -> ReasoningResult:
        if not self.cfg.enabled or not detections:
            # Pass-through: each detection becomes its own (un-grouped) object.
            scene = SceneContext()
            objs = [ReasonedObject(
                d.class_name, d.confidence, d.bbox,
                _location(d.bbox, img_w, img_h), _size_label(d.bbox, img_w, img_h),
                confirmed=d.confidence >= self.cfg.confirm_threshold)
                for d in detections]
            return ReasoningResult(
                confirmed=[o for o in objs if o.confirmed],
                possible=[o for o in objs if not o.confirmed],
                scene=scene, raw=list(detections))

        scene = self._infer_scene(detections)
        groups = self._cluster(detections)
        objs = [self._resolve(g, scene, img_w, img_h) for g in groups]
        objs.sort(key=lambda o: o.confidence, reverse=True)
        interactions = self._interactions(objs, img_w, img_h)

        result = ReasoningResult(
            confirmed=[o for o in objs if o.confirmed],
            possible=[o for o in objs if not o.confirmed],
            interactions=interactions, scene=scene, raw=list(detections))
        logger.info("ORE: {} raw -> {} objects ({} confirmed, {} possible), "
                    "scene={}, {} interactions", len(detections), len(objs),
                    len(result.confirmed), len(result.possible),
                    scene.scene_type, len(interactions))
        return result
