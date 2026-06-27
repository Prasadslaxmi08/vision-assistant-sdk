"""Scene-reasoning layer of the AI Core.

Higher-order inference built on top of detection/tracking:
  * spatial — object-to-object relationships (proximity, following, grouping,
    restricted-area incursion) — scene understanding.
  * registration — automatic EO/IR cross-sensor alignment from shared targets.
  * fusion  — cross-sensor thermal↔visual correlation over a registered pair.

Threat / anomaly scoring moved to the future AI Mission Analyst repo.
"""
from src.reasoning.fusion import (FusionAssessment, FusionEngine, FusionType,
                                   blob_keypoints, detection_keypoints)
from src.reasoning.object_reasoning import (ObjectReasoningEngine, ReasonedObject,
                                            ReasoningResult, SceneContext)
from src.reasoning.registration import (CrossSensorRegistrar, KeyPoint,
                                        Registration, fov_scale_prior)
from src.reasoning.spatial import (Interaction, InteractionType,
                                    SpatialReasoner)

__all__ = ["SpatialReasoner", "Interaction", "InteractionType",
           "FusionEngine", "FusionAssessment", "FusionType",
           "detection_keypoints", "blob_keypoints",
           "CrossSensorRegistrar", "KeyPoint", "Registration",
           "fov_scale_prior",
           "ObjectReasoningEngine", "ReasoningResult", "ReasonedObject",
           "SceneContext"]
