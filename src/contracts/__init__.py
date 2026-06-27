"""Shared contracts between the EO/IR Mission Console (CV) and AI Mission Analyst.

This package is the single import surface for everything that crosses the
CV→AI boundary (repo-split prep, doc 04). It re-exports:

* **Data records** — the serializable types streamed across the boundary
  (``Detection``, ``FrameResult``, ``Modality``, ``Event``, ``Interaction``,
  ``FusionAssessment`` + enums). They physically live in :mod:`src.utils.types`
  (dependency-free); this package just names them as "the contract".
* **Boundary interfaces** — :class:`VlmGateway` (the VLM seam).
* **Shared infra** — the ``logger``.

These records are exactly what the Output Layer serialises to JSON / hands to
callbacks, and what the future AI Mission Analyst repo consumes. The SQLite
``StoreWriter``/``StoreReader`` seam moved out with the persistence layer.
"""
from __future__ import annotations

from src.contracts.vlm import VlmGateway
from src.utils.logger import logger
from src.utils.types import (Detection, EVENT_PRIORITY, Event, EventType, Frame,
                             FrameResult, FusionAssessment, FusionType,
                             Interaction, InteractionType, Modality)

__all__ = [
    # data records
    "Detection", "Frame", "FrameResult", "Modality",
    "Event", "EventType", "EVENT_PRIORITY",
    "Interaction", "InteractionType",
    "FusionAssessment", "FusionType",
    # boundary interfaces
    "VlmGateway",
    # shared infra
    "logger",
]
