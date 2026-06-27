"""Persistent mission memory (V2).

A self-contained, asynchronous SQLite store that continuously records every
detection, track lifecycle, event, VLM observation, interaction, and thermal
correlation for a mission — the temporal-memory substrate the spatial-reasoning,
fusion, threat-scoring, and natural-language-query layers build on.
"""
from src.memory.mission_store import MissionStore

__all__ = ["MissionStore"]
