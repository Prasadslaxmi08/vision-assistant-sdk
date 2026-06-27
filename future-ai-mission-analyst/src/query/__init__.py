"""Mission Query Assistant (V2 phase 5).

Natural-language questions about a completed mission, answered over the
persistent :class:`~src.memory.mission_store.MissionStore`. Questions are routed
to a fixed catalogue of whitelisted, parameterised, read-only SQL intents (never
free-form SQL), the matching evidence is retrieved, and a grounded, cited answer
is produced — optionally reworded by the reused Qwen2.5-VL model (no second
model, honouring the 8 GB VRAM budget).
"""
from src.query.assistant import QueryAssistant, QueryResult
from src.query.intents import INTENTS, Evidence, Intent

__all__ = ["QueryAssistant", "QueryResult", "Evidence", "Intent", "INTENTS"]
