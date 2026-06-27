"""Mission Query Assistant — natural-language Q&A over mission memory (phase 5).

Pipeline for one question:

  1. **Classify** the question into one whitelisted :class:`~src.query.intents.Intent`.
     The VLM proposes an intent + slots as JSON; deterministic regex parsers
     (time windows, ids, classes) override the model for anything numeric, and a
     keyword classifier is the fallback when the VLM is unavailable or replies
     with junk. The model never emits SQL.
  2. **Retrieve** evidence by running that intent's single hard-coded,
     parameterised, read-only query against :class:`MissionStore`.
  3. **Phrase** a grounded answer. The headline fact is already computed in code
     (see :class:`Evidence`); the VLM only rewords it over the cited evidence,
     so the numbers can never drift. Without a VLM we return the deterministic
     answer plus citations directly.

The VLM is reused (Qwen2.5-VL, text-only) — no second model is loaded, honouring
the 8 GB VRAM budget.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.config.settings import QueryConfig
from src.query.intents import (INTENTS, Evidence, classify_keywords,
                               parse_class, parse_global_id, parse_time_window)
from src.query.prompts import (ANSWER_SYSTEM, answer_prompt,
                               intent_classification_prompt)
from src.utils.logger import logger

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class QueryResult:
    question: str
    intent: str
    answer: str
    evidence: Evidence
    grounded: bool = False
    slots: Dict[str, Any] = field(default_factory=dict)


class QueryAssistant:
    """Answers analyst questions against a :class:`MissionStore`.

    ``vlm`` is any object exposing ``infer(images, prompt, system=None) -> str``
    (i.e. :class:`~src.vlm.qwen_vlm.QwenVLM`); pass ``None`` for a fully
    deterministic, GPU-free assistant.
    """

    def __init__(self, store, vlm=None, config: Optional[QueryConfig] = None):
        self.store = store
        self.vlm = vlm
        self.config = config or QueryConfig()

    # ------------------------------------------------------------------- API
    def ask(self, question: str, mission_id: Optional[int] = None) -> QueryResult:
        question = (question or "").strip()
        mid = mission_id if mission_id is not None else self.store.mission_id
        if not question:
            return self._empty(question, "Please enter a question.")
        if mid is None:
            return self._empty(question, "No mission is loaded to query.")

        intent_name, slots = self._classify(question)
        intent = INTENTS.get(intent_name) or INTENTS[self.config.fallback_intent]
        try:
            evidence = intent.handler(self.store, mid, slots)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Query intent '{}' failed: {}", intent.name, exc)
            evidence = Evidence("The query could not be completed against mission "
                                "memory.")
        grounded = bool(evidence.rows) or bool(evidence.citations)
        answer = self._phrase(question, evidence)
        logger.info("Query '{}' -> intent={} grounded={}",
                    question, intent.name, grounded)
        return QueryResult(question, intent.name, answer, evidence, grounded, slots)

    # ------------------------------------------------------------- classify
    def _classify(self, question: str) -> Tuple[str, Dict[str, Any]]:
        # Deterministic slots first — regex beats the model for numbers/ids.
        slots: Dict[str, Any] = {
            "window_sec": parse_time_window(question),
            "global_id": parse_global_id(question),
            "class_name": parse_class(question),
        }
        intent_name: Optional[str] = None
        if self.vlm is not None and self.config.use_vlm_classifier:
            intent_name, vlm_slots = self._vlm_classify(question)
            # Fill only the slots the parsers could not determine.
            if slots["class_name"] is None and vlm_slots.get("class_name"):
                slots["class_name"] = str(vlm_slots["class_name"]).lower()
            if slots["window_sec"] is None and isinstance(
                    vlm_slots.get("window_sec"), (int, float)):
                slots["window_sec"] = float(vlm_slots["window_sec"])
            if slots["global_id"] is None and isinstance(
                    vlm_slots.get("global_id"), int):
                slots["global_id"] = vlm_slots["global_id"]
        if intent_name not in INTENTS:
            intent_name = classify_keywords(question)
        return intent_name, slots

    def _vlm_classify(self, question: str) -> Tuple[Optional[str], Dict[str, Any]]:
        prompt = intent_classification_prompt(question, list(INTENTS.values()))
        try:
            raw = self.vlm.infer([], prompt)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VLM classify failed, using keywords: {}", exc)
            return None, {}
        return _parse_intent_json(raw)

    # --------------------------------------------------------------- phrase
    def _phrase(self, question: str, evidence: Evidence) -> str:
        if self.vlm is not None and self.config.phrase_with_vlm:
            try:
                txt = self.vlm.infer([], answer_prompt(question, evidence),
                                     ANSWER_SYSTEM)
                if txt and not txt.lstrip().startswith("[VLM"):
                    return txt.strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("VLM phrasing failed, using deterministic: {}", exc)
        return self._deterministic(evidence)

    @staticmethod
    def _deterministic(evidence: Evidence) -> str:
        if evidence.citations:
            bullets = "\n".join(f"  • {c}" for c in evidence.citations[:10])
            return f"{evidence.answer}\n{bullets}"
        return evidence.answer

    # ---------------------------------------------------------------- utils
    def _empty(self, question: str, message: str) -> QueryResult:
        return QueryResult(question, "none", message, Evidence(message), False, {})


def _parse_intent_json(raw: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Best-effort extraction of the classifier's JSON object from VLM text."""
    if not raw:
        return None, {}
    m = _JSON_RE.search(raw)
    if not m:
        return None, {}
    try:
        obj = json.loads(m.group(0))
    except (json.JSONDecodeError, ValueError):
        return None, {}
    if not isinstance(obj, dict):
        return None, {}
    intent = obj.get("intent")
    intent = intent if isinstance(intent, str) and intent in INTENTS else None
    slots = {k: obj.get(k) for k in ("class_name", "window_sec", "global_id")}
    return intent, slots
