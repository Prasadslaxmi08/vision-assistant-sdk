"""Prompt templates for the Mission Query Assistant (V2 phase 5).

Two prompts only, by design:
  * :func:`intent_classification_prompt` — asks the VLM to route a question to
    exactly one whitelisted intent and extract its slots, as strict JSON. The
    model never writes SQL; it only picks a category.
  * :func:`answer_prompt` + :data:`ANSWER_SYSTEM` — asks the VLM to rephrase a
    *already-computed* factual answer over the retrieved evidence, so wording is
    natural while the numbers stay grounded in mission memory.
"""
from __future__ import annotations

from typing import List

from src.query.intents import Evidence, Intent


def intent_classification_prompt(question: str, intents: List[Intent]) -> str:
    lines = []
    for it in intents:
        ex = (" e.g. " + " / ".join(f'"{e}"' for e in it.examples[:2])
              if it.examples else "")
        lines.append(f"- {it.name}: {it.description}.{ex}")
    catalog = "\n".join(lines)
    return (
        "You route an analyst's natural-language question about a COMPLETED "
        "EO/IR surveillance mission to exactly ONE query category. Do not answer "
        "the question or write any SQL — only classify it.\n\n"
        f"Categories:\n{catalog}\n\n"
        f'Question: "{question}"\n\n'
        "Respond with ONLY a compact JSON object and nothing else:\n"
        '{"intent": "<one category name>", "class_name": <string or null>, '
        '"window_sec": <number of seconds or null>, '
        '"global_id": <integer or null>}\n'
        "Set a slot to null when the question does not specify it. For "
        "'class_name' use a single object class such as \"person\", \"car\", or "
        "the literal \"vehicle\" for any vehicle."
    )


ANSWER_SYSTEM = (
    "You are an EO/IR mission intelligence analyst answering an operator's "
    "question about a completed surveillance mission. You are given a "
    "pre-computed factual ANSWER and the supporting EVIDENCE rows retrieved from "
    "the mission database. Restate the answer in one or two clear, "
    "analyst-style sentences. You MUST stay faithful to the provided numbers and "
    "evidence — never invent objects, counts, times, or events that are not "
    "present. If the evidence is empty, say the mission memory has no record of "
    "it. Do not add caveats about being an AI."
)


def answer_prompt(question: str, evidence: Evidence) -> str:
    cites = "\n".join(f"  - {c}" for c in evidence.citations) or "  (none)"
    return (
        f'Operator question: "{question}"\n\n'
        f"Pre-computed factual answer: {evidence.answer}\n"
        f"Supporting evidence:\n{cites}\n\n"
        "Write the final grounded answer for the operator."
    )
