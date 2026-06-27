"""VLM boundary interface (repo-split prep, doc 04 Phase 1 / §4.2).

The console cues the language model only through this narrow gateway. Today the
implementation is the in-process :class:`~src.vlm.vlm_worker.VLMWorker`; after the
split the same surface is satisfied by a stream client that ships ``VLMRequest``
messages to the Analyst process. Typing the console's handle as
:class:`VlmGateway` keeps the CV path agnostic to which one is wired in.

The request is intentionally typed ``Any`` so this contract pulls in none of the
heavy VLM dependencies (transformers / bitsandbytes) — ``VLMRequest`` is already a
plain dataclass and becomes the wire message unchanged.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VlmGateway(Protocol):
    """The cueing surface the console depends on (submit work, report health)."""

    busy: bool

    def start(self) -> Any: ...

    def stop(self) -> None: ...

    def submit(self, request: Any) -> bool: ...

    def stats(self) -> dict: ...
