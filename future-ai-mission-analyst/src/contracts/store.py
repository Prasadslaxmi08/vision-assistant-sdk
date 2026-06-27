"""Persistence boundary interfaces (repo-split prep, doc 04 Phase 1).

The live console + offline analyzers (CV / Repo A) only ever *write* mission
records; the query + reporting stack (AI / Repo B) only ever *reads* them. These
two narrow Protocols name that split explicitly. ``MissionStore`` satisfies both
structurally today, so nothing changes at runtime — but typing the console's
handle as :class:`StoreWriter` documents (and lets a checker enforce) that CV
code never reaches for a reader method, which is what makes the eventual physical
split mechanical: Repo A keeps a writer shim, Repo B owns the full store.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from src.utils.types import Detection, Event, Interaction


@runtime_checkable
class StoreWriter(Protocol):
    """The write-only surface the real-time CV path depends on."""

    @property
    def mission_id(self) -> Optional[int]: ...

    def open_mission(self, name: str, source: str, modality: str,
                     **kwargs: Any) -> Optional[int]: ...

    def close_mission(self) -> None: ...

    def record_frame(self, frame_id: int, source_ts: float,
                     detections: Sequence[Detection]) -> None: ...

    def record_events(self, events: Sequence[Event]) -> None: ...

    def record_interaction(self, frame_id: int, source_ts: float, itype: str,
                           *args: Any, **kwargs: Any) -> None: ...

    def record_thermal(self, frame_id: int, source_ts: float, area_px: int,
                       *args: Any, **kwargs: Any) -> None: ...

    def record_vlm(self, frame_id: int, source_ts: float, timestamp: float,
                   *args: Any, **kwargs: Any) -> None: ...

    def update_threat(self, updates: Dict[int, tuple]) -> None: ...

    def update_object_media(self, snapshots: Dict[int, Any]) -> None: ...

    def queue_stats(self) -> dict: ...


@runtime_checkable
class StoreReader(Protocol):
    """The read-only surface the query / reporting stack depends on."""

    def read(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]: ...

    def list_missions(self) -> List[Dict[str, Any]]: ...

    def list_objects(self, mission_id: Optional[int] = None
                     ) -> List[Dict[str, Any]]: ...

    def object_track(self, global_id: int, mission_id: Optional[int] = None
                     ) -> List[Dict[str, Any]]: ...

    def interactions(self, mission_id: Optional[int] = None, limit: int = 1000
                     ) -> List[Dict[str, Any]]: ...

    def thermal_events(self, mission_id: Optional[int] = None, limit: int = 1000
                       ) -> List[Dict[str, Any]]: ...
