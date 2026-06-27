"""Canonical COCO-80 class names (0-indexed), shared by the detectors and the
class selector. RF-DETR and YOLOv11 both emit this id space, so a single list is
the source of truth for "what can be detected" and for name<->id mapping."""
from __future__ import annotations

from typing import List, Optional, Sequence, Set

COCO_CLASSES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# A useful default for ISR / surveillance: people, vehicles, vessels, aircraft.
ISR_PRESET: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat",
]

_NAME_TO_ID = {n: i for i, n in enumerate(COCO_CLASSES)}


def names_to_ids(names: Sequence[str]) -> List[int]:
    """COCO ids for the given class names (unknown names ignored)."""
    return sorted(_NAME_TO_ID[n] for n in names if n in _NAME_TO_ID)


def normalize(names: Optional[Sequence[str]]) -> Optional[Set[str]]:
    """A clean set of valid class names, or ``None`` when nothing/all is selected
    (None means 'do not filter' — report every class)."""
    if not names:
        return None
    keep = {n for n in names if n in _NAME_TO_ID}
    return keep or None
