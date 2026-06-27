"""Live GPU / system telemetry.

Provides a single cheap call, :func:`gpu_stats`, that the UI can poll every
refresh to show *real* GPU load. This exists because Windows Task Manager hides
CUDA-compute work by default (it graphs the ``3D``/``Copy`` engines; CUDA shows
under the ``Compute_0`` dropdown), which makes a fully-loaded GPU read "0%" and
gives the false impression the GPU is idle.

Memory is read via ``torch.cuda.mem_get_info`` (no extra dependency).
Utilisation is read via NVML (``nvidia-ml-py``) when available — a cheap C call,
safe to poll at UI cadence. Everything degrades gracefully to ``None`` so a
missing GPU or missing NVML never breaks the app.
"""
from __future__ import annotations

from typing import Optional, TypedDict

_NVML_READY: Optional[bool] = None   # tri-state: None = not yet tried


class GpuStats(TypedDict):
    available: bool
    name: str
    mem_used_mb: Optional[float]
    mem_total_mb: Optional[float]
    mem_used_pct: Optional[float]
    util_pct: Optional[int]


def _ensure_nvml() -> bool:
    """Initialise NVML once; cache the outcome. Returns True if usable."""
    global _NVML_READY
    if _NVML_READY is not None:
        return _NVML_READY
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        _NVML_READY = True
    except Exception:  # noqa: BLE001 — NVML optional; tolerate any failure
        _NVML_READY = False
    return _NVML_READY


def _nvml_util(index: int = 0) -> Optional[int]:
    if not _ensure_nvml():
        return None
    try:
        import pynvml  # type: ignore

        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        return int(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except Exception:  # noqa: BLE001
        return None


def gpu_stats(index: int = 0) -> GpuStats:
    """Return current GPU memory + utilisation for device ``index``.

    Never raises. On a CPU-only box (or if torch/CUDA is unavailable) returns
    ``available=False`` with empty fields.
    """
    empty: GpuStats = {
        "available": False, "name": "CPU / no CUDA",
        "mem_used_mb": None, "mem_total_mb": None,
        "mem_used_pct": None, "util_pct": None,
    }
    try:
        import torch

        if not torch.cuda.is_available():
            return empty
        name = torch.cuda.get_device_name(index)
        free_b, total_b = torch.cuda.mem_get_info(index)
        used_b = total_b - free_b
        used_mb = used_b / (1024 ** 2)
        total_mb = total_b / (1024 ** 2)
        return {
            "available": True,
            "name": name,
            "mem_used_mb": round(used_mb, 1),
            "mem_total_mb": round(total_mb, 1),
            "mem_used_pct": round(100.0 * used_b / total_b, 1) if total_b else None,
            "util_pct": _nvml_util(index),
        }
    except Exception:  # noqa: BLE001
        return empty
