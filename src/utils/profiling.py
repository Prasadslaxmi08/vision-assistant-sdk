"""Lightweight, thread-safe runtime profiling for the live console.

Milestone 0 of the latency-optimization roadmap
(`docs/architecture-review/03-latency-optimization-roadmap.md`): replace the
static-analysis bottleneck *estimates* in the performance report with **measured**
per-stage timings on the target box, so every later optimization lands with a
before/after.

Design goals:
  * **Near-zero overhead when disabled** — the hot path checks a single bool and
    returns a shared no-op context manager; no allocation, no lock, no timing.
  * **No per-frame logging spam** — samples land in bounded ring buffers; the
    dashboard reads aggregates (EMA + p50/p95) at ~1 Hz.
  * **Thread-safe** — the inference thread records while the UI thread reads.

Nothing here touches a model or the GPU; it is pure instrumentation.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Dict


class _NullTimer:
    """Shared no-op context manager returned when profiling is disabled."""
    __slots__ = ()

    def __enter__(self) -> "_NullTimer":
        return self

    def __exit__(self, *exc) -> bool:
        return False


_NULL = _NullTimer()


class _Timer:
    """Context manager that records elapsed milliseconds on exit."""
    __slots__ = ("_prof", "_stage", "_t0")

    def __init__(self, prof: "StageProfiler", stage: str):
        self._prof = prof
        self._stage = stage
        self._t0 = 0.0

    def __enter__(self) -> "_Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> bool:
        self._prof.record(self._stage, (time.perf_counter() - self._t0) * 1000.0)
        return False


class _Rate:
    """EMA frames-per-second estimator driven by :meth:`mark` timestamps."""
    __slots__ = ("_last", "_ema", "count")

    def __init__(self):
        self._last = 0.0
        self._ema = 0.0
        self.count = 0

    def mark(self, now: float, alpha: float = 0.2) -> None:
        self.count += 1
        if self._last > 0.0:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self._ema = inst if self._ema == 0.0 else alpha * inst + (1 - alpha) * self._ema
        self._last = now

    @property
    def fps(self) -> float:
        return self._ema


def _percentile(sorted_vals, p: float) -> float:
    """Linear-interpolated percentile over an already-sorted list."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    val = sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)
    return round(val, 3)


class StageProfiler:
    """Collects per-stage latencies, rates and gauges for the console.

    Usage on the hot path::

        with profiler.time("detect"):
            ...detect...
        profiler.mark("infer")            # one frame went through inference
        profiler.gauge("eo_frame_age_ms", age)

    When ``enabled`` is False every method is a cheap early return and
    :meth:`time` hands back a shared no-op context manager.
    """

    def __init__(self, enabled: bool = False, window: int = 240):
        self.enabled = enabled
        self._window = max(8, int(window))
        self._stages: Dict[str, Deque[float]] = {}
        self._ema: Dict[str, float] = {}
        self._rates: Dict[str, _Rate] = {}
        self._gauges: Dict[str, float] = {}
        self._lock = threading.Lock()

    # ----------------------------------------------------------- collection
    def time(self, stage: str):
        """Return a context manager timing ``stage`` (no-op when disabled)."""
        if not self.enabled:
            return _NULL
        return _Timer(self, stage)

    def record(self, stage: str, ms: float) -> None:
        """Record a single ``stage`` latency sample in milliseconds."""
        if not self.enabled:
            return
        with self._lock:
            dq = self._stages.get(stage)
            if dq is None:
                dq = self._stages[stage] = deque(maxlen=self._window)
            dq.append(ms)
            prev = self._ema.get(stage, 0.0)
            self._ema[stage] = ms if prev == 0.0 else 0.2 * ms + 0.8 * prev

    def mark(self, rate: str) -> None:
        """Register one tick of a named rate (e.g. ``infer``, ``display``)."""
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            r = self._rates.get(rate)
            if r is None:
                r = self._rates[rate] = _Rate()
            r.mark(now)

    def gauge(self, name: str, value: float) -> None:
        """Set the latest value of a named gauge (queue depth, frame age, …)."""
        if not self.enabled:
            return
        with self._lock:
            self._gauges[name] = float(value)

    def reset(self) -> None:
        with self._lock:
            self._stages.clear()
            self._ema.clear()
            self._rates.clear()
            self._gauges.clear()

    # -------------------------------------------------------------- readout
    def metrics(self) -> dict:
        """Aggregate snapshot for the dashboard. Safe to call from any thread."""
        with self._lock:
            stages = {}
            for name, dq in self._stages.items():
                if not dq:
                    continue
                vals = sorted(dq)
                stages[name] = {
                    "p50": _percentile(vals, 50),
                    "p95": _percentile(vals, 95),
                    "ema": round(self._ema.get(name, 0.0), 3),
                    "max": round(vals[-1], 3),
                    "n": len(vals),
                }
            rates = {k: round(r.fps, 2) for k, r in self._rates.items()}
            counts = {k: r.count for k, r in self._rates.items()}
            gauges = dict(self._gauges)
        return {"enabled": self.enabled, "stages": stages, "rates": rates,
                "counts": counts, "gauges": gauges}
