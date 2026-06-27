"""Automatic EO/IR cross-sensor registration — no calibration, no telemetry.

The EO and IR cameras are unaligned, have different fields of view and zoom, and
provide no boresight or pan/tilt/zoom telemetry. We therefore estimate the
mapping between the two image planes **online, from the imagery itself** — but
*not* from appearance (cross-spectral intensity is unreliable; a hot object can
be bright in IR and dark in EO). Instead we register on **object geometry**: the
targets both sensors can see are the control points.

Method
------
* Each sensor contributes a set of 2-D points (detection centroids; for IR also
  thermal-blob centroids). These are modality-invariant — only positions matter.
* A **similarity transform** (scale, rotation, translation; 4-DOF) maps IR→EO.
  Two point correspondences define one hypothesis, so we enumerate 2-IR × 2-EO
  pairings (optionally class-constrained) and score each by how many *other*
  points map onto an EO point within tolerance (RANSAC).
* The best hypothesis is **EMA-smoothed** over time and governed by a lock state
  machine (UNLOCKED → ACQUIRING → LOCKED, with a WARM hold when targets briefly
  vanish). The estimated **scale recovers the relative zoom** with no telemetry.

If both sensors' HFOV and width are known, their ratio seeds a *scale prior* that
rejects geometrically impossible fits and speeds up lock — but the system runs
with none of that.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.config.settings import RegistrationConfig
from src.utils.logger import logger


@dataclass(frozen=True)
class KeyPoint:
    """One registration control point in a sensor's pixel frame."""
    x: float
    y: float
    cls: Optional[str] = None     # class label if from a detection; None for blobs
    kind: str = "det"             # "det" | "blob"
    weight: float = 0.0           # salience (det confidence·area / blob area) — the
                                  # strongest points are kept when capping for RANSAC


@dataclass
class Registration:
    """Current IR→EO mapping estimate and its lock status."""
    state: str = "UNLOCKED"                 # UNLOCKED | ACQUIRING | LOCKED | WARM
    transform: Optional[np.ndarray] = None  # 2x3 affine (similarity) IR -> EO
    scale: float = 0.0
    rotation_deg: float = 0.0
    inliers: int = 0
    confidence: float = 0.0
    eo_size: Tuple[int, int] = (0, 0)       # (w, h)
    ir_size: Tuple[int, int] = (0, 0)

    @property
    def usable(self) -> bool:
        return self.transform is not None and self.state in ("LOCKED", "WARM",
                                                             "ACQUIRING")

    @property
    def locked(self) -> bool:
        return self.state == "LOCKED"

    def map_point(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        if self.transform is None:
            return None
        v = self.transform @ np.array([x, y, 1.0])
        return float(v[0]), float(v[1])

    def map_box(self, box: Sequence[float]
                ) -> Optional[Tuple[float, float, float, float]]:
        """Map an IR xyxy box into EO (axis-aligned bbox of the mapped corners)."""
        if self.transform is None:
            return None
        x1, y1, x2, y2 = box
        corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]])
        m = (self.transform @ corners.T).T
        return (float(m[:, 0].min()), float(m[:, 1].min()),
                float(m[:, 0].max()), float(m[:, 1].max()))

    def in_eo_view(self, x: float, y: float, margin: float = 0.0) -> bool:
        w, h = self.eo_size
        return (-margin <= x <= w + margin) and (-margin <= y <= h + margin)


def _similarity_from_two(p1, p2, q1, q2) -> Optional[Tuple[float, float, float, float]]:
    """Similarity (s, theta, tx, ty) mapping p1->q1, p2->q2. None if degenerate."""
    vp = (p2[0] - p1[0], p2[1] - p1[1])
    vq = (q2[0] - q1[0], q2[1] - q1[1])
    lp = math.hypot(*vp)
    lq = math.hypot(*vq)
    if lp < 1e-6 or lq < 1e-6:
        return None
    s = lq / lp
    theta = math.atan2(vq[1], vq[0]) - math.atan2(vp[1], vp[0])
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    # t = q1 - s R p1
    tx = q1[0] - s * (cos_t * p1[0] - sin_t * p1[1])
    ty = q1[1] - s * (sin_t * p1[0] + cos_t * p1[1])
    return s, theta, tx, ty


def _matrix(s: float, theta: float, tx: float, ty: float) -> np.ndarray:
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return np.array([[s * cos_t, -s * sin_t, tx],
                     [s * sin_t, s * cos_t, ty]], dtype=float)


class CrossSensorRegistrar:
    def __init__(self, config: RegistrationConfig,
                 scale_prior: Optional[float] = None):
        self.config = config
        self.scale_prior = scale_prior if (scale_prior and scale_prior > 0) else None
        # Smoothed running estimate.
        self._s = 0.0
        self._theta = 0.0
        self._tx = 0.0
        self._ty = 0.0
        self._has_est = False
        self._good_streak = 0
        self._bad_streak = 0
        self._last_good_ts = -1e9
        self._state = "UNLOCKED"
        if self.scale_prior:
            logger.info("Cross-sensor registrar: scale prior ~{:.2f} (FOV ratio)",
                        self.scale_prior)

    def reset(self) -> None:
        self._has_est = False
        self._good_streak = self._bad_streak = 0
        self._last_good_ts = -1e9
        self._state = "UNLOCKED"

    # ------------------------------------------------------------------ API
    def update(self, eo_points: List[KeyPoint], ir_points: List[KeyPoint],
               eo_size: Tuple[int, int], ir_size: Tuple[int, int],
               now: float) -> Registration:
        cfg = self.config
        tol = max(4.0, cfg.inlier_tol_frac * max(1, eo_size[0]))
        fit = None
        if (cfg.enabled and len(eo_points) >= cfg.min_points
                and len(ir_points) >= cfg.min_points):
            fit = self._best_fit(eo_points, ir_points, tol)

        if fit is not None and fit[0] >= cfg.min_points:
            inliers, params = fit
            self._absorb(params)
            self._good_streak += 1
            self._bad_streak = 0
            self._last_good_ts = now
            if (self._good_streak >= cfg.lock_persist_cycles
                    and inliers >= cfg.min_lock_inliers):
                self._state = "LOCKED"
            else:
                self._state = "ACQUIRING"
            conf = inliers / max(2, min(len(eo_points), len(ir_points)))
            return self._snapshot(self._state, inliers, min(1.0, conf),
                                  eo_size, ir_size)

        # No fresh fit this cycle.
        self._bad_streak += 1
        self._good_streak = 0
        warm = (self._has_est and now - self._last_good_ts <= cfg.max_warm_sec
                and self._bad_streak <= cfg.unlock_grace_cycles)
        if warm:
            self._state = "WARM"
            return self._snapshot("WARM", 0, 0.0, eo_size, ir_size)
        self._state = "UNLOCKED"
        self._has_est = self._has_est and False  # keep params but mark unusable
        return self._snapshot("UNLOCKED", 0, 0.0, eo_size, ir_size)

    # --------------------------------------------------------------- internals
    def _best_fit(self, eo: List[KeyPoint], ir: List[KeyPoint], tol: float):
        cfg = self.config
        # Cap to the strongest control points per sensor so the 2x2 hypothesis
        # enumeration stays bounded (it is O(C(n,2)^2)); the strongest points are
        # also the most reliably localised. Inliers are still counted against the
        # capped set — consistent with what we hypothesise from.
        eo = _top_points(eo, cfg.max_points)
        ir = _top_points(ir, cfg.max_points)
        eo_xy = [(p.x, p.y) for p in eo]
        ir_xy = [(p.x, p.y) for p in ir]
        # Enough matched targets to stop early (can't exceed the smaller point set).
        target = min(max(2, cfg.early_exit_inliers), len(eo), len(ir))
        best_inliers, best_params = 0, None
        iters = 0
        lo, hi = self._scale_bounds()
        for i1, i2 in combinations(range(len(ir)), 2):
            for j1, j2 in combinations(range(len(eo)), 2):
                # Two correspondence orderings for this 2x2 pairing.
                for (a, b) in (((j1, j2)), ((j2, j1))):
                    if cfg.class_constrained and not (
                            _cls_ok(ir[i1], eo[a]) and _cls_ok(ir[i2], eo[b])):
                        continue
                    iters += 1
                    if iters > cfg.ransac_iters:
                        return (best_inliers, best_params) if best_params else None
                    est = _similarity_from_two(ir_xy[i1], ir_xy[i2],
                                               eo_xy[a], eo_xy[b])
                    if est is None:
                        continue
                    s, theta, tx, ty = est
                    if not (lo <= s <= hi) or not math.isfinite(s):
                        continue
                    inl = self._count_inliers(_matrix(s, theta, tx, ty),
                                              ir_xy, eo_xy, tol)
                    if inl > best_inliers:
                        best_inliers, best_params = inl, est
                        if best_inliers >= target:   # good enough — stop searching
                            return best_inliers, best_params
        return (best_inliers, best_params) if best_params else None

    @staticmethod
    def _count_inliers(M: np.ndarray, ir_xy, eo_xy, tol: float) -> int:
        """Greedy one-to-one inlier count: each mapped IR point claims its
        nearest unused EO point within ``tol``."""
        pts = np.array([[x, y, 1.0] for (x, y) in ir_xy])
        mapped = (M @ pts.T).T  # Nx2
        eo = np.array(eo_xy)
        used = set()
        inl = 0
        for mp in mapped:
            d = np.hypot(eo[:, 0] - mp[0], eo[:, 1] - mp[1])
            order = np.argsort(d)
            for k in order:
                if d[k] > tol:
                    break
                if k not in used:
                    used.add(int(k))
                    inl += 1
                    break
        return inl

    def _scale_bounds(self) -> Tuple[float, float]:
        if self.scale_prior:
            t = max(1.2, self.config.scale_prior_tol)
            return self.scale_prior / t, self.scale_prior * t
        return 0.05, 20.0

    def _absorb(self, params) -> None:
        s, theta, tx, ty = params
        if not self._has_est:
            self._s, self._theta, self._tx, self._ty = s, theta, tx, ty
            self._has_est = True
            return
        a = self.config.smoothing  # weight on the OLD estimate
        self._s = a * self._s + (1 - a) * s
        self._tx = a * self._tx + (1 - a) * tx
        self._ty = a * self._ty + (1 - a) * ty
        # Angle EMA via unit vectors to avoid wrap-around.
        ox, oy = math.cos(self._theta), math.sin(self._theta)
        nx, ny = math.cos(theta), math.sin(theta)
        self._theta = math.atan2(a * oy + (1 - a) * ny, a * ox + (1 - a) * nx)

    def _snapshot(self, state, inliers, conf, eo_size, ir_size) -> Registration:
        usable = state in ("LOCKED", "WARM", "ACQUIRING") and self._has_est
        M = _matrix(self._s, self._theta, self._tx, self._ty) if usable else None
        return Registration(
            state=state, transform=M, scale=self._s if usable else 0.0,
            rotation_deg=math.degrees(self._theta) if usable else 0.0,
            inliers=inliers, confidence=conf, eo_size=eo_size, ir_size=ir_size)


def _top_points(pts: List[KeyPoint], cap: int) -> List[KeyPoint]:
    """Keep the ``cap`` highest-weight control points (all of them if cap<=0 or
    already within budget). Stable for equal weights / zero-weight inputs."""
    if cap <= 0 or len(pts) <= cap:
        return pts
    return sorted(pts, key=lambda p: p.weight, reverse=True)[:cap]


def _cls_ok(a: KeyPoint, b: KeyPoint) -> bool:
    """Two points may correspond if classes match, or either is class-agnostic
    (a thermal blob has no class)."""
    if a.cls is None or b.cls is None:
        return True
    return a.cls == b.cls


def fov_scale_prior(eo_sensor, ir_sensor) -> Optional[float]:
    """Scale prior for T(IR→EO) from FOV/width: (EO px/deg)/(IR px/deg).

    Returns None unless both sensors' HFOV and delivered width are known. Note
    this is the *wide-EO* prior; the registrar widens its bounds around it, and
    the live estimate tracks the true (zoomed) scale regardless."""
    try:
        if (eo_sensor.hfov_deg > 0 and ir_sensor.hfov_deg > 0
                and eo_sensor.width > 0 and ir_sensor.width > 0):
            eo_ppd = eo_sensor.width / eo_sensor.hfov_deg
            ir_ppd = ir_sensor.width / ir_sensor.hfov_deg
            if ir_ppd > 0:
                return eo_ppd / ir_ppd
    except AttributeError:
        pass
    return None
