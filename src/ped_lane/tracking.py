"""Temporal smoothing for *display*. Changes nothing about what is detected.

The classical baseline fires and drops frame to frame, and its convex hulls
jitter, which makes an overlay look broken even when the underlying decision is
unchanged. This module takes a per-frame (score, polygon) stream and produces a
stable region to draw:

  * **hysteresis** -- a region appears only after the score holds above
    `enter_score` for `enter_frames`, and disappears only after it holds below
    `exit_score` for `exit_frames`. Two thresholds, not one, so a score sitting
    near the boundary cannot flicker.
  * **hold through dropouts** -- a few frames with no detection do not remove
    the region; the last shape is held.
  * **corner EMA** -- the quadrilateral's corners are averaged over time, in a
    canonical order so they cannot swap and turn the shape inside out.
  * **fade** -- opacity ramps instead of popping.

This is a presentation filter. It can only change *when* and *how smoothly* a
region is drawn, never whether the detector fired. Honest reporting therefore
needs both counts, which is why `RegionTracker` keeps `raw_fired_frames`
alongside `shown_frames`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class TrackerParams:
    #: Score needed to start showing a region.
    enter_score: float = 0.40
    #: Score below which the region starts timing out. Lower than
    #: `enter_score` on purpose: that gap is the hysteresis.
    exit_score: float = 0.25
    #: Consecutive frames above `enter_score` before the region appears.
    enter_frames: int = 3
    #: Consecutive frames below `exit_score` before it disappears.
    exit_frames: int = 10
    #: Frames of "no detection at all" the region survives, keeping its shape.
    hold_frames: int = 8
    #: Corner smoothing when the new detection overlaps the current region.
    ema_alpha: float = 0.35
    #: Slower smoothing when it does not: drift toward the new shape rather
    #: than snap to it, which is what makes a jump look like a glitch.
    ema_alpha_mismatch: float = 0.12
    #: IoU above which a new detection counts as the same region.
    min_iou: float = 0.30
    #: Frames to fade fully in or out.
    fade_frames: int = 6


@dataclass(frozen=True)
class TrackState:
    """What to draw this frame."""

    visible: bool
    opacity: float
    quad: np.ndarray | None


def order_quad(points: np.ndarray) -> np.ndarray:
    """Four corners in a canonical order, starting top-left, going clockwise.

    Without a canonical order the EMA averages corner 0 of one frame against a
    different physical corner of the next, and the shape folds in on itself.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) != 4:
        raise ValueError(f"expected 4 points, got {len(pts)}")
    centre = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
    clockwise = pts[np.argsort(angles)]
    # rotate so the corner nearest the top-left of the bounding box comes first
    corner = np.array([clockwise[:, 0].min(), clockwise[:, 1].min()], dtype=np.float32)
    start = int(np.argmin(np.linalg.norm(clockwise - corner, axis=1)))
    return np.roll(clockwise, -start, axis=0)


def fit_quad(points: np.ndarray) -> np.ndarray:
    """Smallest sensible quadrilateral around a point set.

    Tries a 4-vertex simplification of the convex hull first, because a
    crossing in perspective is a trapezoid rather than a rotated rectangle,
    and falls back to the min-area rectangle.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if len(pts) < 3:
        raise ValueError("need at least 3 points")
    hull = cv2.convexHull(pts)
    perimeter = cv2.arcLength(hull, True)
    for fraction in (0.02, 0.03, 0.05, 0.08, 0.12):
        approx = cv2.approxPolyDP(hull, fraction * perimeter, True)
        if len(approx) == 4:
            return order_quad(approx.reshape(-1, 2))
    return order_quad(cv2.boxPoints(cv2.minAreaRect(pts)))


def quad_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two convex quadrilaterals."""
    pa = np.asarray(a, dtype=np.float32).reshape(-1, 2)
    pb = np.asarray(b, dtype=np.float32).reshape(-1, 2)
    inter, _ = cv2.intersectConvexConvex(pa, pb)
    if inter <= 0:
        return 0.0
    area_a = cv2.contourArea(pa)
    area_b = cv2.contourArea(pb)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class RegionTracker:
    """One smoothed region. Feed it one frame at a time."""

    params: TrackerParams = field(default_factory=TrackerParams)

    _present: bool = False
    _above: int = 0
    _below: int = 0
    _dropouts: int = 0
    _opacity: float = 0.0
    _quad: np.ndarray | None = None

    #: Frames on which the detector itself fired (score >= enter_score).
    raw_fired_frames: int = 0
    #: Frames on which the smoothed region was actually drawn.
    shown_frames: int = 0

    def update(self, score: float, polygon: np.ndarray | None) -> TrackState:
        p = self.params

        if score >= p.enter_score:
            self.raw_fired_frames += 1
            self._above += 1
        else:
            self._above = 0
        self._below = self._below + 1 if score < p.exit_score else 0

        if not self._present and self._above >= p.enter_frames:
            self._present = True
            self._below = 0
        elif self._present and self._below >= p.exit_frames:
            self._present = False

        if polygon is not None and len(polygon) >= 3:
            measured = fit_quad(polygon)
            if self._quad is None:
                self._quad = measured
            else:
                overlapping = quad_iou(self._quad, measured) >= p.min_iou
                alpha = p.ema_alpha if overlapping else p.ema_alpha_mismatch
                self._quad = (1.0 - alpha) * self._quad + alpha * measured
            self._dropouts = 0
        else:
            self._dropouts += 1
            if self._dropouts > p.hold_frames and not self._present:
                self._quad = None

        step = 1.0 / max(p.fade_frames, 1)
        target = 1.0 if (self._present and self._quad is not None) else 0.0
        if self._opacity < target:
            self._opacity = min(target, self._opacity + step)
        elif self._opacity > target:
            self._opacity = max(target, self._opacity - step)

        visible = self._opacity > 0.01 and self._quad is not None
        if visible:
            self.shown_frames += 1
        return TrackState(
            visible=visible,
            opacity=round(self._opacity, 4),
            quad=None if self._quad is None else self._quad.copy(),
        )
