"""Detection metrics.

There are two different questions here and conflating them produced a badly
misleading result once already, so they are kept apart by name:

**Geometric** -- did the detector's region actually land on a crosswalk?
This is detection quality. It needs labelled crosswalk polygons and is the
only number that may be quoted as precision/recall/F1.

**Temporal** -- did the detector fire during a second that is known to contain
a crosswalk *somewhere* in frame? This is a coincidence counter. It says
nothing about whether the right thing was found, and on the first measurement
of the classical baseline it reported 30 "on-truth" detections of which, on
inspection, **zero** overlapped a crosswalk. Functions here label it
`temporal (not detection quality)` wherever it is printed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

#: Minimum IoU for a detection to count as having found a labelled crosswalk.
#: Deliberately lenient: the question is "did it land on the crossing at all",
#: not "is the outline tight".
DEFAULT_IOU = 0.10


def _fill(polygon: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def mask_iou(
    polygon_a: np.ndarray, polygon_b: np.ndarray, height: int, width: int
) -> float:
    """IoU of two polygons by rasterisation.

    Rasterised rather than computed analytically because a hand-labelled
    crosswalk outline is not necessarily convex, and `intersectConvexConvex`
    silently gives nonsense for non-convex input.
    """
    if len(polygon_a) < 3 or len(polygon_b) < 3:
        return 0.0
    a = _fill(polygon_a, height, width)
    b = _fill(polygon_b, height, width)
    intersection = int(np.count_nonzero(a & b))
    if intersection == 0:
        return 0.0
    union = int(np.count_nonzero(a | b))
    return intersection / union if union else 0.0


def best_iou(
    detection: np.ndarray,
    truth_polygons: list[np.ndarray],
    height: int,
    width: int,
) -> float:
    """Highest IoU between a detection and any labelled polygon in the frame."""
    return max(
        (mask_iou(detection, truth, height, width) for truth in truth_polygons),
        default=0.0,
    )


@dataclass(frozen=True)
class GeometricResult:
    """Detection quality. Safe to quote as precision / recall / F1."""

    iou_threshold: float
    frames_with_labels: int
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_iou: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        fired = self.true_positives + self.false_positives
        return self.true_positives / fired if fired else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def mean_iou(self) -> float:
        return sum(self.matched_iou) / len(self.matched_iou) if self.matched_iou else 0.0

    def as_lines(self) -> list[str]:
        return [
            f"GEOMETRIC (detection quality, IoU >= {self.iou_threshold:.2f})",
            f"  frames with labels : {self.frames_with_labels}",
            f"  true positives     : {self.true_positives}",
            f"  false positives    : {self.false_positives}",
            f"  false negatives    : {self.false_negatives}",
            f"  precision          : {self.precision:.3f}",
            f"  recall             : {self.recall:.3f}",
            f"  F1                 : {self.f1:.3f}",
            f"  mean IoU (matched) : {self.mean_iou:.3f}",
        ]


@dataclass(frozen=True)
class TemporalResult:
    """Coincidence counter. NOT detection quality -- see module docstring."""

    threshold: float
    fired_seconds: int
    on_truth: int
    off_truth: int
    missed: list[int] = field(default_factory=list)
    false_positive_seconds: list[int] = field(default_factory=list)

    def as_lines(self) -> list[str]:
        return [
            f"TEMPORAL (not detection quality) at score >= {self.threshold:.2f}",
            "  Counts a firing as 'on-truth' when the second is labelled as",
            "  containing a crossing somewhere in frame. It does NOT check that",
            "  the detected region overlaps the crossing, so it can be high while",
            "  the detector is pointing at a manhole cover. Do not quote it as",
            "  precision or recall.",
            f"  fired on          : {self.fired_seconds} second(s)",
            f"  on-truth  (temporal): {self.on_truth}",
            f"  off-truth (temporal): {self.off_truth}",
            f"  seconds missed      : {self.missed}",
            f"  off-truth seconds   : {self.false_positive_seconds[:30]}",
        ]


def evaluate_geometric(
    frames: list[dict],
    iou_threshold: float = DEFAULT_IOU,
) -> GeometricResult:
    """Score detections against labelled polygons.

    `frames` is a list of dicts with keys:
      `detections`  list of polygons the detector produced (may be empty)
      `truth`       list of labelled crosswalk polygons (may be empty)
      `height`, `width`

    One labelled crosswalk may be matched at most once; extra detections on the
    same crossing count as false positives.
    """
    tp = fp = fn = 0
    matched_iou: list[float] = []
    counted = 0
    for frame in frames:
        truth = list(frame.get("truth") or [])
        detections = list(frame.get("detections") or [])
        height, width = frame["height"], frame["width"]
        counted += 1
        unmatched = list(range(len(truth)))
        for detection in detections:
            best_index, best_score = -1, 0.0
            for index in unmatched:
                score = mask_iou(detection, truth[index], height, width)
                if score > best_score:
                    best_index, best_score = index, score
            if best_index >= 0 and best_score >= iou_threshold:
                tp += 1
                matched_iou.append(best_score)
                unmatched.remove(best_index)
            else:
                fp += 1
        fn += len(unmatched)
    return GeometricResult(
        iou_threshold=iou_threshold,
        frames_with_labels=counted,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        matched_iou=matched_iou,
    )


def evaluate_temporal(
    scored_seconds: list[tuple[float, float]],
    truth_seconds: set[int],
    threshold: float,
) -> TemporalResult:
    """Coincidence counter over `(seconds, score)` pairs. Not detection quality."""
    available = {int(round(s)) for s, _ in scored_seconds}
    truth = truth_seconds & available
    predicted = {int(round(s)) for s, score in scored_seconds if score >= threshold}
    return TemporalResult(
        threshold=threshold,
        fired_seconds=len(predicted),
        on_truth=len(predicted & truth),
        off_truth=len(predicted - truth),
        missed=sorted(truth - predicted),
        false_positive_seconds=sorted(predicted - truth),
    )
