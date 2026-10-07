"""Geometric vs temporal metrics.

The distinction is the whole point of this module: the classical baseline once
reported 30 "on-truth" detections of which zero actually overlapped a
crosswalk. Temporal agreement is not detection quality and must never be
printed as if it were.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from ped_lane.evaluation import (
    DEFAULT_IOU,
    best_iou,
    evaluate_geometric,
    evaluate_temporal,
    mask_iou,
)

H, W = 400, 600
BOX = np.array([[100, 100], [300, 100], [300, 200], [100, 200]], dtype=np.float32)


def _shift(polygon, dx, dy):
    return polygon + np.array([dx, dy], dtype=np.float32)


def _frame(detections, truth):
    return {"detections": detections, "truth": truth, "height": H, "width": W}


# --- IoU --------------------------------------------------------------------

def test_identical_polygons_have_iou_one():
    assert mask_iou(BOX, BOX, H, W) == pytest.approx(1.0, abs=0.01)


def test_disjoint_polygons_have_iou_zero():
    assert mask_iou(BOX, _shift(BOX, 400, 0), H, W) == 0.0


def test_partial_overlap_is_between():
    assert 0.0 < mask_iou(BOX, _shift(BOX, 100, 0), H, W) < 1.0


def test_iou_handles_a_non_convex_outline():
    # A hand-traced crosswalk is not necessarily convex; the rasterised IoU
    # must cope where a convex-only routine would not.
    l_shape = np.array(
        [[100, 100], [300, 100], [300, 140], [160, 140], [160, 220], [100, 220]],
        dtype=np.float32,
    )
    assert 0.0 < mask_iou(l_shape, BOX, H, W) < 1.0


def test_degenerate_polygons_score_zero():
    assert mask_iou(np.array([[0, 0], [1, 1]], dtype=np.float32), BOX, H, W) == 0.0


def test_best_iou_picks_the_closest_truth():
    truths = [_shift(BOX, 400, 0), _shift(BOX, 10, 0), _shift(BOX, 250, 0)]
    assert best_iou(BOX, truths, H, W) == pytest.approx(
        mask_iou(BOX, _shift(BOX, 10, 0), H, W)
    )


def test_best_iou_without_any_truth_is_zero():
    assert best_iou(BOX, [], H, W) == 0.0


# --- geometric --------------------------------------------------------------

def test_a_detection_on_the_crosswalk_is_a_true_positive():
    result = evaluate_geometric([_frame([BOX], [BOX])])
    assert (result.true_positives, result.false_positives, result.false_negatives) == (1, 0, 0)
    assert result.precision == 1.0 and result.recall == 1.0


def test_a_detection_somewhere_else_is_a_false_positive_and_a_miss():
    # This is the baseline's actual failure: it fired, during a second that
    # does contain a crossing, on something else entirely.
    result = evaluate_geometric([_frame([_shift(BOX, 400, 150)], [BOX])])
    assert result.true_positives == 0
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.precision == 0.0 and result.recall == 0.0 and result.f1 == 0.0


def test_an_unlabelled_frame_with_a_detection_is_pure_false_positive():
    result = evaluate_geometric([_frame([BOX], [])])
    assert (result.true_positives, result.false_positives, result.false_negatives) == (0, 1, 0)


def test_a_missed_crosswalk_is_a_false_negative():
    result = evaluate_geometric([_frame([], [BOX])])
    assert (result.true_positives, result.false_positives, result.false_negatives) == (0, 0, 1)


def test_one_crosswalk_can_only_be_matched_once():
    result = evaluate_geometric([_frame([BOX, _shift(BOX, 5, 5)], [BOX])])
    assert result.true_positives == 1
    assert result.false_positives == 1


def test_iou_threshold_is_respected():
    detection = _shift(BOX, 170, 0)  # small overlap
    loose = evaluate_geometric([_frame([detection], [BOX])], iou_threshold=0.01)
    strict = evaluate_geometric([_frame([detection], [BOX])], iou_threshold=0.9)
    assert loose.true_positives == 1
    assert strict.true_positives == 0


def test_default_iou_is_lenient_on_purpose():
    # The question is "did it land on the crossing at all", not "is the
    # outline tight", so a low bar is correct here.
    assert DEFAULT_IOU <= 0.2


def test_geometric_report_is_labelled_as_detection_quality():
    text = "\n".join(evaluate_geometric([_frame([BOX], [BOX])]).as_lines())
    assert "GEOMETRIC (detection quality" in text


# --- temporal ---------------------------------------------------------------

def test_temporal_counts_coincidence_not_correctness():
    scored = [(13.0, 0.9), (20.0, 0.9), (30.0, 0.1)]
    result = evaluate_temporal(scored, {13, 40}, threshold=0.4)
    assert result.on_truth == 1      # 13 s
    assert result.off_truth == 1     # 20 s
    assert result.missed == []       # 40 s was never scored, so not counted


def test_temporal_only_scores_seconds_that_were_looked_at():
    result = evaluate_temporal([(5.0, 0.1)], {99}, threshold=0.4)
    assert result.missed == []
    assert result.on_truth == 0


def test_temporal_records_what_was_missed():
    result = evaluate_temporal([(13.0, 0.1), (14.0, 0.9)], {13, 14}, threshold=0.4)
    assert result.missed == [13]
    assert result.on_truth == 1


def test_temporal_report_warns_that_it_is_not_detection_quality():
    text = "\n".join(evaluate_temporal([(13.0, 0.9)], {13}, 0.4).as_lines())
    assert "TEMPORAL (not detection quality)" in text
    assert "manhole" in text  # the concrete failure it is warning about
    assert "Do not quote it as" in text


def test_temporal_has_no_precision_or_recall_attributes():
    # Deliberately absent so it cannot be reported as detection quality.
    result = evaluate_temporal([(13.0, 0.9)], {13}, 0.4)
    assert not hasattr(result, "precision")
    assert not hasattr(result, "recall")
    assert not hasattr(result, "f1")
