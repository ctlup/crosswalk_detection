"""Display smoothing. It may change WHEN a region is drawn, never WHAT fired."""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from ped_lane.tracking import (
    RegionTracker,
    TrackerParams,
    fit_quad,
    order_quad,
    quad_iou,
)

SQUARE = np.array([[100, 100], [300, 100], [300, 200], [100, 200]], dtype=np.float32)


def _run(tracker, frames):
    """Feed (score, polygon) pairs; return the TrackState list."""
    return [tracker.update(score, polygon) for score, polygon in frames]


# --- geometry ---------------------------------------------------------------

def test_order_quad_is_stable_under_input_permutation():
    # If corner order depended on input order, the EMA would average corner 0
    # against a different physical corner and fold the shape inside out.
    base = order_quad(SQUARE)
    for roll in range(4):
        assert np.allclose(order_quad(np.roll(SQUARE, roll, axis=0)), base, atol=1e-3)


def test_order_quad_starts_near_the_top_left():
    ordered = order_quad(SQUARE)
    assert np.allclose(ordered[0], [100, 100], atol=1.0)


def test_order_quad_rejects_the_wrong_point_count():
    with pytest.raises(ValueError):
        order_quad(np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32))


def test_fit_quad_of_a_rectangle_is_that_rectangle():
    quad = fit_quad(SQUARE)
    assert quad.shape == (4, 2)
    assert cv2.contourArea(quad) == pytest.approx(200 * 100, rel=0.05)


def test_fit_quad_simplifies_a_many_sided_hull():
    hull = np.array(
        [[100, 100], [200, 98], [300, 100], [302, 150], [300, 200], [200, 203],
         [100, 200], [98, 150]], dtype=np.float32
    )
    assert fit_quad(hull).shape == (4, 2)


def test_quad_iou_bounds():
    assert quad_iou(SQUARE, SQUARE) == pytest.approx(1.0, abs=1e-3)
    far = SQUARE + np.array([5000, 5000], dtype=np.float32)
    assert quad_iou(SQUARE, far) == 0.0


def test_quad_iou_partial_overlap_is_between():
    shifted = SQUARE + np.array([100, 0], dtype=np.float32)
    assert 0.0 < quad_iou(SQUARE, shifted) < 1.0


# --- hysteresis -------------------------------------------------------------

def test_a_single_spike_never_appears():
    # One loud frame is what makes an overlay look broken.
    tracker = RegionTracker(TrackerParams(enter_frames=3))
    states = _run(tracker, [(0.0, None), (0.9, SQUARE), (0.0, None), (0.0, None)])
    assert not any(s.visible for s in states)


def test_it_appears_only_after_enter_frames():
    tracker = RegionTracker(TrackerParams(enter_frames=3, fade_frames=1))
    states = _run(tracker, [(0.9, SQUARE)] * 5)
    assert [s.visible for s in states] == [False, False, True, True, True]


def test_it_survives_a_brief_score_dip():
    # The gap between exit_score and enter_score is the whole point.
    params = TrackerParams(enter_frames=2, exit_frames=10, fade_frames=1)
    tracker = RegionTracker(params)
    _run(tracker, [(0.9, SQUARE)] * 3)
    dipped = _run(tracker, [(0.30, SQUARE)] * 6)  # below enter, above exit
    assert all(s.visible for s in dipped)


def test_it_disappears_only_after_exit_frames():
    params = TrackerParams(enter_frames=2, exit_frames=4, fade_frames=1, hold_frames=99)
    tracker = RegionTracker(params)
    _run(tracker, [(0.9, SQUARE)] * 3)
    after = _run(tracker, [(0.0, SQUARE)] * 6)
    # Hidden ON the 4th consecutive below-exit frame, not after it.
    assert [s.visible for s in after] == [True, True, True, False, False, False]


def test_a_dropout_holds_the_last_shape():
    params = TrackerParams(enter_frames=2, exit_frames=20, hold_frames=8, fade_frames=1)
    tracker = RegionTracker(params)
    _run(tracker, [(0.9, SQUARE)] * 3)
    held = _run(tracker, [(0.9, None)] * 4)  # detector found nothing these frames
    assert all(s.visible for s in held)
    assert all(s.quad is not None for s in held)


# --- smoothing --------------------------------------------------------------

def test_corners_move_gradually_not_instantly():
    params = TrackerParams(enter_frames=1, fade_frames=1, ema_alpha=0.35)
    tracker = RegionTracker(params)
    tracker.update(0.9, SQUARE)
    moved = SQUARE + np.array([40, 0], dtype=np.float32)
    state = tracker.update(0.9, moved)
    shift = state.quad[:, 0].mean() - SQUARE[:, 0].mean()
    assert 0 < shift < 40  # partway, not snapped


def test_repeated_updates_converge_on_the_new_shape():
    tracker = RegionTracker(TrackerParams(enter_frames=1, fade_frames=1))
    tracker.update(0.9, SQUARE)
    moved = SQUARE + np.array([40, 0], dtype=np.float32)
    for _ in range(40):
        state = tracker.update(0.9, moved)
    assert state.quad[:, 0].mean() == pytest.approx(moved[:, 0].mean(), abs=2.0)


def test_opacity_ramps_rather_than_popping():
    tracker = RegionTracker(TrackerParams(enter_frames=1, fade_frames=4))
    opacities = [s.opacity for s in _run(tracker, [(0.9, SQUARE)] * 5)]
    assert opacities == sorted(opacities)
    assert opacities[0] < 1.0 and opacities[-1] == pytest.approx(1.0)


def test_opacity_ramps_back_down():
    params = TrackerParams(enter_frames=1, exit_frames=1, fade_frames=4, hold_frames=0)
    tracker = RegionTracker(params)
    _run(tracker, [(0.9, SQUARE)] * 6)
    fading = [s.opacity for s in _run(tracker, [(0.0, None)] * 3)]
    assert fading == sorted(fading, reverse=True)


# --- honesty of the counters ------------------------------------------------

def test_raw_and_shown_counts_are_tracked_separately():
    # The smoothing changes the display; the report must still show what the
    # detector actually did.
    params = TrackerParams(enter_frames=3, exit_frames=10, fade_frames=1)
    tracker = RegionTracker(params)
    _run(tracker, [(0.9, SQUARE)] * 5 + [(0.0, None)] * 5)
    assert tracker.raw_fired_frames == 5
    assert tracker.shown_frames != tracker.raw_fired_frames


def test_smoothing_suppresses_isolated_false_positives():
    params = TrackerParams(enter_frames=3, fade_frames=1)
    tracker = RegionTracker(params)
    _run(tracker, [(0.9, SQUARE), (0.0, None)] * 6)
    assert tracker.raw_fired_frames == 6
    assert tracker.shown_frames == 0


def test_nothing_is_shown_without_a_polygon():
    tracker = RegionTracker(TrackerParams(enter_frames=1, fade_frames=1))
    states = _run(tracker, [(0.95, None)] * 5)
    assert not any(s.visible for s in states)
