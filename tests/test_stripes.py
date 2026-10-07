"""The classical baseline, on synthetic geometry.

Synthetic rather than real frames on purpose: with only 9 real positives,
tuning against them and then testing on them would be measuring nothing. These
pin the *geometric reasoning* -- a ladder is a crossing, beads on a string are
a lane line -- which is the part that should hold on footage we have not seen.
"""
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from ped_lane.stripes import (
    StripeParams,
    detect,
    find_stripes,
    group_stripes,
    road_band,
    vanishing_point_for,
)

W, H = 1920, 400
VP = (960.0, 20.0)


def _blank() -> np.ndarray:
    return np.full((H, W, 3), 40, dtype=np.uint8)


def _crossing(n: int = 6) -> np.ndarray:
    """A ladder: wide bars stacked in depth, spacing stretching toward us."""
    img = _blank()
    y = 90.0
    gap = 16.0
    for _ in range(n):
        half_w = 120 + (y - 90) * 1.4
        thickness = max(5, int(gap * 0.45))
        cv2.rectangle(
            img,
            (int(VP[0] - half_w), int(y)),
            (int(VP[0] + half_w), int(y + thickness)),
            (235, 235, 235),
            -1,
        )
        y += gap
        gap *= 1.35
    return img


def _dashed_lane_line(n: int = 7) -> np.ndarray:
    """Beads on a string: dashes strung out ALONG one line toward the VP."""
    img = _blank()
    y = 90.0
    gap = 20.0
    for _ in range(n):
        length = max(8, int(gap * 0.6))
        width = max(4, int(gap * 0.22))
        cv2.rectangle(
            img,
            (int(VP[0] - width / 2), int(y)),
            (int(VP[0] + width / 2), int(y + length)),
            (235, 235, 235),
            -1,
        )
        y += gap + length
        gap *= 1.3
    return img


def _single_stop_line() -> np.ndarray:
    img = _blank()
    cv2.rectangle(img, (int(VP[0] - 300), 250), (int(VP[0] + 300), 272), (235, 235, 235), -1)
    return img


# --- the geometric discrimination ------------------------------------------

def test_a_ladder_is_detected_as_a_crossing():
    result = detect(_crossing(), VP)
    assert result.found
    assert result.score > 0.0
    assert result.groups[0].count >= StripeParams().min_stripes


def test_more_bars_scores_higher_than_fewer():
    assert detect(_crossing(7), VP).score >= detect(_crossing(4), VP).score


def test_a_dashed_lane_line_is_rejected():
    # Beads on a string: axes lie ALONG the line of centres, not across it.
    assert not detect(_dashed_lane_line(), VP).found


def test_a_single_stop_line_is_rejected():
    # Correct orientation, but one bar is not a ladder.
    assert not detect(_single_stop_line(), VP).found


def test_empty_road_finds_nothing():
    assert not detect(_blank(), VP).found


def test_three_bars_are_not_enough():
    assert not detect(_crossing(3), VP).found


def test_score_is_zero_when_nothing_is_found():
    assert detect(_blank(), VP).score == 0.0


# --- components -------------------------------------------------------------

def test_find_stripes_picks_up_the_painted_bars():
    stripes = find_stripes(_crossing(6))
    assert len(stripes) >= 4
    assert all(s.length >= s.width for s in stripes)


def test_find_stripes_on_an_empty_array_is_safe():
    assert find_stripes(np.zeros((0, 0, 3), dtype=np.uint8)) == []


def test_grouping_needs_the_minimum_count():
    stripes = find_stripes(_crossing(6))
    strict = StripeParams(min_stripes=99)
    assert group_stripes(stripes, VP, W, strict, frame_height=H) == []


def test_road_band_crops_to_the_vanishing_point_and_below():
    frame = np.zeros((1000, 100, 3), dtype=np.uint8)
    band, offset = road_band(frame, vp_y=600, margin_fraction=0.05)
    assert offset == 550
    assert band.shape[0] == 450


def test_road_band_clamps_a_vanishing_point_off_the_top():
    frame = np.zeros((100, 10, 3), dtype=np.uint8)
    band, offset = road_band(frame, vp_y=-500)
    assert offset == 0
    assert band.shape[0] == 100


def test_vanishing_point_defaults_are_the_measured_ones():
    # Measured on samples/dashcam.mp4 by optical-flow focus-of-expansion.
    x, y = vanishing_point_for(1920, 1080)
    assert x == pytest.approx(1069.4, abs=1.0)
    assert y == pytest.approx(732.2, abs=1.0)


def test_clahe_is_off_by_default():
    # It drags the Otsu threshold above the bars on a dark road band; measured
    # to take the bar count at t=100s from 20 to 4.
    assert StripeParams().use_clahe is False


def test_road_trapezoid_is_off_by_default():
    # Measured to remove real bars and drop recall to zero on our 9 positives.
    assert StripeParams().use_road_trapezoid is False
