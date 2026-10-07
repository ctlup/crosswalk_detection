import numpy as np
import pytest

from ped_lane import classes
from ped_lane.overlay import (
    apply_crop,
    crop_bounds,
    draw_hud,
    draw_masks,
    draw_polygons,
)


def test_default_hood_crop():
    top, bottom = crop_bounds(1000, 0.0, 0.18)
    assert (top, bottom) == (0, 820)


def test_sky_and_hood_crop_together():
    top, bottom = crop_bounds(1000, 0.2, 0.2)
    assert (top, bottom) == (200, 800)


def test_no_crop_keeps_every_row():
    assert crop_bounds(720, 0.0, 0.0) == (0, 720)


def test_negative_fractions_are_clamped():
    assert crop_bounds(720, -1.0, -0.5) == (0, 720)


def test_degenerate_crop_keeps_the_whole_frame():
    # A misconfigured crop should degrade the view, not kill the loop mid-drive.
    assert crop_bounds(480, 0.9, 0.9) == (0, 480)
    assert crop_bounds(480, 0.99, 0.99) == (0, 480)


def test_zero_height_rejected():
    with pytest.raises(ValueError):
        crop_bounds(0, 0.0, 0.1)


def test_apply_crop_matches_bounds():
    frame = np.zeros((100, 50, 3), dtype=np.uint8)
    top, bottom = crop_bounds(100, 0.1, 0.2)
    assert apply_crop(frame, top, bottom).shape == (bottom - top, 50, 3)


def test_draw_masks_colours_by_class_and_leaves_input_untouched():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True

    out = draw_masks(frame, [mask], [classes.CROSSWALK], alpha=1.0, draw_outlines=False)

    assert tuple(out[7, 7]) == classes.COLORS_BGR[classes.CROSSWALK]
    assert tuple(out[0, 0]) == (0, 0, 0)
    assert frame.sum() == 0


def test_draw_masks_with_nothing_returns_a_copy():
    frame = np.full((8, 8, 3), 7, dtype=np.uint8)
    out = draw_masks(frame, [], [], alpha=0.4)
    assert np.array_equal(out, frame)
    assert out is not frame


def test_draw_polygons_fills_the_polygon():
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    square = np.array([[5, 5], [25, 5], [25, 25], [5, 25]], dtype=np.float32)

    out = draw_polygons(frame, [square], [classes.VEHICLE], alpha=1.0)

    assert tuple(out[15, 15]) == classes.COLORS_BGR[classes.VEHICLE]


def test_draw_polygons_skips_degenerate_shapes():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    out = draw_polygons(frame, [np.array([[1, 1], [2, 2]], dtype=np.float32)], [0], 1.0)
    assert out.sum() == 0


def test_draw_hud_writes_into_the_frame_without_resizing_it():
    frame = np.zeros((120, 320, 3), dtype=np.uint8)
    draw_hud(frame, ["12.3 FPS", "infer 8.1 ms"])
    assert frame.shape == (120, 320, 3)
    assert frame.sum() > 0


def test_draw_hud_on_a_tiny_frame_does_not_crash():
    frame = np.zeros((12, 16, 3), dtype=np.uint8)
    draw_hud(frame, ["a very long line that cannot possibly fit"])
    assert frame.shape == (12, 16, 3)
