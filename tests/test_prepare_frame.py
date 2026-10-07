import numpy as np
import pytest

from ped_lane.config import Config, parse_ignore_regions
from ped_lane.overlay import blank_regions, region_pixel_box
from ped_lane.run import prepare_frame


def test_region_pixel_box_scales_fractions():
    assert region_pixel_box(200, 100, 0.0, 0.5, 0.5, 1.0) == (0, 50, 100, 100)


def test_region_pixel_box_accepts_reversed_corners():
    assert region_pixel_box(100, 100, 0.8, 0.9, 0.2, 0.3) == (20, 30, 80, 90)


def test_region_pixel_box_clamps_out_of_range_fractions():
    assert region_pixel_box(100, 100, -0.5, -0.5, 1.5, 1.5) == (0, 0, 100, 100)


def test_region_pixel_box_rejects_empty_frame():
    with pytest.raises(ValueError):
        region_pixel_box(0, 100, 0.0, 0.0, 1.0, 1.0)


def test_blank_regions_zeroes_only_the_region():
    frame = np.full((100, 100, 3), 200, dtype=np.uint8)

    out = blank_regions(frame, [(0.0, 0.0, 0.5, 0.5)])

    assert out[10, 10].sum() == 0
    assert tuple(out[80, 80]) == (200, 200, 200)
    assert frame.sum() > 0  # input untouched


def test_blank_regions_without_regions_avoids_a_copy():
    frame = np.full((8, 8, 3), 3, dtype=np.uint8)
    assert blank_regions(frame, []) is frame


def test_blank_regions_ignores_a_zero_area_region():
    frame = np.full((50, 50, 3), 9, dtype=np.uint8)
    out = blank_regions(frame, [(0.5, 0.5, 0.5, 0.9)])
    assert np.array_equal(out, frame)


def test_parse_ignore_regions_mapping_form():
    raw = [{"x0": 0.1, "y0": 0.2, "x1": 0.3, "y1": 0.4}]
    assert parse_ignore_regions(raw) == ((0.1, 0.2, 0.3, 0.4),)


def test_parse_ignore_regions_sequence_form_and_empty():
    assert parse_ignore_regions([[0, 0, 1, 1]]) == ((0.0, 0.0, 1.0, 1.0),)
    assert parse_ignore_regions(None) == ()
    assert parse_ignore_regions([]) == ()


def test_parse_ignore_regions_rejects_a_typo_loudly():
    # A region that silently does nothing would look like a model failure.
    with pytest.raises(ValueError, match="ignore_regions\\[0\\]"):
        parse_ignore_regions([{"x0": 0.1, "y0": 0.2, "x1": 0.3}])
    with pytest.raises(ValueError):
        parse_ignore_regions([[0.1, 0.2]])


def test_config_round_trips_ignore_regions():
    cfg = Config.from_dict(
        {"dashcam": {"ignore_regions": [{"x0": 0.0, "y0": 0.9, "x1": 0.3, "y1": 1.0}]}}
    )
    assert cfg.dashcam.ignore_regions == ((0.0, 0.9, 0.3, 1.0),)


def test_prepare_frame_blanks_before_cropping():
    # The region covers rows 0-10% of the ORIGINAL frame; a 20% sky crop must
    # not shift it, so the blanked rows are gone entirely and row 0 of the
    # output is ordinary image content.
    frame = np.full((100, 60, 3), 150, dtype=np.uint8)
    cfg = Config.from_dict(
        {
            "dashcam": {
                "crop_top_fraction": 0.2,
                "crop_bottom_fraction": 0.1,
                "ignore_regions": [{"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 0.1}],
            }
        }
    )

    out = prepare_frame(frame, cfg.dashcam)

    assert out.shape == (70, 60, 3)
    assert tuple(out[0, 0]) == (150, 150, 150)


def test_prepare_frame_keeps_a_region_inside_the_kept_rows():
    frame = np.full((100, 60, 3), 150, dtype=np.uint8)
    cfg = Config.from_dict(
        {
            "dashcam": {
                "crop_top_fraction": 0.1,
                "crop_bottom_fraction": 0.1,
                # bottom-left corner sticker, inside the kept band
                "ignore_regions": [{"x0": 0.0, "y0": 0.8, "x1": 0.2, "y1": 0.88}],
            }
        }
    )

    out = prepare_frame(frame, cfg.dashcam)

    assert out.shape == (80, 60, 3)
    assert out[int(0.82 * 100) - 10, 5].sum() == 0


def test_prepare_frame_without_regions_matches_a_plain_crop():
    frame = np.random.default_rng(0).integers(0, 255, (90, 40, 3), dtype=np.uint8)
    cfg = Config.from_dict({"dashcam": {"crop_top_fraction": 0.0, "crop_bottom_fraction": 0.2}})

    out = prepare_frame(frame, cfg.dashcam)

    assert np.array_equal(out, frame[0:72])
