"""The burned-in GPS/date overlay must never reach a saved frame.

This is the privacy guarantee the whole extraction pipeline rests on, so it is
tested two ways: synthetically (always runs, in CI or on a clean checkout) and
against the real clip when it is present, using the same always-bright
detector that located the overlay in the first place.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pathlib import Path

from ped_lane.config import Config, load_config
from ped_lane.overlay import apply_top_crop, redact, top_crop_rows
from ped_lane.run import prepare_frame

CLIP = Path("samples/dashcam.mp4")

# Measured from samples/dashcam.mp4 over 200 frames spanning the whole clip:
# the burned-in "NEXTBASE <lat> <lon> <speed> <time> <date>" bar.
OVERLAY_TOP_FRACTION = 0.9565
OVERLAY_BOTTOM_FRACTION = 0.9769


# --- synthetic: always runs -------------------------------------------------

def test_redact_removes_a_bar_in_the_overlay_band():
    h, w = 1080, 1920
    frame = np.full((h, w, 3), 40, dtype=np.uint8)
    y0, y1 = int(h * OVERLAY_TOP_FRACTION), int(h * OVERLAY_BOTTOM_FRACTION)
    frame[y0:y1, int(w * 0.16):int(w * 0.76)] = 255  # stand-in for the text

    out = redact(frame, [], crop_bottom_fraction=0.10)

    assert out.shape[0] == h - int(h * 0.10)
    assert out.max() < 255  # the bright bar is gone entirely


def test_redact_blanks_the_ignore_region():
    frame = np.full((1080, 1920, 3), 200, dtype=np.uint8)
    out = redact(frame, [(0.87, 0.76, 1.0, 1.0)], crop_bottom_fraction=0.10)
    assert out[int(1080 * 0.80), int(1920 * 0.95)].sum() == 0
    assert out[int(1080 * 0.80), int(1920 * 0.50)].sum() > 0


def test_redact_does_not_apply_the_top_crop():
    # crop_top must stay out of saved frames so retuning it never forces
    # re-extraction or re-labelling.
    frame = np.full((1000, 100, 3), 7, dtype=np.uint8)
    out = redact(frame, [], crop_bottom_fraction=0.10)
    assert out.shape[0] == 900  # only the bottom 10% removed


def test_top_crop_rows_uses_the_original_height_not_the_redacted_one():
    # A redacted 1080-high frame is 972 high; the crop must still be computed
    # from 1080, or the horizon silently shifts.
    assert top_crop_rows(1080, 0.40) == 432
    assert top_crop_rows(972, 0.40) == 388  # wrong input -> wrong answer, by 44 px


def test_top_crop_rows_clamps_and_validates():
    assert top_crop_rows(1080, -1.0) == 0
    assert top_crop_rows(1080, 5.0) == int(1080 * 0.99)
    with pytest.raises(ValueError):
        top_crop_rows(0, 0.4)


def test_apply_top_crop_never_empties_the_frame():
    frame = np.zeros((100, 10, 3), dtype=np.uint8)
    assert apply_top_crop(frame, 0).shape[0] == 100
    assert apply_top_crop(frame, 40).shape[0] == 60
    assert apply_top_crop(frame, 100).shape[0] == 100  # refuses to wipe it out
    assert apply_top_crop(frame, 500).shape[0] == 100


def test_redact_then_top_crop_equals_the_live_pipeline():
    frame = np.random.default_rng(1).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    cfg = load_config("config.yaml")

    staged = apply_top_crop(
        redact(frame, cfg.dashcam.ignore_regions, cfg.dashcam.crop_bottom_fraction),
        top_crop_rows(frame.shape[0], cfg.dashcam.crop_top_fraction),
    )

    assert np.array_equal(staged, prepare_frame(frame, cfg.dashcam))


# --- against the real clip --------------------------------------------------

def test_overlay_band_is_geometrically_outside_the_redacted_frame():
    """The strongest form of the guarantee: the overlay rows cannot be kept.

    Deterministic and video-free, so it also guards a clean checkout. If
    anyone lowers crop_bottom_fraction below what the overlay needs, this
    fails immediately rather than at the next privacy review.
    """
    cfg = load_config("config.yaml")
    source_height = 1080
    kept_rows = source_height - int(source_height * cfg.dashcam.crop_bottom_fraction)
    overlay_top_row = int(source_height * OVERLAY_TOP_FRACTION)

    assert kept_rows <= overlay_top_row, (
        f"crop_bottom_fraction={cfg.dashcam.crop_bottom_fraction} keeps {kept_rows} rows, "
        f"but the burned-in overlay starts at row {overlay_top_row}"
    )


@pytest.mark.skipif(not CLIP.exists(), reason="samples/dashcam.mp4 not present")
def test_no_burned_in_text_survives_redaction_of_the_real_clip():
    """Pixels bright in EVERY sampled frame, in the strip where the overlay
    would land if the crop were wrong.

    Restricted to the bottom of the redacted frame on purpose: over a small
    sample, sky is bright in every frame too, and a whole-frame check would
    fail on scene content rather than on a privacy defect.
    """
    cfg = load_config("config.yaml")
    cap = cv2.VideoCapture(str(CLIP))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        running_min = None
        for i in (int(round(k * (total - 1) / 39)) for k in range(40)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                continue
            out = redact(
                frame, cfg.dashcam.ignore_regions, cfg.dashcam.crop_bottom_fraction
            )
            grey = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            running_min = grey if running_min is None else np.minimum(running_min, grey)
    finally:
        cap.release()

    assert running_min is not None
    bottom_strip = running_min[int(running_min.shape[0] * 0.80):]
    assert int((bottom_strip > 150).sum()) == 0


@pytest.mark.skipif(not CLIP.exists(), reason="samples/dashcam.mp4 not present")
def test_ignore_region_covers_the_windscreen_clutter_on_the_real_clip():
    """The blue windscreen object swings; its above-crop extent must stay masked.

    Keyed on *persistence*, not on any single frame: sky-lit shadows on tarmac
    are strongly blue too, so a per-frame colour test fails on scene content.
    Only a pixel that is blue in a meaningful fraction of frames is fixed
    windscreen clutter.
    """
    cfg = load_config("config.yaml")
    cap = cv2.VideoCapture(str(CLIP))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    try:
        persistence = None
        sampled = 0
        for i in (int(round(k * (total - 1) / 59)) for k in range(60)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            if not ok:
                continue
            out = redact(
                frame, cfg.dashcam.ignore_regions, cfg.dashcam.crop_bottom_fraction
            )
            b = out[:, :, 0].astype(np.int16)
            g = out[:, :, 1].astype(np.int16)
            r = out[:, :, 2].astype(np.int16)
            hit = ((b - np.maximum(r, g)) > 25).astype(np.float32)
            persistence = hit if persistence is None else persistence + hit
            sampled += 1
    finally:
        cap.release()

    assert sampled >= 50
    persistence /= sampled
    lower_right = np.zeros(persistence.shape, dtype=bool)
    lower_right[
        int(persistence.shape[0] * 0.70):, int(persistence.shape[1] * 0.70):
    ] = True
    persistent_blue = int((np.where(lower_right, persistence, 0) > 0.05).sum())
    assert persistent_blue == 0


@pytest.mark.skipif(not CLIP.exists(), reason="samples/dashcam.mp4 not present")
def test_redacted_clip_frames_have_the_expected_shape():
    cfg = load_config("config.yaml")
    cap = cv2.VideoCapture(str(CLIP))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    assert ok
    out = redact(frame, cfg.dashcam.ignore_regions, cfg.dashcam.crop_bottom_fraction)
    assert out.shape[:2] == (972, 1920)
