"""Teacher evaluation: frame selection, mask->polygon, and the render rules.

Nothing here loads the checkpoint; these are the parts that must be right
before 826 MB of CC BY-NC weights are fetched.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

_SPEC = importlib.util.spec_from_file_location(
    "teacher_eval", Path(__file__).resolve().parents[1] / "scripts" / "teacher_eval.py"
)
te = importlib.util.module_from_spec(_SPEC)
sys.modules["teacher_eval"] = te
_SPEC.loader.exec_module(te)

from ped_lane import classes, mapillary  # noqa: E402
from ped_lane.teacher import Segment, TeacherResult  # noqa: E402

POSITIVES = {13, 14, 42, 43, 44, 99, 100, 148, 149}


def _manifest(seconds):
    return {"frames": [{"file": f"f_{s:05.1f}.jpg", "seconds": float(s)} for s in seconds]}


# --- frame selection --------------------------------------------------------

def test_every_labelled_positive_is_selected():
    manifest = _manifest(range(0, 180))
    chosen = te.select_frames(manifest, POSITIVES, empty_count=20)
    assert POSITIVES <= {int(round(r["seconds"])) for r in chosen}


def test_the_requested_number_of_empties_is_added():
    manifest = _manifest(range(0, 180))
    chosen = te.select_frames(manifest, POSITIVES, empty_count=20)
    empties = [r for r in chosen if int(round(r["seconds"])) not in POSITIVES]
    assert len(empties) == 20


def test_empty_frames_keep_clear_of_the_crossings():
    # A frame one second before a crossing is not "empty" in any useful sense:
    # the crossing is usually already in view.
    manifest = _manifest(range(0, 180))
    chosen = te.select_frames(manifest, POSITIVES, empty_count=20)
    for record in chosen:
        second = int(round(record["seconds"]))
        if second not in POSITIVES:
            assert all(abs(second - p) > 3 for p in POSITIVES)


def test_selection_is_time_ordered():
    chosen = te.select_frames(_manifest(range(0, 180)), POSITIVES, 20)
    assert [r["seconds"] for r in chosen] == sorted(r["seconds"] for r in chosen)


def test_zero_empties_gives_only_positives():
    chosen = te.select_frames(_manifest(range(0, 180)), POSITIVES, 0)
    assert len(chosen) == len(POSITIVES)


# --- mask -> polygon --------------------------------------------------------

def _result_with_crosswalk(mask: np.ndarray) -> TeacherResult:
    seg = np.full(mask.shape, -1, dtype=np.int32)
    seg[mask.astype(bool)] = 7
    segments = [Segment(7, 8, classes.CROSSWALK, 0.9, int(mask.sum()))]
    return TeacherResult(segmentation=seg, segments=segments, inference_ms=1.0)


def test_a_crosswalk_mask_becomes_a_polygon():
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[80:140, 50:250] = 1
    polygons = te.crosswalk_polygons(_result_with_crosswalk(mask), min_area=100)
    assert len(polygons) == 1
    assert cv2.contourArea(polygons[0].astype(np.float32)) == pytest.approx(
        60 * 200, rel=0.1
    )


def test_two_separate_crossings_give_two_polygons():
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[20:60, 20:120] = 1
    mask[140:180, 180:280] = 1
    assert len(te.crosswalk_polygons(_result_with_crosswalk(mask), min_area=100)) == 2


def test_specks_below_min_area_are_dropped():
    mask = np.zeros((200, 300), dtype=np.uint8)
    mask[10:13, 10:14] = 1
    assert te.crosswalk_polygons(_result_with_crosswalk(mask), min_area=400) == []


def test_no_crosswalk_class_gives_no_polygons():
    seg = np.full((50, 50), -1, dtype=np.int32)
    seg[10:20, 10:20] = 3
    result = TeacherResult(
        segmentation=seg,
        segments=[Segment(3, mapillary.SKY_ID, None, 0.9, 100)],
        inference_ms=1.0,
    )
    assert te.crosswalk_polygons(result, min_area=10) == []


# --- label loading ----------------------------------------------------------

def test_labels_are_read_back_in_pixels(tmp_path):
    (tmp_path / "f_00013.0.txt").write_text(
        "0 0.0 0.0 0.5 0.0 0.5 0.5 0.0 0.5\n", encoding="utf-8"
    )
    polygons = te.load_labels(tmp_path, "f_00013.0.jpg", 972, 1920)
    assert len(polygons) == 1
    assert polygons[0].max(axis=0).tolist() == pytest.approx([960.0, 486.0])


def test_a_missing_label_file_is_none_not_empty(tmp_path):
    # None means "not labelled"; [] would mean "labelled as empty", and
    # conflating them would silently turn unreviewed frames into negatives.
    assert te.load_labels(tmp_path, "nope.jpg", 100, 100) is None


def test_an_empty_label_file_is_an_explicit_negative(tmp_path):
    (tmp_path / "e.txt").write_text("", encoding="utf-8")
    assert te.load_labels(tmp_path, "e.jpg", 100, 100) == []


# --- render rules -----------------------------------------------------------

def test_nothing_is_drawn_when_nothing_is_detected():
    image = np.full((100, 100, 3), 77, dtype=np.uint8)
    assert np.array_equal(te.render(image, []), image)


def test_a_detection_is_filled_translucently_not_opaquely():
    image = np.full((200, 200, 3), 20, dtype=np.uint8)
    polygon = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float32)

    out = te.render(image, [polygon])

    inside = out[100, 100]
    assert not np.array_equal(inside, image[100, 100])      # something drawn
    assert not np.array_equal(inside, np.array(te.FILL))    # but see-through
    assert np.array_equal(out[5, 5], image[5, 5])           # outside untouched


def test_render_does_not_mutate_the_input():
    image = np.full((80, 80, 3), 10, dtype=np.uint8)
    before = image.copy()
    te.render(image, [np.array([[10, 10], [70, 10], [70, 70]], dtype=np.float32)])
    assert np.array_equal(image, before)


def test_render_has_no_text_or_boxes():
    # The render rule is fill + outline only. Guard against a box or caption
    # creeping back in: the top-left corner must stay untouched.
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    out = te.render(image, [np.array([[100, 100], [200, 100], [200, 200]], np.float32)])
    assert out[0:40, 0:200].sum() == 0


# --- the download gate ------------------------------------------------------

def test_running_without_confirmation_refuses_and_downloads_nothing():
    assert te.main(["--frames", "data/frames/dashcam"]) == 3


def test_the_declared_download_size_matches_the_documented_one():
    assert te.WEIGHTS_MB == 826


def test_output_goes_to_its_own_folder():
    # Must never overwrite runs/demo or runs/demo_clean.
    assert te.OUT_DIR == Path("runs/teacher_eval")
