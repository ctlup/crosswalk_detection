import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

_SPEC = importlib.util.spec_from_file_location(
    "label_crosswalk", Path(__file__).resolve().parents[1] / "scripts" / "label_crosswalk.py"
)
label = importlib.util.module_from_spec(_SPEC)
# @dataclass resolves annotations via sys.modules, so register before exec.
sys.modules["label_crosswalk"] = label
_SPEC.loader.exec_module(label)

from ped_lane import classes  # noqa: E402


def test_yolo_seg_line_is_normalised():
    polygon = np.array([[0, 0], [960, 0], [960, 486], [0, 486]], dtype=np.float32)
    line = label.to_yolo_seg(polygon, 1920, 972, class_id=classes.CROSSWALK)
    assert line.split()[0] == "0"
    assert line == "0 0.000000 0.000000 0.500000 0.000000 0.500000 0.500000 0.000000 0.500000"


def test_coordinates_outside_the_frame_are_clipped():
    polygon = np.array([[-50, -50], [2000, 10], [1000, 1200]], dtype=np.float32)
    values = [float(v) for v in label.to_yolo_seg(polygon, 1920, 972).split()[1:]]
    assert all(0.0 <= v <= 1.0 for v in values)


def test_round_trip_through_the_label_format():
    polygon = np.array([[10, 20], [300, 25], [310, 180], [15, 170]], dtype=np.float32)
    line = label.to_yolo_seg(polygon, 1920, 972)
    class_id, restored = label.from_yolo_seg(line, 1920, 972)
    assert class_id == classes.CROSSWALK
    assert np.allclose(restored, polygon, atol=0.01)


def test_a_polygon_needs_three_points():
    with pytest.raises(ValueError):
        label.to_yolo_seg(np.array([[0, 0], [1, 1]], dtype=np.float32), 100, 100)


def test_malformed_label_lines_are_rejected():
    for bad in ("", "0 0.1 0.2", "0 0.1 0.2 0.3 0.4 0.5"):
        with pytest.raises(ValueError):
            label.from_yolo_seg(bad, 100, 100)


def test_label_file_sits_beside_the_image_by_stem():
    path = label.label_path_for(Path("data/frames/dashcam/dashcam_f000420_t00014.01s.jpg"),
                                Path("data/labels/dashcam"))
    assert path == Path("data/labels/dashcam/dashcam_f000420_t00014.01s.txt")


def test_a_reviewed_frame_with_no_polygon_is_an_explicit_negative():
    # The distinction matters: "reviewed, nothing here" is training signal,
    # "not looked at yet" is not.
    assert label.FrameLabels(reviewed=True).is_negative
    assert not label.FrameLabels(reviewed=False).is_negative
    assert not label.FrameLabels(
        polygons=[np.zeros((3, 2), np.float32)], reviewed=True
    ).is_negative


def test_the_tool_labels_the_single_active_class():
    assert label.CROSSWALK_CLASS == classes.CROSSWALK == 0
