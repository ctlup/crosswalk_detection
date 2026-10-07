import numpy as np
import pytest

from ped_lane import classes, mapillary
from ped_lane.teacher import Segment, TeacherResult


def test_crosswalk_covers_both_vistas_spellings():
    # Vistas labels the painted area and the individual stripes separately;
    # both are a crosswalk to us.
    assert mapillary.to_project_class(8) == classes.CROSSWALK   # Crosswalk - Plain
    assert mapillary.to_project_class(23) == classes.CROSSWALK  # Lane Marking - Crosswalk


def test_sidewalk_includes_pedestrian_area():
    assert mapillary.to_project_class(15) == classes.SIDEWALK  # Sidewalk
    assert mapillary.to_project_class(11) == classes.SIDEWALK  # Pedestrian Area


def test_riders_are_people_and_their_vehicles_are_vehicles():
    for rider in (19, 20, 21, 22):  # Person, Bicyclist, Motorcyclist, Other Rider
        assert mapillary.to_project_class(rider) == classes.PERSON
    for vehicle in (52, 54, 55, 57, 61):  # Bicycle, Bus, Car, Motorcycle, Truck
        assert mapillary.to_project_class(vehicle) == classes.VEHICLE


def test_general_lane_marking_is_not_a_crosswalk():
    # Stop lines and arrows are the hard negative the student must reject.
    assert mapillary.to_project_class(24) is None
    assert 24 in mapillary.HARD_NEGATIVE_IDS


def test_road_and_curb_are_dropped():
    for vistas_id in (2, 7, 9, 10, 13, 14):
        assert mapillary.to_project_class(vistas_id) is None


def test_rig_classes_are_not_training_targets():
    # Ego Vehicle is our own hood and Car Mount the bracket: both are cropped
    # or blanked away, so neither may leak into the vehicle class.
    assert mapillary.to_project_class(mapillary.EGO_VEHICLE_ID) is None
    assert mapillary.to_project_class(mapillary.CAR_MOUNT_ID) is None
    assert mapillary.to_project_class(mapillary.SKY_ID) is None


def test_mapped_and_excluded_sets_do_not_overlap():
    assert not (set(mapillary.VISTAS_TO_PROJECT) & mapillary.EXCLUDED_IDS)


def test_every_mapped_id_targets_a_real_project_class():
    assert set(mapillary.VISTAS_TO_PROJECT.values()) == set(range(len(classes.NAMES)))


def test_label_reference_matches_the_mapping():
    labels = mapillary.vistas_labels()
    if not labels:
        pytest.skip("label reference file not present")
    assert len(labels) == 65
    assert labels[8] == "Crosswalk - Plain"
    assert labels[23] == "Lane Marking - Crosswalk"
    assert labels[64] == "Ego Vehicle"
    for vistas_id in mapillary.VISTAS_TO_PROJECT:
        assert vistas_id in labels


def _fake_result() -> TeacherResult:
    """Two crosswalk stripes, one sidewalk, two cars, and some sky."""
    seg = np.full((10, 10), -1, dtype=np.int32)
    seg[0:3, :] = 1      # sky
    seg[5:6, 0:4] = 2    # crosswalk stripe
    seg[5:6, 6:10] = 3   # crosswalk stripe
    seg[7:10, 0:3] = 4   # sidewalk
    seg[4:5, 0:2] = 5    # car
    seg[4:5, 8:10] = 6   # car
    segments = [
        Segment(1, mapillary.SKY_ID, None, 0.99, 30),
        Segment(2, 23, classes.CROSSWALK, 0.9, 4),
        Segment(3, 23, classes.CROSSWALK, 0.8, 4),
        Segment(4, 15, classes.SIDEWALK, 0.95, 9),
        Segment(5, 55, classes.VEHICLE, 0.7, 2),
        Segment(6, 55, classes.VEHICLE, 0.6, 2),
    ]
    return TeacherResult(segmentation=seg, segments=segments, inference_ms=1.0)


def test_class_mask_unions_every_segment_of_that_class():
    result = _fake_result()
    mask = result.class_mask(classes.CROSSWALK)
    assert mask.sum() == 8
    assert mask[5, 0] and mask[5, 9]
    assert not mask[5, 5]


def test_class_mask_is_empty_for_an_absent_class():
    assert not _fake_result().class_mask(classes.PERSON).any()


def test_vistas_mask_accepts_one_id_or_many():
    result = _fake_result()
    assert result.vistas_mask(mapillary.SKY_ID).sum() == 30
    assert result.vistas_mask([23, 15]).sum() == 17


def test_instances_keeps_things_separate_largest_first():
    result = _fake_result()
    crosswalks = result.instances(classes.CROSSWALK)
    assert len(crosswalks) == 2  # stripes stay separate segments
    areas = [s.area for s in result.instances(classes.VEHICLE)]
    assert areas == sorted(areas, reverse=True)


def test_counts_ignores_unmapped_segments():
    assert _fake_result().counts() == {
        classes.CROSSWALK: 2,
        classes.SIDEWALK: 1,
        classes.VEHICLE: 2,
    }


def test_segment_exposes_its_vistas_name():
    if not mapillary.vistas_labels():
        pytest.skip("label reference file not present")
    assert Segment(1, 8, classes.CROSSWALK, 1.0, 1).vistas_name == "Crosswalk - Plain"
