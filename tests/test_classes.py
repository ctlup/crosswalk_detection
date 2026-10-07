from ped_lane import classes
from ped_lane.classes import remap_from_names


def test_taxonomy_is_self_consistent():
    assert len(classes.NAMES) == len(classes.IS_THING)
    assert set(classes.COLORS_BGR) == set(range(len(classes.NAMES)))
    assert classes.NAMES.index("crosswalk") == classes.CROSSWALK
    assert classes.NAMES.index("vehicle") == classes.VEHICLE


def test_stuff_and_things_are_split_as_expected():
    assert classes.IS_THING[classes.PERSON]
    assert classes.IS_THING[classes.VEHICLE]
    assert not classes.IS_THING[classes.CROSSWALK]
    assert not classes.IS_THING[classes.SIDEWALK]


def test_coco_vehicles_collapse_into_one_class():
    coco = {0: "person", 2: "car", 5: "bus", 7: "truck"}
    assert remap_from_names([0, 2, 5, 7], coco) == [
        classes.PERSON,
        classes.VEHICLE,
        classes.VEHICLE,
        classes.VEHICLE,
    ]


def test_unmodelled_coco_classes_become_none():
    # None means "drop it" rather than colour it as something it is not.
    coco = {24: "backpack", 11: "stop sign", 0: "person"}
    assert remap_from_names([24, 11, 0], coco) == [None, None, classes.PERSON]


def test_student_class_names_pass_through():
    student = dict(enumerate(classes.NAMES))
    assert remap_from_names(list(range(4)), student) == [0, 1, 2, 3]


def test_unknown_id_becomes_none():
    assert remap_from_names([99], {0: "person"}) == [None]


def test_name_and_colour_lookups_are_safe():
    assert classes.name_for(classes.SIDEWALK) == "sidewalk"
    assert classes.name_for(42) == "cls42"
    assert classes.color_for(42) == (170, 170, 170)
