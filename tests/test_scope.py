"""Class scoping, and the guard rails around the gated teacher.

The teacher assertions here are deliberately blunt: the weights are CC BY-NC
4.0 and not cleared for this project, so "off unless explicitly turned on" is a
licensing constraint, not a preference, and a regression in it should fail the
build rather than quietly start an 826 MB download.
"""
import numpy as np
import pytest

from ped_lane import classes
from ped_lane.classes import active_names, parse_active
from ped_lane.config import Config, load_config
from ped_lane.detector import Detection, FrameResult


# --- active class scope -----------------------------------------------------

def test_default_scope_is_crosswalk_only():
    assert classes.DEFAULT_ACTIVE == (classes.CROSSWALK,)
    assert parse_active(None) == (classes.CROSSWALK,)


def test_empty_list_means_the_default_not_nothing():
    # A config that disabled every class would look like a broken model.
    assert parse_active([]) == classes.DEFAULT_ACTIVE


def test_names_are_parsed_deduplicated_and_sorted():
    assert parse_active(["vehicle", "crosswalk", "vehicle"]) == (
        classes.CROSSWALK,
        classes.VEHICLE,
    )


def test_parsing_is_case_and_whitespace_insensitive():
    assert parse_active([" Crosswalk ", "SIDEWALK"]) == (
        classes.CROSSWALK,
        classes.SIDEWALK,
    )


def test_unknown_class_name_is_rejected_with_the_valid_options():
    with pytest.raises(ValueError, match="unknown class"):
        parse_active(["zebra"])


def test_active_names_round_trips():
    assert active_names(parse_active(["person", "crosswalk"])) == ("crosswalk", "person")


def test_the_other_three_classes_still_exist():
    # Narrowed scope must not delete the taxonomy; widening is a config change.
    assert classes.NAMES == ("crosswalk", "sidewalk", "person", "vehicle")
    for name in ("sidewalk", "person", "vehicle"):
        assert name in classes.NAMES


# --- the scope filter -------------------------------------------------------

def _detection(class_id: int) -> Detection:
    polygon = np.array([[0, 0], [1, 0], [1, 1]], dtype=np.float32)
    return Detection(class_id=class_id, confidence=0.9, polygon=polygon, box=(0, 0, 1, 1))


def test_restricted_to_drops_out_of_scope_detections():
    result = FrameResult(
        detections=[_detection(c) for c in (classes.CROSSWALK, classes.PERSON, classes.VEHICLE)],
        inference_ms=12.0,
    )

    narrowed = result.restricted_to([classes.CROSSWALK])

    assert [d.class_id for d in narrowed.detections] == [classes.CROSSWALK]
    assert narrowed.counts() == {classes.CROSSWALK: 1}


def test_restricted_to_keeps_the_measured_time_honest():
    # Narrowing the scope does not make the network faster, so the reported
    # inference time must not change.
    result = FrameResult(detections=[_detection(classes.PERSON)], inference_ms=15.5)
    assert result.restricted_to([classes.CROSSWALK]).inference_ms == 15.5


def test_restricted_to_an_empty_scope_yields_nothing():
    result = FrameResult(detections=[_detection(classes.PERSON)], inference_ms=1.0)
    assert result.restricted_to([]).detections == []


# --- teacher gating ---------------------------------------------------------

def test_teacher_is_off_by_default():
    assert Config.from_dict({}).teacher.enabled is False


def test_teacher_is_off_when_the_section_is_absent_or_empty():
    assert Config.from_dict({"teacher": None}).teacher.enabled is False
    assert Config.from_dict({"teacher": {}}).teacher.enabled is False


def test_shipped_config_keeps_the_teacher_disabled():
    # The weights are CC BY-NC 4.0 and not cleared; see LICENSING.md.
    cfg = load_config("config.yaml")
    assert cfg.teacher.enabled is False


def test_shipped_config_is_crosswalk_only():
    assert load_config("config.yaml").active_classes == (classes.CROSSWALK,)


def test_enabling_the_teacher_takes_an_explicit_true():
    assert Config.from_dict({"teacher": {"enabled": True}}).teacher.enabled is True


def test_importing_the_teacher_module_downloads_nothing():
    # Importing must stay free of network and weight loading; only constructing
    # Mask2FormerTeacher may download.
    import ped_lane.teacher as teacher

    assert teacher.CHECKPOINT.startswith("facebook/mask2former")
    assert hasattr(teacher, "Mask2FormerTeacher")


# --- extraction exclusion ---------------------------------------------------

def test_exclude_ranges_default_to_none():
    assert Config.from_dict({}).extraction.exclude_ranges == ()
    assert not Config.from_dict({}).extraction.is_excluded(5.0)


def test_exclude_ranges_are_inclusive_and_sorted():
    cfg = Config.from_dict({"extraction": {"exclude_ranges": [[27.5, 30.5], [6.5, 9.5]]}})
    assert cfg.extraction.exclude_ranges == ((6.5, 9.5), (27.5, 30.5))
    assert cfg.extraction.is_excluded(6.5)
    assert cfg.extraction.is_excluded(9.5)
    assert cfg.extraction.is_excluded(29.0)
    assert not cfg.extraction.is_excluded(6.4)
    assert not cfg.extraction.is_excluded(15.0)


def test_reversed_range_is_accepted():
    cfg = Config.from_dict({"extraction": {"exclude_ranges": [[9.5, 6.5]]}})
    assert cfg.extraction.is_excluded(8.0)


def test_malformed_exclude_range_is_rejected():
    import pytest as _pytest
    with _pytest.raises(ValueError, match=r"exclude_ranges\[0\]"):
        Config.from_dict({"extraction": {"exclude_ranges": [[1.0]]}})


def test_shipped_config_excludes_the_hand_seconds():
    # A vehicle occupant's hand is visible above the ignore_region mask at
    # ~8s and ~29s; see LICENSING.md.
    cfg = load_config("config.yaml")
    for second in (8.0, 29.0):
        assert cfg.extraction.is_excluded(second)
