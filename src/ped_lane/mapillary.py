"""Mapillary Vistas (65 classes) -> this project's 4 classes.

The teacher predicts the full Vistas label space. Everything the student needs
is folded into `classes.NAMES`; everything else becomes `None` and is dropped.
A few Vistas classes are not training targets but are still useful, because
they say where the car and the camera mount are in frame -- those are kept
separately in `GEOMETRY_CLASSES` and drive the dashcam crop tuning.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Final

from . import classes

LABELS_PATH: Final = (
    Path(__file__).resolve().parents[2] / "data" / "reference" / "mapillary_vistas_labels.json"
)

# --- training targets -------------------------------------------------------

CROSSWALK_IDS: Final[frozenset[int]] = frozenset({
    8,   # Crosswalk - Plain      (the painted area as a region)
    23,  # Lane Marking - Crosswalk (the individual zebra stripes)
})

SIDEWALK_IDS: Final[frozenset[int]] = frozenset({
    15,  # Sidewalk
    11,  # Pedestrian Area
})

# Riders are people. Vistas separates a rider from the vehicle they are on, and
# so do we: the rider lands in `person`, the bicycle/motorcycle in `vehicle`.
PERSON_IDS: Final[frozenset[int]] = frozenset({
    19,  # Person
    20,  # Bicyclist
    21,  # Motorcyclist
    22,  # Other Rider
})

VEHICLE_IDS: Final[frozenset[int]] = frozenset({
    52,  # Bicycle
    54,  # Bus
    55,  # Car
    56,  # Caravan
    57,  # Motorcycle
    58,  # On Rails
    59,  # Other Vehicle
    60,  # Trailer
    61,  # Truck
    62,  # Wheeled Slow
})

#: Deliberately NOT mapped, and why:
#:   2  Curb      - the kerb edge, not walkable surface; would fatten sidewalk masks
#:   9  Curb Cut  - the ramp at a crossing; sits between sidewalk and crosswalk
#:   7  Bike Lane - road marking, not a pedestrian surface
#:   24 Lane Marking - General - the hard negative we want the student to reject
#:                     (stop lines and arrows look like zebra stripes at a distance)
#:   13 Road, 10 Parking, 14 Service Lane - drivable surface, out of scope
EXCLUDED_IDS: Final[frozenset[int]] = frozenset({2, 7, 9, 10, 13, 14, 24})

#: The hard negatives worth keeping track of when building the dataset: road
#: paint that is not a crosswalk is exactly what a crosswalk detector overfits to.
HARD_NEGATIVE_IDS: Final[frozenset[int]] = frozenset({7, 24})

# --- not training targets, but useful geometry ------------------------------

SKY_ID: Final = 27
CAR_MOUNT_ID: Final = 63
EGO_VEHICLE_ID: Final = 64

#: Classes that describe the rig rather than the scene. `Ego Vehicle` is the
#: car hood, `Car Mount` the windscreen bracket -- exactly what
#: `dashcam.crop_bottom_fraction` and `dashcam.ignore_regions` exist to remove,
#: so the teacher can measure whether those settings are right. `Sky` bounds
#: `crop_top_fraction` the same way.
GEOMETRY_CLASSES: Final[dict[str, int]] = {
    "sky": SKY_ID,
    "car_mount": CAR_MOUNT_ID,
    "ego_vehicle": EGO_VEHICLE_ID,
}


def _build_map() -> dict[int, int]:
    mapping: dict[int, int] = {}
    for vistas_id in CROSSWALK_IDS:
        mapping[vistas_id] = classes.CROSSWALK
    for vistas_id in SIDEWALK_IDS:
        mapping[vistas_id] = classes.SIDEWALK
    for vistas_id in PERSON_IDS:
        mapping[vistas_id] = classes.PERSON
    for vistas_id in VEHICLE_IDS:
        mapping[vistas_id] = classes.VEHICLE
    return mapping


#: Vistas class id -> project class id. Absent key means "drop it".
VISTAS_TO_PROJECT: Final[dict[int, int]] = _build_map()


def to_project_class(vistas_id: int) -> int | None:
    """Project class for a Vistas class id, or None if we do not model it."""
    return VISTAS_TO_PROJECT.get(int(vistas_id))


@lru_cache(maxsize=1)
def vistas_labels() -> dict[int, str]:
    """Vistas id -> name, read from the checkpoint's own config (cached locally).

    Falls back to an empty map if the reference file is missing, so importing
    this module never depends on having downloaded anything.
    """
    try:
        raw = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {int(k): v for k, v in raw.items()}


def vistas_name(vistas_id: int) -> str:
    return vistas_labels().get(int(vistas_id), f"vistas{vistas_id}")
