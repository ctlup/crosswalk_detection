"""The project's 4-class taxonomy and the maps from other label spaces into it.

Everything downstream (student weights, panoptic composition, evaluation) uses
these ids, so they are the single source of truth. `crosswalk` and `sidewalk`
are "stuff"; `person` and `vehicle` are "things" that get instance ids.
"""
from __future__ import annotations

from typing import Final, Sequence

CROSSWALK: Final = 0
SIDEWALK: Final = 1
PERSON: Final = 2
VEHICLE: Final = 3

NAMES: Final[tuple[str, ...]] = ("crosswalk", "sidewalk", "person", "vehicle")

#: True for classes that get per-instance ids in the panoptic label map.
IS_THING: Final[tuple[bool, ...]] = (False, False, True, True)

#: Classes currently in scope for training and for the overlay. The project is
#: narrowed to crosswalks for now; the other three stay fully defined here so
#: widening scope again is a config change, not a code change. Override with
#: `active_classes` in config.yaml.
DEFAULT_ACTIVE: Final[tuple[int, ...]] = (CROSSWALK,)

#: BGR, for OpenCV. Chosen to stay distinguishable at low overlay alpha.
COLORS_BGR: Final[dict[int, tuple[int, int, int]]] = {
    CROSSWALK: (60, 220, 255),   # amber
    SIDEWALK: (255, 180, 80),    # blue-cyan
    PERSON: (90, 90, 255),       # red
    VEHICLE: (120, 255, 120),    # green
}

#: COCO class names (what the generic `yolo11n-seg.pt` fallback predicts) that
#: map onto our taxonomy. Lets the plumbing be verified before the student is
#: trained; no COCO class corresponds to crosswalk or sidewalk.
COCO_NAME_TO_CLASS: Final[dict[str, int]] = {
    "person": PERSON,
    "bicycle": VEHICLE,
    "car": VEHICLE,
    "motorcycle": VEHICLE,
    "bus": VEHICLE,
    "truck": VEHICLE,
    "train": VEHICLE,
}


def parse_active(names: "Sequence[str] | None") -> tuple[int, ...]:
    """Class names from config -> sorted class ids, deduplicated.

    An empty or missing list means `DEFAULT_ACTIVE` rather than "nothing":
    a config that silently disabled every class would look like a model that
    detects nothing.
    """
    if not names:
        return DEFAULT_ACTIVE
    ids: set[int] = set()
    for name in names:
        key = str(name).strip().lower()
        if key not in NAMES:
            raise ValueError(
                f"unknown class {name!r} in active_classes; known classes are "
                f"{', '.join(NAMES)}"
            )
        ids.add(NAMES.index(key))
    return tuple(sorted(ids))


def active_names(active: "Sequence[int]") -> tuple[str, ...]:
    """Names for a set of active class ids, in canonical order."""
    return tuple(NAMES[i] for i in sorted(set(active)))


def color_for(class_id: int) -> tuple[int, int, int]:
    """BGR colour for a class id; grey for anything unmapped."""
    return COLORS_BGR.get(class_id, (170, 170, 170))


def name_for(class_id: int) -> str:
    """Canonical name for a class id, or ``cls<id>`` if it is not one of ours."""
    if 0 <= class_id < len(NAMES):
        return NAMES[class_id]
    return f"cls{class_id}"


def remap_from_names(
    predicted_ids: list[int],
    id_to_name: dict[int, str],
) -> list[int | None]:
    """Translate a model's own class ids into project ids.

    Used for the generic COCO fallback model. A name the project does not model
    yields ``None`` so the caller can drop that detection rather than colour it
    as something it is not.
    """
    out: list[int | None] = []
    for pid in predicted_ids:
        name = id_to_name.get(pid, "")
        if name in NAMES:
            out.append(NAMES.index(name))
        else:
            out.append(COCO_NAME_TO_CLASS.get(name))
    return out
