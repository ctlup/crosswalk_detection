"""Typed view over config.yaml.

`load_config` does file IO; `Config.from_dict` is pure so it can be tested and
so scripts can build a config without a yaml file on disk.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import classes


def parse_ignore_regions(
    raw: Any,
) -> tuple[tuple[float, float, float, float], ...]:
    """Read `dashcam.ignore_regions` into (x0, y0, x1, y1) fraction tuples.

    Accepts the mapping form from config.yaml
    (``{x0: .., y0: .., x1: .., y1: ..}``) and a bare 4-element sequence.
    A malformed entry raises rather than being silently ignored: a region that
    quietly does nothing would look like a model failure, not a config typo.
    """
    if not raw:
        return ()
    regions: list[tuple[float, float, float, float]] = []
    for index, entry in enumerate(raw):
        try:
            if isinstance(entry, dict):
                box = (entry["x0"], entry["y0"], entry["x1"], entry["y1"])
            else:
                x0, y0, x1, y1 = entry
                box = (x0, y0, x1, y1)
            regions.append(tuple(float(v) for v in box))  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"dashcam.ignore_regions[{index}] must have x0, y0, x1, y1 "
                f"as fractions of the frame; got {entry!r}"
            ) from exc
    return tuple(regions)


@dataclass(frozen=True)
class ModelConfig:
    weights: Path = Path("models/crosswalk-seg.pt")
    imgsz: int = 640
    conf: float = 0.35
    device: str = "auto"


@dataclass(frozen=True)
class StreamConfig:
    reconnect_seconds: float = 3.0
    max_fps: float = 15.0


@dataclass(frozen=True)
class DashcamConfig:
    crop_bottom_fraction: float = 0.18
    crop_top_fraction: float = 0.0
    #: Fractions of the original full frame, blanked before the crop.
    ignore_regions: tuple[tuple[float, float, float, float], ...] = ()


@dataclass(frozen=True)
class OutputConfig:
    show_window: bool = True
    save_video: Path | None = None
    overlay_alpha: float = 0.4


def parse_exclude_ranges(raw: Any) -> tuple[tuple[float, float], ...]:
    """Read `extraction.exclude_ranges` into sorted (start, end) second pairs.

    A malformed entry raises: a range that silently covers nothing would let
    content we meant to drop reach disk.
    """
    if not raw:
        return ()
    ranges: list[tuple[float, float]] = []
    for index, entry in enumerate(raw):
        try:
            start, end = (float(entry[0]), float(entry[1]))
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            raise ValueError(
                f"extraction.exclude_ranges[{index}] must be [start_seconds, "
                f"end_seconds]; got {entry!r}"
            ) from exc
        if end < start:
            start, end = end, start
        ranges.append((start, end))
    return tuple(sorted(ranges))


@dataclass(frozen=True)
class ExtractionConfig:
    """Frame extraction settings.

    `exclude_ranges` drops time spans outright. It exists for content that the
    geometric redaction cannot remove -- a vehicle occupant's hand entering
    frame above the ignore region, for instance. Masking a region large enough
    to cover that would throw away most of the right-hand roadway, so the
    affected seconds are dropped instead.
    """

    exclude_ranges: tuple[tuple[float, float], ...] = ()

    def is_excluded(self, seconds: float) -> bool:
        return any(start <= seconds <= end for start, end in self.exclude_ranges)


@dataclass(frozen=True)
class TeacherConfig:
    """Mask2Former teacher. Off by default: the released weights are
    CC BY-NC 4.0 (non-commercial) and are not cleared for this project yet.
    See LICENSING.md. Nothing is downloaded while `enabled` is False."""

    enabled: bool = False
    checkpoint: str = "facebook/mask2former-swin-large-mapillary-vistas-panoptic"
    device: str = "auto"
    fp16: bool = True


@dataclass(frozen=True)
class TemporalConfig:
    smoothing_frames: int = 3


@dataclass(frozen=True)
class Config:
    source: str = "samples/dashcam.mp4"
    model: ModelConfig = field(default_factory=ModelConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    dashcam: DashcamConfig = field(default_factory=DashcamConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    #: Class ids in scope for training and the overlay.
    active_classes: tuple[int, ...] = classes.DEFAULT_ACTIVE

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "Config":
        raw = raw or {}
        model = raw.get("model") or {}
        stream = raw.get("stream") or {}
        dashcam = raw.get("dashcam") or {}
        output = raw.get("output") or {}
        temporal = raw.get("temporal") or {}
        teacher = raw.get("teacher") or {}
        extraction = raw.get("extraction") or {}
        save = output.get("save_video")
        return cls(
            source=str(raw.get("source", cls.source)),
            model=ModelConfig(
                weights=Path(model.get("weights", "models/crosswalk-seg.pt")),
                imgsz=int(model.get("imgsz", 640)),
                conf=float(model.get("conf", 0.35)),
                device=str(model.get("device", "auto")),
            ),
            stream=StreamConfig(
                reconnect_seconds=float(stream.get("reconnect_seconds", 3.0)),
                max_fps=float(stream.get("max_fps", 15.0)),
            ),
            dashcam=DashcamConfig(
                crop_bottom_fraction=float(dashcam.get("crop_bottom_fraction", 0.18)),
                crop_top_fraction=float(dashcam.get("crop_top_fraction", 0.0)),
                ignore_regions=parse_ignore_regions(dashcam.get("ignore_regions")),
            ),
            output=OutputConfig(
                show_window=bool(output.get("show_window", True)),
                save_video=None if save in (None, "", "null") else Path(str(save)),
                overlay_alpha=float(output.get("overlay_alpha", 0.4)),
            ),
            temporal=TemporalConfig(
                smoothing_frames=int(temporal.get("smoothing_frames", 3)),
            ),
            teacher=TeacherConfig(
                enabled=bool(teacher.get("enabled", False)),
                checkpoint=str(
                    teacher.get(
                        "checkpoint",
                        "facebook/mask2former-swin-large-mapillary-vistas-panoptic",
                    )
                ),
                device=str(teacher.get("device", "auto")),
                fp16=bool(teacher.get("fp16", True)),
            ),
            extraction=ExtractionConfig(
                exclude_ranges=parse_exclude_ranges(extraction.get("exclude_ranges")),
            ),
            active_classes=classes.parse_active(raw.get("active_classes")),
        )


def load_config(path: str | Path) -> Config:
    """Read a yaml config file into a `Config`."""
    with open(path, encoding="utf-8") as f:
        return Config.from_dict(yaml.safe_load(f))


def resolve_source(cli_source: str | None, cfg: Config) -> str | int:
    """Pick the frame source: --source, then $CAMERA_URL, then the config file.

    A pure-digit value is a local camera index, so it is returned as an int.
    """
    chosen = cli_source or os.environ.get("CAMERA_URL") or cfg.source
    text = str(chosen).strip()
    if text.isdigit():
        return int(text)
    return text
