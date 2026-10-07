"""Ultralytics segmentation wrapper that speaks the project's class ids.

Keeps the rest of the code free of Ultralytics result objects, and papers over
the one case that matters before training is done: when the student weights do
not exist yet, fall back to the generic COCO `yolo11n-seg` and remap only the
classes that have a counterpart here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from . import classes
from .devices import describe_device, resolve_device

FALLBACK_WEIGHTS = "models/yolo11n-seg.pt"


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    polygon: np.ndarray  # (N, 2) float32, in frame pixel coordinates
    box: tuple[float, float, float, float]  # xyxy


@dataclass(frozen=True)
class FrameResult:
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0

    def counts(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for det in self.detections:
            out[det.class_id] = out.get(det.class_id, 0) + 1
        return out

    def restricted_to(self, active: "Iterable[int]") -> "FrameResult":
        """Drop detections outside the classes currently in scope.

        Filtering here rather than in the model keeps the measured inference
        time honest: narrowing the scope does not make the network faster.
        """
        wanted = set(active)
        return FrameResult(
            detections=[d for d in self.detections if d.class_id in wanted],
            inference_ms=self.inference_ms,
        )


class SegDetector:
    """Instance-segmentation inference on one frame at a time."""

    def __init__(
        self,
        weights: str | Path,
        device: str = "auto",
        imgsz: int = 640,
        conf: float = 0.35,
    ):
        from ultralytics import YOLO  # imported late: heavy, and not needed by tests

        self.requested_weights = Path(weights)
        self.is_generic = not self.requested_weights.exists()
        self.weights = FALLBACK_WEIGHTS if self.is_generic else str(self.requested_weights)
        self.device = resolve_device(device)
        self.imgsz = imgsz
        self.conf = conf
        self.model = YOLO(self.weights)
        self.model.to(self.device)
        self._id_to_name: dict[int, str] = dict(self.model.names or {})

    @property
    def label(self) -> str:
        """Short description for the HUD and benchmark tables."""
        return f"{Path(self.weights).name} @ {describe_device(self.device)}"

    def warmup(self, height: int = 384, width: int = 640, runs: int = 2) -> None:
        """Run inference on blank frames so the first real frame is not an outlier.

        CUDA kernel autotuning and lazy module loading make the first call several
        times slower, which would otherwise poison the measured p95.
        """
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(max(runs, 0)):
            self.predict(blank)

    def predict(self, frame: np.ndarray) -> FrameResult:
        started = time.perf_counter()
        result = self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return FrameResult(
            detections=self._convert(result), inference_ms=elapsed_ms
        )

    def _convert(self, result) -> list[Detection]:
        if result.masks is None or result.boxes is None:
            return []
        raw_ids = [int(c) for c in result.boxes.cls.tolist()]
        confidences = [float(c) for c in result.boxes.conf.tolist()]
        boxes = result.boxes.xyxy.tolist()
        polygons = list(result.masks.xy)
        mapped = classes.remap_from_names(raw_ids, self._id_to_name)

        detections: list[Detection] = []
        for class_id, confidence, box, polygon in zip(
            mapped, confidences, boxes, polygons
        ):
            if class_id is None:  # a class this project does not model
                continue
            if polygon is None or len(polygon) < 3:
                continue
            detections.append(
                Detection(
                    class_id=class_id,
                    confidence=confidence,
                    polygon=np.asarray(polygon, dtype=np.float32),
                    box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                )
            )
        return detections
