"""Mask2Former (Swin-L, Mapillary Vistas) panoptic teacher.

Accuracy reference and pseudo-labeller, not a real-time model. It runs on the
*full* frame on purpose: the dashcam crop is a speed optimisation for the
student, and the teacher is what we use to decide whether that crop throws away
anything useful, so it must see the rows the student never will.

LICENCE: the released Mask2Former weights are CC BY-NC 4.0 (non-commercial).
See README.md -- this file does not download anything until it is constructed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from . import mapillary
from .devices import describe_device, resolve_device

CHECKPOINT = "facebook/mask2former-swin-large-mapillary-vistas-panoptic"


@dataclass(frozen=True)
class Segment:
    """One panoptic segment: a connected region with a Vistas class."""

    segment_id: int
    vistas_id: int
    project_class: int | None
    score: float
    area: int

    @property
    def vistas_name(self) -> str:
        return mapillary.vistas_name(self.vistas_id)


@dataclass(frozen=True)
class TeacherResult:
    """Panoptic output for one frame, at the frame's own resolution."""

    #: (H, W) int32; each pixel holds a `Segment.segment_id`, or -1 for nothing.
    segmentation: np.ndarray
    segments: list[Segment] = field(default_factory=list)
    inference_ms: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.segmentation.shape[:2]

    def segment_mask(self, segment_id: int) -> np.ndarray:
        return self.segmentation == segment_id

    def vistas_mask(self, vistas_ids: int | Iterable[int]) -> np.ndarray:
        """Boolean mask of every pixel belonging to the given Vistas class(es)."""
        wanted = {vistas_ids} if isinstance(vistas_ids, int) else set(vistas_ids)
        ids = [s.segment_id for s in self.segments if s.vistas_id in wanted]
        return self._union(ids)

    def class_mask(self, project_class: int) -> np.ndarray:
        """Boolean mask for one of the project's 4 classes."""
        ids = [s.segment_id for s in self.segments if s.project_class == project_class]
        return self._union(ids)

    def instances(self, project_class: int) -> list[Segment]:
        """Segments of a class, largest first -- the `things` for the panoptic map."""
        found = [s for s in self.segments if s.project_class == project_class]
        return sorted(found, key=lambda s: s.area, reverse=True)

    def counts(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for segment in self.segments:
            if segment.project_class is not None:
                out[segment.project_class] = out.get(segment.project_class, 0) + 1
        return out

    def _union(self, segment_ids: list[int]) -> np.ndarray:
        mask = np.zeros(self.shape, dtype=bool)
        for segment_id in segment_ids:
            mask |= self.segmentation == segment_id
        return mask


class Mask2FormerTeacher:
    """Lazy wrapper around the Hugging Face checkpoint.

    Constructing this downloads ~826 MB on first use (cached in the HF cache
    afterwards); nothing is downloaded at import time.
    """

    def __init__(
        self,
        checkpoint: str = CHECKPOINT,
        device: str = "auto",
        fp16: bool = True,
        local_files_only: bool = False,
    ):
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.checkpoint = checkpoint
        self.device = resolve_device(device)
        # fp16 roughly halves the time and the VRAM on the A4000; it is a
        # reference model, not a deployment target, so the precision loss on
        # mask boundaries is acceptable. Never on CPU, where it is slower.
        self.fp16 = bool(fp16) and self.device.startswith("cuda")

        self.processor = AutoImageProcessor.from_pretrained(
            checkpoint, local_files_only=local_files_only
        )
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpoint, local_files_only=local_files_only
        )
        self.model.to(self.device)
        if self.fp16:
            self.model.half()
        self.model.eval()

    @property
    def label(self) -> str:
        precision = "fp16" if self.fp16 else "fp32"
        return f"mask2former-swin-l-vistas ({precision}) @ {describe_device(self.device)}"

    def warmup(self, height: int = 1080, width: int = 1920, runs: int = 1) -> None:
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        for _ in range(max(runs, 0)):
            self.predict(blank)

    def predict(self, frame_bgr: np.ndarray, threshold: float = 0.5) -> TeacherResult:
        """Panoptic segmentation of one BGR frame, at the frame's own size."""
        import torch

        height, width = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB, no copy of the data itself

        started = time.perf_counter()
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        if self.fp16:
            inputs = {
                k: (v.half() if v.dtype == torch.float32 else v) for k, v in inputs.items()
            }

        with torch.inference_mode():
            outputs = self.model(**inputs)

        processed = self.processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[(height, width)], threshold=threshold
        )[0]
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        return self._to_result(processed, elapsed_ms)

    @staticmethod
    def _to_result(processed: dict, elapsed_ms: float) -> TeacherResult:
        segmentation = processed["segmentation"].cpu().numpy().astype(np.int32)
        segments: list[Segment] = []
        for info in processed.get("segments_info", []):
            vistas_id = int(info["label_id"])
            segment_id = int(info["id"])
            segments.append(
                Segment(
                    segment_id=segment_id,
                    vistas_id=vistas_id,
                    project_class=mapillary.to_project_class(vistas_id),
                    score=float(info.get("score", 1.0)),
                    area=int(np.count_nonzero(segmentation == segment_id)),
                )
            )
        return TeacherResult(
            segmentation=segmentation, segments=segments, inference_ms=elapsed_ms
        )
