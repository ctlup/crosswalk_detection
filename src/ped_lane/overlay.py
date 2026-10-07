"""Frame cropping, mask overlay and the HUD.

The crop maths is pure so it can be tested; the drawing helpers composite all
masks into a single layer and blend once, because the per-polygon blend in the
original skeleton cost one full-frame `addWeighted` per detection.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from . import classes

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def crop_bounds(
    height: int,
    crop_top_fraction: float,
    crop_bottom_fraction: float,
) -> tuple[int, int]:
    """Row range ``[top, bottom)`` to keep after removing sky and car hood.

    Fractions are clamped into [0, 1) and, if together they would leave nothing,
    the whole frame is kept: a misconfigured crop should degrade the view, not
    crash the loop mid-drive.
    """
    if height <= 0:
        raise ValueError("height must be > 0")
    top_fraction = min(max(crop_top_fraction, 0.0), 0.99)
    bottom_fraction = min(max(crop_bottom_fraction, 0.0), 0.99)
    top = int(height * top_fraction)
    bottom = height - int(height * bottom_fraction)
    if bottom - top < 1:
        return 0, height
    return top, bottom


def apply_crop(frame: np.ndarray, top: int, bottom: int) -> np.ndarray:
    """Crop rows, returning a view (no copy)."""
    return frame[top:bottom]


def region_pixel_box(
    width: int,
    height: int,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> tuple[int, int, int, int]:
    """Fractional rectangle -> pixel box ``(x0, y0, x1, y1)``, clamped and ordered.

    Fractions refer to the *original* full frame, so an ignore region stays put
    when the sky/hood crop changes. Reversed corners are accepted.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be > 0")
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    px0 = min(max(int(round(left * width)), 0), width)
    px1 = min(max(int(round(right * width)), 0), width)
    py0 = min(max(int(round(top * height)), 0), height)
    py1 = min(max(int(round(bottom * height)), 0), height)
    return px0, py0, px1, py1


def blank_regions(
    frame: np.ndarray,
    regions: Sequence[tuple[float, float, float, float]],
) -> np.ndarray:
    """Zero out fixed obstructions (wiper mount, sticker, bonnet badge, OSD text).

    Returns `frame` unchanged when there is nothing to blank, so the common case
    costs no copy.
    """
    if not regions:
        return frame
    height, width = frame.shape[:2]
    out = frame.copy()
    for x0, y0, x1, y1 in regions:
        px0, py0, px1, py1 = region_pixel_box(width, height, x0, y0, x1, y1)
        if px1 > px0 and py1 > py0:
            out[py0:py1, px0:px1] = 0
    return out


def redact(
    frame: np.ndarray,
    ignore_regions: Sequence[tuple[float, float, float, float]],
    crop_bottom_fraction: float,
) -> np.ndarray:
    """The privacy-critical processing, and only that.

    Blanks fixed obstructions and removes the bottom strip that carries the
    dashcam's burned-in GPS / date / speed overlay. This is what gets baked
    into every frame written to disk.

    `crop_top_fraction` is deliberately NOT applied here: it is a speed/accuracy
    tuning knob, and baking it in would mean re-extracting and re-labelling
    every frame each time it is retuned. It is applied at training and
    inference time instead, via `top_crop_rows`.
    """
    masked = blank_regions(frame, ignore_regions)
    _, bottom = crop_bounds(frame.shape[0], 0.0, crop_bottom_fraction)
    return masked[:bottom]


def top_crop_rows(source_height: int, crop_top_fraction: float) -> int:
    """Rows to drop from the top of an already-redacted frame.

    `crop_top_fraction` is a fraction of the ORIGINAL frame height, so the
    redacted frame's own (shorter) height must not be used to compute it --
    that would silently shift the crop. Callers working from saved frames read
    the original height out of the extraction manifest.
    """
    if source_height <= 0:
        raise ValueError("source_height must be > 0")
    fraction = min(max(crop_top_fraction, 0.0), 0.99)
    return int(source_height * fraction)


def apply_top_crop(redacted: np.ndarray, rows: int) -> np.ndarray:
    """Drop `rows` from the top, keeping at least one row."""
    if rows <= 0:
        return redacted
    if rows >= redacted.shape[0]:
        return redacted
    return redacted[rows:]


def draw_masks(
    frame: np.ndarray,
    masks: list[np.ndarray],
    class_ids: list[int],
    alpha: float,
    draw_outlines: bool = True,
) -> np.ndarray:
    """Blend per-class coloured masks onto a copy of `frame`.

    `masks` are boolean or 0/1 arrays matching the frame's height and width.
    """
    if not masks:
        return frame.copy()
    alpha = min(max(alpha, 0.0), 1.0)
    layer = frame.copy()
    for mask, class_id in zip(masks, class_ids):
        if mask is None or not mask.any():
            continue
        layer[mask.astype(bool)] = classes.color_for(class_id)
    out = cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0)
    if draw_outlines:
        for mask, class_id in zip(masks, class_ids):
            if mask is None or not mask.any():
                continue
            binary = mask.astype(np.uint8)
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, classes.color_for(class_id), 2)
    return out


def draw_polygons(
    frame: np.ndarray,
    polygons: list[np.ndarray],
    class_ids: list[int],
    alpha: float,
) -> np.ndarray:
    """Same as `draw_masks` but from polygon point lists (Ultralytics `masks.xy`).

    Cheaper than rasterising full masks when there are few, large detections.
    """
    usable = [
        (poly, cid)
        for poly, cid in zip(polygons, class_ids)
        if poly is not None and len(poly) >= 3
    ]
    if not usable:
        return frame.copy()
    alpha = min(max(alpha, 0.0), 1.0)
    layer = frame.copy()
    for poly, class_id in usable:
        cv2.fillPoly(layer, [np.asarray(poly, dtype=np.int32)], classes.color_for(class_id))
    out = cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0)
    for poly, class_id in usable:
        cv2.polylines(
            out, [np.asarray(poly, dtype=np.int32)], True, classes.color_for(class_id), 2
        )
    return out


def draw_hud(
    frame: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (10, 10),
    scale: float = 0.5,
) -> np.ndarray:
    """Draw left-aligned text lines on a translucent panel, in place."""
    if not lines:
        return frame
    thickness = 1
    line_height = int(22 * scale / 0.5)
    sizes = [cv2.getTextSize(t, _FONT, scale, thickness)[0] for t in lines]
    panel_w = max(w for w, _ in sizes) + 16
    panel_h = line_height * len(lines) + 12
    x0, y0 = origin
    x1 = min(x0 + panel_w, frame.shape[1])
    y1 = min(y0 + panel_h, frame.shape[0])
    panel = frame[y0:y1, x0:x1]
    if panel.size:
        frame[y0:y1, x0:x1] = cv2.addWeighted(
            panel, 0.45, np.zeros_like(panel), 0.55, 0
        )
    for i, text in enumerate(lines):
        baseline = y0 + 8 + line_height * (i + 1) - 6
        cv2.putText(
            frame, text, (x0 + 8, baseline), _FONT, scale, (255, 255, 255),
            thickness, cv2.LINE_AA,
        )
    return frame


def draw_legend(frame: np.ndarray, counts: dict[int, int]) -> np.ndarray:
    """Colour key with per-class detection counts, bottom-left, in place."""
    if not counts:
        return frame
    swatch, pad, row = 12, 10, 20
    height = row * len(counts) + pad
    y0 = max(frame.shape[0] - height - pad, 0)
    for i, (class_id, count) in enumerate(sorted(counts.items())):
        y = y0 + pad // 2 + row * i
        cv2.rectangle(
            frame, (pad, y), (pad + swatch, y + swatch),
            classes.color_for(class_id), -1,
        )
        cv2.putText(
            frame, f"{classes.name_for(class_id)} x{count}",
            (pad + swatch + 8, y + swatch), _FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame
