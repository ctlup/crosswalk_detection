"""Classical zebra-stripe detector. No model weights, no training, OpenCV only.

Two jobs:

1. **Baseline.** Whatever we train later has to beat stripe-counting, or it is
   not earning its licence cost and its GPU.
2. **Candidate proposer.** Run it over new footage to rank timestamps by how
   likely they are to contain a crossing, so labelling starts with the frames
   that matter instead of scrubbing through hours of empty road.

The geometry it exploits: a zebra crossing is a ladder of bars lying across
the carriageway, stacked one behind another in depth. In a forward-facing
camera each bar therefore appears *tangential* to the road's vanishing point
-- its long axis is perpendicular to the line joining it to the VP -- and
consecutive bars sit at increasing radius from the VP with the spacing
stretching as they come closer, exactly as perspective dictates.

The test that separates a crossing from a dashed lane line -- the marking most
likely to be confused with it -- is a *group* property, not a per-bar one.
Perspective does not preserve right angles, so a single bar left or right of
the vanishing point does not look perpendicular to anything useful. But take
the line through a group's bar centres and compare it with the bars' own axes:

  * a **crossing** has its bars lying ACROSS that line (a ladder, rungs square
    to the rails);
  * a **dashed lane line** has its dashes lying ALONG it (beads on a string).

That holds whatever the viewing angle. Stop lines pass the orientation test but
there is only ever one of them, so the "at least four, spaced consistently with
perspective" rule rejects them.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass(frozen=True)
class StripeParams:
    """Tunables. Defaults are set for 1920-wide dashcam frames."""

    #: Width of the top-hat kernel, as a fraction of frame width. Should be a
    #: little wider than the widest stripe we expect to see.
    tophat_width_fraction: float = 0.030
    #: Reject blobs smaller than this fraction of the ROI area (noise, chippings).
    min_area_fraction: float = 0.00012
    #: Reject blobs larger than this (sunlit walls, sky through a gap).
    max_area_fraction: float = 0.06
    #: A stripe is long and thin; reject anything squarer than this.
    min_elongation: float = 1.8
    #: Bars of one crossing are near-parallel in the image; cluster axes to
    #: within this many degrees.
    axis_cluster_degrees: float = 22.0
    #: Max |cos| between the bars' mean axis and the line through their
    #: centres. Low means a ladder (crossing); high means beads on a string
    #: (a dashed lane line), which is rejected.
    max_axis_vs_centreline_cos: float = 0.55
    #: Minimum bars in a group before it counts as a crossing. Three would
    #: admit a stop line plus noise; four is the smallest real zebra.
    min_stripes: int = 4
    #: Consecutive bars must be separated in radius by at least this fraction
    #: of frame height, so one blob split in two does not count twice.
    min_radial_gap_fraction: float = 0.008
    #: ...and at most this, so unrelated markings do not chain together.
    max_radial_gap_fraction: float = 0.30
    #: Perspective stretches the spacing as bars approach; consecutive gap
    #: ratios must stay inside this band.
    gap_ratio_bounds: tuple[float, float] = (0.35, 3.2)
    #: Regularity: max coefficient of variation of those gap ratios.
    max_gap_ratio_cv: float = 0.75
    #: Carriageway trapezoid, converging on the vanishing point. Roadside
    #: railings and fences are near-perfect ladders and are the dominant false
    #: positive, so restricting to the carriageway ought to help. Measured on
    #: samples/dashcam.mp4 it did not: it removed enough real bars to drop
    #: groups below `min_stripes` and recall went to zero. OFF until there is
    #: enough footage to tune it against something other than 9 frames.
    use_road_trapezoid: bool = False
    road_base_left_fraction: float = 0.02
    road_base_right_fraction: float = 0.98
    road_apex_half_width_fraction: float = 0.08
    #: Local contrast normalisation before thresholding. OFF by default:
    #: on a road band that is mostly dark asphalt it amplifies surface noise
    #: and drags the Otsu threshold up until the actual bars fall below it.
    #: Measured on samples/dashcam.mp4, it took the bar count at t=100s from
    #: 20 down to 4. Kept as an option for genuinely low-contrast night work.
    use_clahe: bool = False


@dataclass(frozen=True)
class Stripe:
    """One candidate painted bar."""

    box: np.ndarray            # (4, 2) float32 corners of the min-area rect
    centre: tuple[float, float]
    angle_rad: float           # orientation of the major axis
    length: float
    width: float
    area: float
    bearing: float = 0.0       # angle of the centre around the vanishing point
    radius: float = 0.0        # distance of the centre from the vanishing point


@dataclass(frozen=True)
class StripeGroup:
    """A run of stripes that looks like one crossing."""

    stripes: list[Stripe]
    polygon: np.ndarray        # (N, 2) int32 convex hull over the member stripes
    score: float

    @property
    def count(self) -> int:
        return len(self.stripes)


@dataclass(frozen=True)
class StripeResult:
    groups: list[StripeGroup] = field(default_factory=list)
    stripes_considered: int = 0

    @property
    def score(self) -> float:
        """Best group score, 0.0 if nothing was found."""
        return max((g.score for g in self.groups), default=0.0)

    @property
    def found(self) -> bool:
        return bool(self.groups)


def _odd(value: int) -> int:
    return value if value % 2 else value + 1


def road_polygon(
    width: int, height: int, vp: tuple[float, float], params: StripeParams
) -> np.ndarray:
    """Trapezoid approximating the carriageway inside an ROI."""
    apex_y = max(0.0, min(float(vp[1]), height - 1.0))
    half = width * params.road_apex_half_width_fraction
    return np.array(
        [
            [width * params.road_base_left_fraction, height],
            [width * params.road_base_right_fraction, height],
            [vp[0] + half, apex_y],
            [vp[0] - half, apex_y],
        ],
        dtype=np.int32,
    )


def find_stripes(
    roi_bgr: np.ndarray,
    params: StripeParams = StripeParams(),
    vp: tuple[float, float] | None = None,
) -> list[Stripe]:
    """Bright, elongated, roughly stripe-shaped blobs in the ROI.

    With `vp` given, only blobs whose centre lies on the carriageway
    trapezoid are kept.
    """
    if roi_bgr.size == 0:
        return []
    height, width = roi_bgr.shape[:2]
    grey = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    if params.use_clahe:
        grey = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(grey)

    # Top-hat keeps structures brighter than their surroundings and narrower
    # than the kernel: road paint survives, the road itself does not.
    k = max(3, _odd(int(width * params.tophat_width_fraction)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, kernel)
    tophat = cv2.GaussianBlur(tophat, (3, 3), 0)
    _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Opening, not closing: closing bridges the gaps between adjacent bars and
    # welds a whole crossing into one blob, which is the one shape this
    # detector must never see.
    binary = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )

    roi_area = float(height * width)
    min_area = roi_area * params.min_area_fraction
    max_area = roi_area * params.max_area_fraction

    road = (
        road_polygon(width, height, vp, params)
        if vp is not None and params.use_road_trapezoid
        else None
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stripes: list[Stripe] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        (cx, cy), (w, h), angle_deg = cv2.minAreaRect(contour)
        if road is not None and cv2.pointPolygonTest(road, (float(cx), float(cy)), False) < 0:
            continue
        long_side, short_side = max(w, h), min(w, h)
        if short_side <= 0 or long_side / short_side < params.min_elongation:
            continue
        # minAreaRect's angle describes the width side; rotate when height is longer.
        angle = math.radians(angle_deg if w >= h else angle_deg + 90.0)
        stripes.append(
            Stripe(
                box=cv2.boxPoints(((cx, cy), (w, h), angle_deg)).astype(np.float32),
                centre=(float(cx), float(cy)),
                angle_rad=angle,
                length=float(long_side),
                width=float(short_side),
                area=float(area),
            )
        )
    return stripes


def _radial_alignment(stripe: Stripe, vp: tuple[float, float]) -> float:
    """|cos| of the angle between the stripe's axis and its radial direction.

    1.0 means the bar points straight at the vanishing point (a lane dash);
    0.0 means it lies square across the line of sight (a crossing bar).
    """
    cx, cy = stripe.centre
    rx, ry = cx - vp[0], cy - vp[1]
    norm = math.hypot(rx, ry)
    if norm < 1e-6:
        return 1.0
    rx, ry = rx / norm, ry / norm
    ux, uy = math.cos(stripe.angle_rad), math.sin(stripe.angle_rad)
    return abs(ux * rx + uy * ry)


def _is_tangential(stripe: Stripe, vp: tuple[float, float], max_cos: float) -> bool:
    """True for a bar lying across the line of sight, i.e. a crossing bar."""
    return _radial_alignment(stripe, vp) <= max_cos


def _bearing_and_radius(
    stripe: Stripe, vp: tuple[float, float]
) -> tuple[float, float]:
    dx, dy = stripe.centre[0] - vp[0], stripe.centre[1] - vp[1]
    return math.atan2(dy, dx), math.hypot(dx, dy)


def _coefficient_of_variation(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean <= 0:
        return float("inf")
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / mean


def _principal_direction(points: np.ndarray) -> tuple[float, float]:
    """Unit vector along the dominant axis of a point set (PCA, 2-D)."""
    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    vx, vy = float(vt[0][0]), float(vt[0][1])
    norm = math.hypot(vx, vy)
    return (vx / norm, vy / norm) if norm > 1e-9 else (1.0, 0.0)


def _axis_clusters(
    stripes: list[Stripe], tolerance_deg: float
) -> list[list[Stripe]]:
    """Split stripes into groups of near-parallel image axes (mod 180 deg)."""
    remaining = sorted(stripes, key=lambda s: s.angle_rad % math.pi)
    clusters: list[list[Stripe]] = []
    current: list[Stripe] = []
    tolerance = math.radians(tolerance_deg)
    for stripe in remaining:
        angle = stripe.angle_rad % math.pi
        if current and min(
            abs(angle - (current[-1].angle_rad % math.pi)),
            math.pi - abs(angle - (current[-1].angle_rad % math.pi)),
        ) <= tolerance:
            current.append(stripe)
        else:
            if current:
                clusters.append(current)
            current = [stripe]
    if current:
        clusters.append(current)
    return clusters


def group_stripes(
    stripes: list[Stripe],
    vp: tuple[float, float],
    frame_width: int,
    params: StripeParams = StripeParams(),
    frame_height: int | None = None,
) -> list[StripeGroup]:
    """Find ladders: near-parallel bars stacked in depth, lying across their
    own line of centres."""
    height = frame_height if frame_height is not None else frame_width
    if len(stripes) < params.min_stripes:
        return []

    enriched: list[Stripe] = []
    for stripe in stripes:
        bearing, radius = _bearing_and_radius(stripe, vp)
        enriched.append(
            Stripe(
                box=stripe.box, centre=stripe.centre, angle_rad=stripe.angle_rad,
                length=stripe.length, width=stripe.width, area=stripe.area,
                bearing=bearing, radius=radius,
            )
        )

    min_gap = height * params.min_radial_gap_fraction
    max_gap = height * params.max_radial_gap_fraction

    groups: list[StripeGroup] = []
    for cluster in _axis_clusters(enriched, params.axis_cluster_degrees):
        if len(cluster) < params.min_stripes:
            continue
        cluster.sort(key=lambda s: s.radius)
        run = [cluster[0]]
        for stripe in cluster[1:]:
            gap = stripe.radius - run[-1].radius
            if gap < min_gap:
                continue                      # fragments of one bar
            if gap <= max_gap:
                run.append(stripe)
                continue
            groups.extend(_finish_run(run, params))
            run = [stripe]
        groups.extend(_finish_run(run, params))
    return sorted(groups, key=lambda g: g.score, reverse=True)


def _finish_run(run: list[Stripe], params: StripeParams) -> list[StripeGroup]:
    """Score one depth-ordered ladder, or reject it."""
    if len(run) < params.min_stripes:
        return []

    centres = np.array([s.centre for s in run], dtype=np.float64)
    centre_dir = _principal_direction(centres)
    # Mean axis as a doubled-angle average, so 179 deg and 1 deg agree.
    mean_sin = sum(math.sin(2 * s.angle_rad) for s in run) / len(run)
    mean_cos = sum(math.cos(2 * s.angle_rad) for s in run) / len(run)
    mean_axis_angle = 0.5 * math.atan2(mean_sin, mean_cos)
    axis_dir = (math.cos(mean_axis_angle), math.sin(mean_axis_angle))

    alignment = abs(axis_dir[0] * centre_dir[0] + axis_dir[1] * centre_dir[1])
    if alignment > params.max_axis_vs_centreline_cos:
        return []                            # beads on a string: a lane line

    gaps = [b.radius - a.radius for a, b in zip(run, run[1:])]
    if any(g <= 0 for g in gaps):
        return []
    low, high = params.gap_ratio_bounds
    ratios = [b / a for a, b in zip(gaps, gaps[1:])] or [1.0]
    if any(r < low or r > high for r in ratios):
        return []
    ratio_cv = _coefficient_of_variation(ratios)
    if ratio_cv > params.max_gap_ratio_cv:
        return []

    count_score = min(len(run) / 6.0, 1.0)
    regularity = max(0.0, 1.0 - ratio_cv / params.max_gap_ratio_cv)
    squareness = max(0.0, 1.0 - alignment / params.max_axis_vs_centreline_cos)
    uniformity = max(0.0, 1.0 - _coefficient_of_variation([s.length for s in run]))
    score = round(
        0.40 * count_score + 0.25 * regularity + 0.20 * squareness + 0.15 * uniformity, 4
    )

    hull = cv2.convexHull(np.vstack([s.box for s in run]).astype(np.float32))
    return [
        StripeGroup(
            stripes=list(run), polygon=hull.reshape(-1, 2).astype(np.int32), score=score
        )
    ]


def detect(
    roi_bgr: np.ndarray,
    vp: tuple[float, float],
    params: StripeParams = StripeParams(),
) -> StripeResult:
    """Detect crossings in an ROI. `vp` is in the ROI's own pixel coordinates."""
    stripes = find_stripes(roi_bgr, params, vp=vp)
    groups = group_stripes(
        stripes, vp, roi_bgr.shape[1], params, frame_height=roi_bgr.shape[0]
    )
    return StripeResult(groups=groups, stripes_considered=len(stripes))


def road_band(
    frame: np.ndarray, vp_y: float, margin_fraction: float = 0.05
) -> tuple[np.ndarray, int]:
    """Crop to the road: a little above the vanishing point, down to the bottom.

    Running the detector on anything higher is wasted work and actively
    harmful -- roadside vegetation and building facades produce hundreds of
    bright elongated blobs, and a few of them will line up into a convincing
    false ladder. Returns the band and the row offset applied.
    """
    top = int(max(0, min(frame.shape[0] - 1, vp_y - frame.shape[0] * margin_fraction)))
    return frame[top:], top


def vanishing_point_for(
    width: int, height: int, vp_x_fraction: float = 0.557, vp_y_fraction: float = 0.678
) -> tuple[float, float]:
    """Default VP in pixel coordinates of a full (uncropped) frame.

    Defaults are the measured medians for samples/dashcam.mp4 (optical-flow
    focus-of-expansion over 149 frame pairs). A different camera or mount needs
    its own measurement.
    """
    return (width * vp_x_fraction, height * vp_y_fraction)
