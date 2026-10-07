"""Render overlay videos showing what the classical baseline actually does.

Diagnostic output, not a demo reel: every frame is captioned with the ground
truth and with the baseline's verdict, so a miss is as visible as a hit. The
detector is used exactly as configured -- this script never tunes it.

The HIT / MISS captions and the printed counts are TEMPORAL only: they ask
whether the second is labelled as containing a crossing, never whether the
drawn region lands on one. For detection quality use
`scripts/propose_candidates.py --labels`.

Dashcam sources go through `redact()` first, so the burned-in GPS/date bar is
not in the output, and frames inside `extraction.exclude_ranges` are skipped.
Everything stays on this PC.

    python scripts/render_demo.py --help
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ped_lane.config import load_config  # noqa: E402
from ped_lane.overlay import redact  # noqa: E402
from ped_lane.privacy import enforce_local_only  # noqa: E402
from ped_lane.stripes import (  # noqa: E402
    StripeParams,
    detect,
    road_band,
    vanishing_point_for,
)
from ped_lane.tracking import RegionTracker, TrackerParams  # noqa: E402

GREEN = (90, 230, 90)
GREY = (170, 170, 170)
YELLOW = (60, 220, 255)
RED = (90, 90, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python scripts/render_demo.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", required=True, help="output mp4")
    ap.add_argument("--start", type=float, default=0.0, help="window start, seconds")
    ap.add_argument("--end", type=float, default=0.0, help="window end (0 = to the end)")
    ap.add_argument(
        "--truth-seconds", default="", help="comma-separated ground-truth positive seconds"
    )
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument(
        "--no-redact",
        action="store_true",
        help="source has no burned-in overlay and a different geometry (stock footage)",
    )
    ap.add_argument("--scale", type=float, default=0.75, help="output scale")
    ap.add_argument(
        "--style",
        choices=("clean", "eval"),
        default="clean",
        help="clean: smoothed translucent region + status badge (default). "
             "eval: raw per-frame detections with hit/miss captions.",
    )
    ap.add_argument("--exit-score", type=float, default=0.25)
    ap.add_argument("--enter-frames", type=int, default=3)
    ap.add_argument("--exit-frames", type=int, default=10)
    return ap.parse_args(argv)


def _badge(canvas, detected: bool) -> None:
    """Small status chip, bottom-left. No scores, no numbers."""
    height, width = canvas.shape[:2]
    scale = width / 1600.0
    text = "Crosswalk detected" if detected else "No crosswalk"
    colour = GREEN if detected else GREY
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.62 * scale, 2)
    pad = int(14 * scale)
    x0, y1 = pad, height - pad
    x1, y0 = x0 + tw + int(46 * scale), y1 - th - int(20 * scale)
    chip = canvas[y0:y1, x0:x1]
    if chip.size:
        canvas[y0:y1, x0:x1] = cv2.addWeighted(chip, 0.35, np.zeros_like(chip), 0.65, 0)
    cv2.circle(canvas, (x0 + int(18 * scale), (y0 + y1) // 2), int(7 * scale), colour, -1)
    cv2.putText(canvas, text, (x0 + int(34 * scale), y1 - int(10 * scale)), FONT,
                0.62 * scale, (240, 240, 240), 2, cv2.LINE_AA)


def _draw_region(canvas, quad, opacity: float) -> None:
    """Translucent fill plus a soft edge. No boxes, no per-frame numbers."""
    pts = np.asarray(quad, dtype=np.int32).reshape(-1, 2)
    fill = canvas.copy()
    cv2.fillPoly(fill, [pts], (70, 210, 120))
    cv2.addWeighted(fill, 0.38 * opacity, canvas, 1.0 - 0.38 * opacity, 0, canvas)
    edge = canvas.copy()
    cv2.polylines(edge, [pts], True, (120, 245, 170), max(2, canvas.shape[1] // 600),
                  cv2.LINE_AA)
    cv2.addWeighted(edge, opacity, canvas, 1.0 - opacity, 0, canvas)


def _caption(canvas, seconds, is_truth, fired, score, threshold, note):
    """Two-line banner: ground truth, then the baseline's verdict."""
    height, width = canvas.shape[:2]
    bar = int(78 * width / 1440) + 8
    cv2.rectangle(canvas, (0, 0), (width, bar), (0, 0, 0), -1)
    scale = width / 1600.0

    truth_text = "KNOWN CROSSING" if is_truth else "no crossing"
    cv2.putText(canvas, f"t={seconds:6.2f}s", (12, int(28 * scale) + 4), FONT,
                0.8 * scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, truth_text, (int(190 * scale) + 12, int(28 * scale) + 4), FONT,
                0.8 * scale, GREEN if is_truth else GREY, 2, cv2.LINE_AA)

    if is_truth and fired:
        verdict, colour = f"HIT  score {score:.2f}", GREEN
    elif is_truth and not fired:
        verdict, colour = f"MISS  score {score:.2f} < {threshold:.2f}", (80, 80, 255)
    elif fired:
        verdict, colour = f"FALSE POSITIVE  score {score:.2f}", (0, 140, 255)
    else:
        verdict, colour = "correct reject", GREY
    cv2.putText(canvas, f"baseline: {verdict}", (12, int(58 * scale) + 6), FONT,
                0.75 * scale, colour, 2, cv2.LINE_AA)
    if note:
        cv2.putText(canvas, note, (width - int(430 * scale), int(58 * scale) + 6), FONT,
                    0.5 * scale, GREY, 1, cv2.LINE_AA)
    return canvas


def main(argv: list[str] | None = None) -> int:
    enforce_local_only()
    args = parse_args(argv)
    cfg = load_config(args.config)
    params = StripeParams()
    truth = {int(round(float(t))) for t in args.truth_seconds.split(",") if t.strip()}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[error] cannot open {args.video}", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    end = args.end if args.end > 0 else total / fps

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    written = skipped = hits = misses = false_positives = 0
    shown_on_truth = shown_off_truth = 0
    tracker = RegionTracker(
        TrackerParams(
            enter_score=args.threshold,
            exit_score=args.exit_score,
            enter_frames=args.enter_frames,
            exit_frames=args.exit_frames,
        )
    )

    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            seconds = index / fps
            index += 1
            if seconds < args.start or seconds > end:
                continue
            if not args.no_redact and cfg.extraction.is_excluded(seconds):
                skipped += 1
                continue

            image = frame if args.no_redact else redact(
                frame, cfg.dashcam.ignore_regions, cfg.dashcam.crop_bottom_fraction
            )
            vp_x, vp_y = vanishing_point_for(image.shape[1], source_height)
            band, offset = road_band(image, vp_y)
            result = detect(band, (vp_x, vp_y - offset), params)

            best = None
            if result.found:
                best = result.groups[0].polygon.astype(np.float32).copy()
                best[:, 1] += offset

            fired = result.score >= args.threshold
            is_truth = int(round(seconds)) in truth
            if is_truth and fired:
                hits += 1
            elif is_truth:
                misses += 1
            elif fired:
                false_positives += 1

            # The tracker is fed every frame in both styles so its counters stay
            # comparable; only `clean` draws its output.
            state = tracker.update(result.score, best)
            if state.visible:
                if is_truth:
                    shown_on_truth += 1
                else:
                    shown_off_truth += 1

            vis = image.copy()
            if args.style == "eval":
                for group in result.groups:
                    polygon = group.polygon.copy()
                    polygon[:, 1] += offset
                    cv2.polylines(vis, [polygon], True, YELLOW, 3)
                    for stripe in group.stripes:
                        box = stripe.box.astype(np.int32).copy()
                        box[:, 1] += offset
                        cv2.drawContours(vis, [box], -1, RED, 2)
                cv2.line(vis, (0, offset), (vis.shape[1], offset), (70, 70, 70), 1)

            if args.scale != 1.0:
                vis = cv2.resize(
                    vis, (int(vis.shape[1] * args.scale), int(vis.shape[0] * args.scale))
                )

            if args.style == "eval":
                _caption(vis, seconds, is_truth, fired, result.score, args.threshold,
                         "TEMPORAL labels @1fps - not detection quality")
            else:
                if state.visible:
                    _draw_region(vis, np.asarray(state.quad) * args.scale, state.opacity)
                _badge(vis, state.visible)

            if writer is None:
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                    (vis.shape[1], vis.shape[0]),
                )
                if not writer.isOpened():
                    print(f"[error] cannot open writer for {out_path}", file=sys.stderr)
                    return 3
            writer.write(vis)
            written += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    print(f"[info] {out_path}  style={args.style}  frames={written}  "
          f"skipped(privacy)={skipped}")
    print(f"       raw   : fired on {tracker.raw_fired_frames} frame(s)")
    print(f"       clean : region shown on {tracker.shown_frames} frame(s)")
    print("       --- TEMPORAL counts below are NOT detection quality: they only")
    print("           ask whether the second is labelled, never whether the drawn")
    print("           region overlaps the crossing. See ped_lane.evaluation. ---")
    print(f"       raw   temporal: on-truth={hits} off-truth={false_positives} "
          f"not-fired-on-truth={misses}")
    print(f"       clean temporal: on-truth={shown_on_truth} off-truth={shown_off_truth}")
    delta = tracker.shown_frames - tracker.raw_fired_frames
    print(f"       smoothing changes displayed frames by {delta:+d} "
          f"(hold/hysteresis fills gaps and suppresses isolated spikes)")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
