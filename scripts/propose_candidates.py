"""Run the classical stripe baseline over footage and rank frames by score.

Two uses:

  * **Proposer** -- point it at new footage and it lists the timestamps most
    likely to contain a crossing, so labelling starts where the crossings are.
  * **Baseline evaluation** -- pass `--labels` (a directory of YOLO-seg
    crosswalk polygons) for GEOMETRIC precision/recall, which is the only
    detection-quality number worth quoting. `--truth-seconds` additionally
    reports a TEMPORAL coincidence counter, clearly labelled as not being
    detection quality.

    python scripts/propose_candidates.py --help
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ped_lane.config import load_config  # noqa: E402
from ped_lane.evaluation import (  # noqa: E402
    DEFAULT_IOU,
    evaluate_geometric,
    evaluate_temporal,
)
from ped_lane.overlay import apply_top_crop, redact, top_crop_rows  # noqa: E402
from ped_lane.privacy import enforce_local_only  # noqa: E402
from ped_lane.stripes import (  # noqa: E402
    StripeParams,
    detect,
    road_band,
    vanishing_point_for,
)


@dataclass
class Scored:
    seconds: float
    label: str
    score: float
    stripes: int
    groups: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python scripts/propose_candidates.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", help="directory of extracted frames (with manifest.json)")
    src.add_argument("--video", help="video file; frames are redacted on the fly")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fps", type=float, default=1.0, help="sample rate for --video")
    ap.add_argument("--threshold", type=float, default=0.45, help="score to call a candidate")
    ap.add_argument("--top", type=int, default=25, help="how many to list")
    ap.add_argument("--csv", default=None, help="write all scores here")
    ap.add_argument("--annotate", default=None, help="directory for annotated candidates")
    ap.add_argument(
        "--labels",
        default=None,
        help="directory of YOLO-seg crosswalk labels; enables GEOMETRIC scoring",
    )
    ap.add_argument(
        "--iou", type=float, default=DEFAULT_IOU, help="IoU for a geometric match"
    )
    ap.add_argument(
        "--truth-seconds",
        default=None,
        help="comma-separated labelled positive seconds; reports the TEMPORAL "
             "coincidence counter only, which is NOT detection quality",
    )
    ap.add_argument(
        "--no-crop-top",
        action="store_true",
        help="use crop_top_fraction instead of the road band (for comparison)",
    )
    ap.add_argument(
        "--no-redact",
        action="store_true",
        help="skip redaction for a source with no burned-in overlay and a "
             "different camera geometry (e.g. stock footage). NEVER use on "
             "dashcam footage.",
    )
    ap.add_argument("--vp-x", type=float, default=0.557, help="vanishing point x fraction")
    ap.add_argument("--vp-y", type=float, default=0.678, help="vanishing point y fraction")
    return ap.parse_args(argv)


def _roi_and_vp(frame, source_height: int, crop_top: float, args):
    """Road-band ROI plus the vanishing point expressed in its coordinates.

    The baseline uses the road band (from just above the vanishing point
    downwards) rather than `crop_top_fraction`. `crop_top` is a budget for the
    neural model's input size; the classical detector only ever wants road.
    """
    vp_x, vp_y = vanishing_point_for(frame.shape[1], source_height, args.vp_x, args.vp_y)
    if args.no_crop_top:
        rows = top_crop_rows(source_height, crop_top)
        return apply_top_crop(frame, rows), (vp_x, vp_y - rows)
    roi, offset = road_band(frame, vp_y)
    return roi, (vp_x, vp_y - offset)


def _iter_frames(args, cfg):
    """Yield (seconds, label, redacted_frame, source_height)."""
    if args.frames:
        d = Path(args.frames)
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        source_height = manifest["source"]["height"]
        for record in manifest["frames"]:
            img = cv2.imread(str(d / record["file"]))
            if img is not None:
                yield float(record["seconds"]), record["file"], img, source_height
        return

    path = Path(args.video)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"[error] cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(src_fps / args.fps)))
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                seconds = index / src_fps
                red = (
                    frame
                    if args.no_redact
                    else redact(
                        frame,
                        cfg.dashcam.ignore_regions,
                        cfg.dashcam.crop_bottom_fraction,
                    )
                )
                yield seconds, f"{path.stem}@{seconds:.2f}s", red, source_height
            index += 1
    finally:
        cap.release()


def _load_truth_polygons(labels_dir: Path, image_name: str, height: int, width: int):
    """Read YOLO-seg polygons for one frame, in pixel coordinates."""
    path = labels_dir / (Path(image_name).stem + ".txt")
    if not path.exists():
        return None
    polygons = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 7:
            continue
        values = [float(v) for v in fields[1:]]
        polygons.append(
            np.array(
                [[values[i] * width, values[i + 1] * height]
                 for i in range(0, len(values), 2)],
                dtype=np.float32,
            )
        )
    return polygons


def main(argv: list[str] | None = None) -> int:
    enforce_local_only()
    args = parse_args(argv)
    cfg = load_config(args.config)
    params = StripeParams()

    annotate_dir = Path(args.annotate) if args.annotate else None
    if annotate_dir:
        annotate_dir.mkdir(parents=True, exist_ok=True)

    labels_dir = Path(args.labels) if args.labels else None
    geometric_frames: list[dict] = []

    scored: list[Scored] = []
    for seconds, label, frame, source_height in _iter_frames(args, cfg):
        roi, vp = _roi_and_vp(frame, source_height, cfg.dashcam.crop_top_fraction, args)
        result = detect(roi, vp, params)
        scored.append(
            Scored(seconds, label, result.score, result.stripes_considered, len(result.groups))
        )
        if labels_dir is not None:
            truth = _load_truth_polygons(
                labels_dir, label, frame.shape[0], frame.shape[1]
            )
            if truth is not None:
                offset = frame.shape[0] - roi.shape[0]
                detections = []
                if result.score >= args.threshold:
                    for group in result.groups:
                        polygon = group.polygon.astype(np.float32).copy()
                        polygon[:, 1] += offset
                        detections.append(polygon)
                geometric_frames.append(
                    {
                        "detections": detections,
                        "truth": truth,
                        "height": frame.shape[0],
                        "width": frame.shape[1],
                    }
                )
        if annotate_dir and result.found:
            vis = roi.copy()
            for group in result.groups:
                cv2.polylines(vis, [group.polygon], True, (60, 220, 255), 3)
                for stripe in group.stripes:
                    cv2.drawContours(vis, [stripe.box.astype(int)], -1, (90, 90, 255), 2)
            cv2.putText(
                vis, f"{seconds:.1f}s score={result.score:.2f} n={result.groups[0].count}",
                (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.imwrite(str(annotate_dir / f"cand_{seconds:08.2f}s.jpg"), vis)

    if not scored:
        print("[error] no frames scored", file=sys.stderr)
        return 2

    scored.sort(key=lambda s: s.score, reverse=True)
    print(f"[info] scored {len(scored)} frames; threshold {args.threshold}")
    above = [s for s in scored if s.score >= args.threshold]
    print(f"[info] {len(above)} frame(s) at or above threshold\n")
    print(f"{'seconds':>9} {'score':>6} {'stripes':>8} {'groups':>7}  label")
    for s in scored[: args.top]:
        mark = "*" if s.score >= args.threshold else " "
        print(f"{s.seconds:9.2f} {s.score:6.3f} {s.stripes:8d} {s.groups:7d} {mark} {s.label}")

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["seconds", "score", "stripes", "groups", "label"])
            for s in sorted(scored, key=lambda s: s.seconds):
                writer.writerow([f"{s.seconds:.3f}", f"{s.score:.4f}", s.stripes, s.groups, s.label])
        print(f"\n[info] wrote {out}")

    if labels_dir is not None:
        if not geometric_frames:
            print()
            print(f"[warn] no label files found in {labels_dir}; "
                  "geometric scoring skipped. Run scripts/label_crosswalk.py first.")
        else:
            print()
            for line in evaluate_geometric(geometric_frames, args.iou).as_lines():
                print(line)
    else:
        print()
        print("[warn] no --labels given, so NO detection-quality metric "
              "was computed. Geometric precision/recall needs labelled polygons.")

    if args.truth_seconds:
        truth = {int(round(float(t))) for t in args.truth_seconds.split(",") if t.strip()}
        print()
        temporal = evaluate_temporal(
            [(s.seconds, s.score) for s in scored], truth, args.threshold
        )
        for line in temporal.as_lines():
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
