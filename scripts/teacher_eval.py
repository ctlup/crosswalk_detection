"""Evaluate the Mask2Former Mapillary teacher on the labelled frames.

INTERNAL EVALUATION ONLY. The released weights are CC BY-NC 4.0 and are not
cleared for this project (see LICENSING.md). This script therefore refuses to
run without an explicit `--confirm-download` flag, and nothing it produces may
be used as training data.

What it does, once confirmed:

  1. downloads `facebook/mask2former-swin-large-mapillary-vistas-panoptic`
     (826 MB) into the local Hugging Face cache, telemetry disabled;
  2. runs it on the labelled crossing frames plus a sample of empty frames,
     full frame, no `crop_top` (the teacher is the reference; cropping is a
     speed optimisation for the student);
  3. scores GEOMETRIC IoU / precision / recall against the hand labels;
  4. measures FPS, latency and GPU use on this PC;
  5. renders overlays into `runs/teacher_eval/`.

Render rules (deliberately different from the demo renders): a translucent
filled polygon following the detected region, no bounding boxes, no score
text, no temporal smoothing, no region held over from a previous frame. If
nothing is detected on a frame, nothing is drawn.

    python scripts/teacher_eval.py --help
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ped_lane import classes, mapillary  # noqa: E402
from ped_lane.config import load_config  # noqa: E402
from ped_lane.evaluation import DEFAULT_IOU, evaluate_geometric  # noqa: E402
from ped_lane.privacy import enforce_local_only  # noqa: E402

OUT_DIR = Path("runs/teacher_eval")
WEIGHTS_MB = 826
FILL = (70, 210, 120)
EDGE = (120, 245, 170)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python scripts/teacher_eval.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--frames", default="data/frames/dashcam")
    ap.add_argument("--labels", default="data/labels/dashcam")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument(
        "--positive-seconds",
        default="13,14,42,43,44,99,100,148,149",
        help="labelled crossing seconds",
    )
    ap.add_argument("--empty-frames", type=int, default=20)
    ap.add_argument("--iou", type=float, default=DEFAULT_IOU)
    ap.add_argument("--threshold", type=float, default=0.5, help="panoptic threshold")
    ap.add_argument("--min-area", type=int, default=400, help="drop tiny mask blobs")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument(
        "--confirm-download",
        action="store_true",
        help=f"REQUIRED. Acknowledges the {WEIGHTS_MB} MB CC BY-NC 4.0 download.",
    )
    ap.add_argument("--benchmarks", default="docs/BENCHMARKS.md")
    return ap.parse_args(argv)


def select_frames(manifest: dict, positives: set[int], empty_count: int) -> list[dict]:
    """The labelled crossing frames plus evenly spread empty ones."""
    records = manifest["frames"]
    chosen = [r for r in records if int(round(r["seconds"])) in positives]
    # keep empties well away from a crossing, so "empty" really is empty
    far = [
        r for r in records
        if all(abs(int(round(r["seconds"])) - p) > 3 for p in positives)
    ]
    if far and empty_count > 0:
        step = max(1, len(far) // empty_count)
        chosen += far[::step][:empty_count]
    return sorted(chosen, key=lambda r: r["seconds"])


def crosswalk_polygons(result, min_area: int) -> list[np.ndarray]:
    """Crosswalk mask from the teacher's panoptic output, as contours."""
    mask = result.class_mask(classes.CROSSWALK).astype(np.uint8)
    if not mask.any():
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = 0.004 * cv2.arcLength(contour, True)
        polygons.append(
            cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).astype(np.float32)
        )
    return polygons


def load_labels(labels_dir: Path, image_name: str, height: int, width: int):
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


def render(image: np.ndarray, polygons: list[np.ndarray]) -> np.ndarray:
    """Translucent fill following the region. Nothing drawn when nothing found."""
    out = image.copy()
    if polygons is None or len(polygons) == 0:
        return out
    fill = out.copy()
    for polygon in polygons:
        cv2.fillPoly(fill, [polygon.astype(np.int32)], FILL)
    cv2.addWeighted(fill, 0.38, out, 0.62, 0, out)
    for polygon in polygons:
        cv2.polylines(out, [polygon.astype(np.int32)], True, EDGE, 3, cv2.LINE_AA)
    return out


def main(argv: list[str] | None = None) -> int:
    enforce_local_only()
    args = parse_args(argv)

    if not args.confirm_download:
        print(
            "[blocked] This downloads the Mask2Former Mapillary checkpoint\n"
            f"          (~{WEIGHTS_MB} MB, CC BY-NC 4.0, NOT cleared for this project).\n"
            "          Nothing has been downloaded.\n\n"
            "          Re-run with --confirm-download to proceed. Output is for\n"
            "          internal evaluation only and must not be used as training\n"
            "          data. See LICENSING.md sections 2 and 4.",
            file=sys.stderr,
        )
        return 3

    cfg = load_config(args.config)
    frames_dir = Path(args.frames)
    manifest = json.loads((frames_dir / "manifest.json").read_text(encoding="utf-8"))
    positives = {int(round(float(s))) for s in args.positive_seconds.split(",") if s.strip()}
    selected = select_frames(manifest, positives, args.empty_frames)
    print(f"[info] {len(selected)} frames "
          f"({sum(1 for r in selected if int(round(r['seconds'])) in positives)} labelled "
          f"positive, rest empty)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = Path(args.labels)

    from ped_lane.teacher import Mask2FormerTeacher

    print(f"[info] loading {cfg.teacher.checkpoint} (first run downloads ~{WEIGHTS_MB} MB)")
    teacher = Mask2FormerTeacher(
        checkpoint=cfg.teacher.checkpoint, device=cfg.teacher.device, fp16=cfg.teacher.fp16
    )
    print(f"[info] {teacher.label}")
    teacher.warmup(height=972, width=1920, runs=2)

    latencies: list[float] = []
    geometric_frames: list[dict] = []
    per_frame: list[dict] = []
    missing_labels = 0

    for record in selected:
        image = cv2.imread(str(frames_dir / record["file"]))
        if image is None:
            continue
        started = time.perf_counter()
        result = teacher.predict(image, threshold=args.threshold)
        latencies.append((time.perf_counter() - started) * 1000.0)

        polygons = crosswalk_polygons(result, args.min_area)
        truth = load_labels(labels_dir, record["file"], image.shape[0], image.shape[1])
        is_positive = int(round(record["seconds"])) in positives
        if truth is None:
            if is_positive:
                missing_labels += 1
            truth = []
        geometric_frames.append({
            "detections": polygons, "truth": truth,
            "height": image.shape[0], "width": image.shape[1],
        })
        per_frame.append({
            "file": record["file"], "seconds": record["seconds"],
            "labelled_positive": is_positive,
            "crosswalk_regions": len(polygons),
            "crosswalk_pixels": int(result.class_mask(classes.CROSSWALK).sum()),
            "inference_ms": round(latencies[-1], 2),
        })
        if not args.no_render:
            cv2.imwrite(
                str(out_dir / f"teacher_{record['seconds']:08.2f}s.jpg"),
                render(image, polygons), [cv2.IMWRITE_JPEG_QUALITY, 92],
            )

    if not latencies:
        print("[error] no frames processed", file=sys.stderr)
        return 2

    geometric = evaluate_geometric(geometric_frames, args.iou)
    mean_ms = statistics.fmean(latencies)
    ordered = sorted(latencies)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]

    print()
    for line in geometric.as_lines():
        print(line)
    if missing_labels:
        print(f"  [warn] {missing_labels} labelled-positive frame(s) had no label file; "
              "recall is understated")
    print(f"\nspeed: {1000.0 / mean_ms:.2f} FPS  mean {mean_ms:.1f} ms  p95 {p95:.1f} ms  "
          f"({teacher.label})")

    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "checkpoint": cfg.teacher.checkpoint,
                "device": teacher.device,
                "fp16": teacher.fp16,
                "iou_threshold": args.iou,
                "geometric": {
                    "true_positives": geometric.true_positives,
                    "false_positives": geometric.false_positives,
                    "false_negatives": geometric.false_negatives,
                    "precision": round(geometric.precision, 4),
                    "recall": round(geometric.recall, 4),
                    "f1": round(geometric.f1, 4),
                    "mean_iou": round(geometric.mean_iou, 4),
                },
                "speed": {
                    "frames": len(latencies),
                    "fps": round(1000.0 / mean_ms, 2),
                    "mean_ms": round(mean_ms, 2),
                    "p95_ms": round(p95, 2),
                },
                "frames": per_frame,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[info] wrote {out_dir / 'results.json'} and {len(per_frame)} overlay images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
