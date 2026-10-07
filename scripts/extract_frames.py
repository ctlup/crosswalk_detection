"""Extract frames from a dashcam clip at a fixed rate, redacted before writing.

Every frame written to disk has already been through `overlay.redact()`:
fixed obstructions blanked and the bottom strip carrying the dashcam's
burned-in GPS / date / speed overlay removed. Nothing leaves this machine.

`crop_top_fraction` is deliberately NOT applied. It is a tuning knob, and
baking it in would force re-extraction and re-labelling every time it changes.
It is recorded in the manifest as the default to apply at training and
inference time, and `top_crop_rows(manifest.source_height, fraction)` converts
it correctly for these shorter frames.

    python scripts/extract_frames.py --help
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ped_lane.config import load_config  # noqa: E402
from ped_lane.overlay import redact, top_crop_rows  # noqa: E402
from ped_lane.privacy import enforce_local_only  # noqa: E402

MANIFEST_NAME = "manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python scripts/extract_frames.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--source", required=True, help="dashcam clip to extract from")
    ap.add_argument("--config", default="config.yaml", help="path to the yaml config")
    ap.add_argument("--out", default=None, help="output dir (default data/frames/<clip stem>)")
    ap.add_argument("--fps", type=float, default=1.0, help="frames per second to keep")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N frames (0 = all)")
    ap.add_argument(
        "--overwrite", action="store_true", help="allow writing into a non-empty output dir"
    )
    ap.add_argument(
        "--allow-no-bottom-crop",
        action="store_true",
        help="permit crop_bottom_fraction == 0 (the burned-in overlay would be KEPT)",
    )
    return ap.parse_args(argv)


def frame_step(source_fps: float, target_fps: float) -> int:
    """How many source frames to advance between kept frames (at least 1)."""
    if source_fps <= 0:
        raise ValueError("source fps must be > 0")
    if target_fps <= 0:
        raise ValueError("target fps must be > 0")
    return max(1, int(round(source_fps / target_fps)))


def frame_name(stem: str, index: int, seconds: float) -> str:
    """Sortable name carrying both the source frame index and its timestamp."""
    return f"{stem}_f{index:06d}_t{seconds:08.2f}s.jpg"


def main(argv: list[str] | None = None) -> int:
    enforce_local_only()
    args = parse_args(argv)

    source = Path(args.source)
    if not source.is_file():
        print(f"[error] no such clip: {source}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    dashcam = cfg.dashcam

    if dashcam.crop_bottom_fraction <= 0 and not args.allow_no_bottom_crop:
        print(
            "[error] crop_bottom_fraction is 0, so the burned-in GPS/date overlay "
            "would be written to disk. Set it in config.yaml, or pass "
            "--allow-no-bottom-crop if this clip genuinely has no overlay.",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out) if args.out else Path("data/frames") / source.stem
    if out_dir.exists() and any(out_dir.glob("*.jpg")) and not args.overwrite:
        print(
            f"[error] {out_dir} already contains frames; pass --overwrite to replace them.",
            file=sys.stderr,
        )
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)

    # Purge before rewriting. Without this a frame that a newly added
    # exclude_range is meant to drop would survive from the previous run --
    # exactly the content we are trying to remove.
    purged = 0
    for stale in out_dir.glob("*.jpg"):
        stale.unlink()
        purged += 1
    if purged:
        print(f"[info] purged {purged} frame(s) from the previous run")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        print(f"[error] cannot open {source}", file=sys.stderr)
        return 2

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = frame_step(source_fps, args.fps)

    print(f"[info] {source}: {width}x{height}, {source_fps:.2f} fps, {total} frames "
          f"({total / source_fps:.1f} s)")
    print(f"[info] keeping every {step}th frame -> {args.fps:g} fps")
    print(f"[info] redaction: bottom crop {dashcam.crop_bottom_fraction}, "
          f"{len(dashcam.ignore_regions)} ignore region(s)")
    if cfg.extraction.exclude_ranges:
        print(f"[info] exclude_ranges: {[list(r) for r in cfg.extraction.exclude_ranges]} s")
    print(f"[info] crop_top_fraction {dashcam.crop_top_fraction} is NOT baked in "
          f"(recorded in the manifest for train/inference)")
    print(f"[info] output: {out_dir.resolve()}")

    records: list[dict] = []
    excluded: list[float] = []
    index = 0
    kept_shape: tuple[int, int] | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                seconds = index / source_fps
                if cfg.extraction.is_excluded(seconds):
                    excluded.append(round(seconds, 3))
                    index += 1
                    continue
                out = redact(frame, dashcam.ignore_regions, dashcam.crop_bottom_fraction)
                if kept_shape is None:
                    kept_shape = out.shape[:2]
                    print(f"[info] redacted frame size: {out.shape[1]}x{out.shape[0]}")
                elif out.shape[:2] != kept_shape:
                    print(f"[error] frame {index} changed shape; aborting", file=sys.stderr)
                    return 3
                name = frame_name(source.stem, index, seconds)
                if not cv2.imwrite(
                    str(out_dir / name), out, [cv2.IMWRITE_JPEG_QUALITY, args.quality]
                ):
                    print(f"[error] failed to write {name}", file=sys.stderr)
                    return 3
                records.append(
                    {"file": name, "source_frame": index, "seconds": round(seconds, 3)}
                )
                if args.limit and len(records) >= args.limit:
                    break
            index += 1
    finally:
        cap.release()

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "path": str(source),
            "width": width,
            "height": height,
            "fps": round(source_fps, 4),
            "frame_count": total,
        },
        "extraction": {
            "target_fps": args.fps,
            "frame_step": step,
            "jpeg_quality": args.quality,
            "exclude_ranges": [list(r) for r in cfg.extraction.exclude_ranges],
            "excluded_seconds": excluded,
        },
        "redaction_applied": {
            "crop_bottom_fraction": dashcam.crop_bottom_fraction,
            "ignore_regions": [list(r) for r in dashcam.ignore_regions],
            "note": "Applied before writing. Fractions refer to the ORIGINAL frame.",
        },
        "apply_at_train_and_inference": {
            "crop_top_fraction": dashcam.crop_top_fraction,
            "top_crop_rows": top_crop_rows(height, dashcam.crop_top_fraction),
            "note": (
                "NOT baked into these frames. Compute rows from source.height, "
                "never from the saved frame height."
            ),
        },
        "frame_size": {"width": kept_shape[1], "height": kept_shape[0]} if kept_shape else None,
        "frames": records,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[info] wrote {len(records)} frames + {MANIFEST_NAME}")
    if excluded:
        print(f"[info] dropped {len(excluded)} frame(s) inside exclude_ranges: {excluded}")
    if kept_shape:
        rows = top_crop_rows(height, dashcam.crop_top_fraction)
        print(f"[info] at train/inference: drop {rows} rows from the top -> "
              f"{kept_shape[1]}x{kept_shape[0] - rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
