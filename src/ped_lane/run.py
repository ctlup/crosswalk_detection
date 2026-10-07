"""Live loop: grab a frame -> mask/crop -> segment -> overlay -> display/record.

Run with ``--help`` for the options. Recorded clips are processed frame by
frame and the run ends at EOF; live sources always use the newest frame and
reconnect on failure.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from . import classes
from .config import Config, load_config, resolve_source
from .detector import SegDetector
from .metrics import FpsCounter, RollingTimer
from .overlay import (
    apply_top_crop,
    draw_hud,
    draw_legend,
    draw_polygons,
    redact,
    top_crop_rows,
)
from .privacy import enforce_local_only
from .stream import FrameSource, LatestFrameGrabber, open_source

WINDOW_NAME = "pedestrian-lane"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m ped_lane.run",
        description="Panoptic-style crosswalk / sidewalk / person / vehicle overlay "
        "for forward-facing dashcam footage.",
    )
    ap.add_argument("--config", default="config.yaml", help="path to the yaml config")
    ap.add_argument(
        "--source",
        default=None,
        help="video file, camera index, or stream URL; overrides CAMERA_URL and the config",
    )
    ap.add_argument("--weights", default=None, help="override model.weights")
    ap.add_argument(
        "--device",
        default=None,
        help="override model.device (auto, cpu, a GPU index like 0, or cuda:0)",
    )
    ap.add_argument("--imgsz", type=int, default=None, help="override model.imgsz")
    ap.add_argument("--conf", type=float, default=None, help="override model.conf")
    ap.add_argument("--save-video", default=None, help="write the annotated video here")
    ap.add_argument(
        "--no-window", action="store_true", help="headless: do not open a display window"
    )
    ap.add_argument(
        "--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)"
    )
    ap.add_argument("--no-hud", action="store_true", help="hide the FPS/latency overlay")
    return ap.parse_args(argv)


def merge_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Apply CLI overrides onto a loaded config."""
    model = cfg.model
    if args.weights is not None:
        model = replace(model, weights=Path(args.weights))
    if args.device is not None:
        model = replace(model, device=args.device)
    if args.imgsz is not None:
        model = replace(model, imgsz=args.imgsz)
    if args.conf is not None:
        model = replace(model, conf=args.conf)

    output = cfg.output
    if args.save_video is not None:
        output = replace(output, save_video=Path(args.save_video))
    if args.no_window:
        output = replace(output, show_window=False)

    return replace(cfg, model=model, output=output)


def prepare_frame(frame: np.ndarray, dashcam) -> np.ndarray:
    """Redact (mask + hood/overlay crop), then drop the sky rows.

    The same two steps the extraction pipeline uses, in the same order, so a
    live frame and a saved frame reach the model looking identical.
    """
    redacted = redact(frame, dashcam.ignore_regions, dashcam.crop_bottom_fraction)
    rows = top_crop_rows(frame.shape[0], dashcam.crop_top_fraction)
    return apply_top_crop(redacted, rows)


def hud_lines(
    detector_label: str,
    fps: FpsCounter,
    inference: RollingTimer,
    end_to_end: RollingTimer,
    frame_index: int,
    source_label: str,
) -> list[str]:
    inf = inference.stats()
    e2e = end_to_end.stats()
    return [
        f"{fps.fps:5.1f} FPS   frame {frame_index}",
        f"infer   {inf.mean_ms:5.1f} ms  p95 {inf.p95_ms:5.1f}",
        f"end2end {e2e.mean_ms:5.1f} ms  p95 {e2e.p95_ms:5.1f}",
        detector_label,
        source_label,
    ]


def _make_writer(path: Path, size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, max(fps, 1.0), size)
    if not writer.isOpened():
        raise RuntimeError(f"cannot open a video writer for {path}")
    return writer


def _print_summary(
    detector_label: str,
    frames: int,
    wall_seconds: float,
    inference: RollingTimer,
    end_to_end: RollingTimer,
    class_totals: dict[int, int],
) -> None:
    average_fps = frames / wall_seconds if wall_seconds > 0 else 0.0
    print("\n--- run summary ---")
    print(f"model/device : {detector_label}")
    print(f"frames       : {frames} in {wall_seconds:.1f} s ({average_fps:.1f} FPS average)")
    print(inference.stats().as_line("inference    "))
    print(end_to_end.stats().as_line("end-to-end   "))
    if class_totals:
        detail = ", ".join(
            f"{classes.name_for(cid)}={n}" for cid, n in sorted(class_totals.items())
        )
        print(f"detections   : {detail}")
    else:
        print("detections   : none")


def _open_frames(source: str | int, cfg: Config) -> FrameSource | None:
    try:
        return open_source(source, cfg.stream.reconnect_seconds)
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        print(
            "[hint] put a dashcam clip in samples\\ , or pass --source "
            "<file|camera index|stream url>",
            file=sys.stderr,
        )
        return None


def run(cfg: Config, args: argparse.Namespace) -> int:
    source = resolve_source(args.source, cfg)
    frames_in = _open_frames(source, cfg)
    if frames_in is None:
        return 2

    detector = SegDetector(
        weights=cfg.model.weights,
        device=cfg.model.device,
        imgsz=cfg.model.imgsz,
        conf=cfg.model.conf,
    )
    scope = ", ".join(classes.active_names(cfg.active_classes))
    print(f"[info] classes in scope: {scope}")
    if detector.is_generic:
        print(
            f"[warn] {cfg.model.weights} not found, falling back to "
            f"{detector.weights}: it detects person/vehicle only, never crosswalk "
            "or sidewalk. Expected until the student is trained."
        )
        if not set(cfg.active_classes) & {classes.PERSON, classes.VEHICLE}:
            print(
                f"[warn] the fallback model cannot detect {scope}, so this run "
                "will show no detections at all. It still exercises the loop, "
                "the crop and the timing."
            )
    print(f"[info] model/device: {detector.label}")
    detector.warmup()

    live = frames_in.is_live
    min_dt = 1.0 / cfg.stream.max_fps if (live and cfg.stream.max_fps > 0) else 0.0
    source_label = f"{'live' if live else 'file'}: {source}"

    fps = FpsCounter()
    inference = RollingTimer(window=120)
    end_to_end = RollingTimer(window=120)
    class_totals: dict[int, int] = {}

    writer: cv2.VideoWriter | None = None
    save_path = cfg.output.save_video
    writer_fps = getattr(frames_in, "source_fps", 0.0) or cfg.stream.max_fps

    last_seq, last_started, frames_done = -1, 0.0, 0
    started_wall = time.perf_counter()

    try:
        while True:
            if frames_in.exhausted:
                print("[info] end of video")
                break

            if isinstance(frames_in, LatestFrameGrabber):
                seq, frame, frame_age_s = frames_in.read_with_age()
            else:
                seq, frame = frames_in.read()
                frame_age_s = 0.0

            if frame is None:
                if not frames_in.exhausted:
                    time.sleep(0.005)
                continue
            if live and (seq == last_seq or time.perf_counter() - last_started < min_dt):
                time.sleep(0.002)
                continue

            last_seq = seq
            last_started = time.perf_counter()
            # Charge the time the frame already spent buffered to end-to-end latency.
            loop_started = last_started - frame_age_s

            prepared = prepare_frame(frame, cfg.dashcam)

            result = detector.predict(prepared).restricted_to(cfg.active_classes)
            inference.add_ms(result.inference_ms)
            counts = result.counts()
            for class_id, count in counts.items():
                class_totals[class_id] = class_totals.get(class_id, 0) + count

            vis = draw_polygons(
                prepared,
                [d.polygon for d in result.detections],
                [d.class_id for d in result.detections],
                cfg.output.overlay_alpha,
            )
            draw_legend(vis, counts)

            frames_done += 1
            fps.tick()
            end_to_end.add_seconds(time.perf_counter() - loop_started)

            if not args.no_hud:
                draw_hud(
                    vis,
                    hud_lines(
                        detector.label, fps, inference, end_to_end, frames_done, source_label
                    ),
                )

            if save_path is not None:
                if writer is None:
                    writer = _make_writer(save_path, (vis.shape[1], vis.shape[0]), writer_fps)
                writer.write(vis)

            if cfg.output.show_window:
                cv2.imshow(WINDOW_NAME, vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[info] quit requested")
                    break

            if args.max_frames and frames_done >= args.max_frames:
                print(f"[info] reached --max-frames {args.max_frames}")
                break
    except KeyboardInterrupt:
        print("\n[info] interrupted")
    finally:
        wall = time.perf_counter() - started_wall
        reconnects = getattr(frames_in, "reconnects", 0)
        frames_in.stop()
        if writer is not None:
            writer.release()
            print(f"[info] wrote {save_path}")
        cv2.destroyAllWindows()
        _print_summary(detector.label, frames_done, wall, inference, end_to_end, class_totals)
        if reconnects:
            print(f"[warn] source reconnected {reconnects} time(s)")

    return 0 if frames_done else 1


def main(argv: list[str] | None = None) -> int:
    enforce_local_only()  # before anything touches frames or the network
    args = parse_args(argv)
    if not Path(args.config).exists():
        print(f"[error] config not found: {args.config}", file=sys.stderr)
        return 2
    cfg = merge_overrides(load_config(args.config), args)
    return run(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
