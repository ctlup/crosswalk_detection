"""Measure FPS, latency and resource use for each stage of the pipeline.

Frames are decoded once and held in RAM so the numbers describe the stage
being measured rather than the video decoder; decode is timed separately and
reported alongside, because it is a real cost in the live loop.

    python scripts/benchmark.py --help
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ped_lane.config import load_config  # noqa: E402
from ped_lane.metrics import RollingTimer, percentile  # noqa: E402
from ped_lane.overlay import draw_polygons, redact  # noqa: E402
from ped_lane.privacy import enforce_local_only  # noqa: E402
from ped_lane.run import prepare_frame  # noqa: E402
from ped_lane.stripes import StripeParams, detect, road_band, vanishing_point_for  # noqa: E402


@dataclass
class Sampler:
    """Polls GPU and CPU while a benchmark runs."""

    interval: float = 0.1
    gpu_util: list[float] = field(default_factory=list)
    gpu_mem_mb: list[float] = field(default_factory=list)
    cpu_pct: list[float] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def _poll(self) -> None:
        proc = psutil.Process()
        proc.cpu_percent(None)
        while not self._stop.is_set():
            self.cpu_pct.append(proc.cpu_percent(None))
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=4,
                )
                if out.returncode == 0 and out.stdout.strip():
                    util, mem = out.stdout.strip().splitlines()[0].split(",")
                    self.gpu_util.append(float(util))
                    self.gpu_mem_mb.append(float(mem))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "Sampler":
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self) -> dict:
        def avg(values):
            return round(statistics.fmean(values), 1) if values else 0.0
        return {
            "cpu_pct_mean": avg(self.cpu_pct[1:]),
            "cpu_pct_max": round(max(self.cpu_pct[1:], default=0.0), 1),
            "gpu_util_mean": avg(self.gpu_util),
            "gpu_util_max": round(max(self.gpu_util, default=0.0), 1),
            "gpu_mem_mb_max": round(max(self.gpu_mem_mb, default=0.0), 1),
        }


@dataclass
class Result:
    name: str
    frames: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float
    fps: float
    resources: dict
    note: str = ""


def time_stage(name: str, fn, frames: list, warmup: int, note: str = "") -> Result:
    for frame in frames[:warmup]:
        fn(frame)
    timer = RollingTimer(window=len(frames) + 1)
    with Sampler() as sampler:
        started = time.perf_counter()
        for frame in frames:
            t0 = time.perf_counter()
            fn(frame)
            timer.add_seconds(time.perf_counter() - t0)
        wall = time.perf_counter() - started
    stats = timer.stats()
    return Result(
        name=name,
        frames=len(frames),
        mean_ms=round(stats.mean_ms, 2),
        p50_ms=round(stats.p50_ms, 2),
        p95_ms=round(stats.p95_ms, 2),
        max_ms=round(stats.max_ms, 2),
        fps=round(len(frames) / wall, 1) if wall > 0 else 0.0,
        resources=sampler.summary(),
        note=note,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python scripts/benchmark.py", description=__doc__.split("\n\n")[0]
    )
    ap.add_argument("--source", default="samples/dashcam.mp4")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--frames", type=int, default=320, help="frames to time")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default="docs/BENCHMARKS.md")
    ap.add_argument("--json", default="runs/benchmarks.json")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    enforce_local_only()
    args = parse_args(argv)
    cfg = load_config(args.config)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[error] cannot open {args.source}", file=sys.stderr)
        return 2
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[info] decoding {args.frames + args.warmup} frames from {args.source} ...")
    decode_times: list[float] = []
    raw: list[np.ndarray] = []
    while len(raw) < args.frames + args.warmup:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        decode_times.append((time.perf_counter() - t0) * 1000.0)
        raw.append(frame)
    cap.release()
    if len(raw) < args.frames + args.warmup:
        print(f"[warn] only {len(raw)} frames available")
    warm, timed = raw[: args.warmup], raw[args.warmup:]
    print(f"[info] {len(timed)} timed frames at {width}x{height}")

    results: list[Result] = []

    # --- decode, measured during the read above -----------------------------
    decode_mean = statistics.fmean(decode_times[args.warmup:]) if decode_times else 0.0
    results.append(
        Result(
            "video decode (1920x1080 h264, CPU)", len(decode_times[args.warmup:]),
            round(decode_mean, 2), round(percentile(decode_times[args.warmup:], 0.5), 2),
            round(percentile(decode_times[args.warmup:], 0.95), 2),
            round(max(decode_times[args.warmup:], default=0), 2),
            round(1000.0 / decode_mean, 1) if decode_mean else 0.0,
            {}, "cv2.VideoCapture.read(), unavoidable in the live loop",
        )
    )

    # --- redaction -----------------------------------------------------------
    results.append(time_stage(
        "redact() mask + bottom crop",
        lambda f: redact(f, cfg.dashcam.ignore_regions, cfg.dashcam.crop_bottom_fraction),
        timed, args.warmup, "privacy step applied to every saved/processed frame",
    ))

    # --- classical stripe baseline ------------------------------------------
    params = StripeParams()
    vp_x, vp_y = vanishing_point_for(width, height)

    def stripe_stage(frame):
        img = redact(frame, cfg.dashcam.ignore_regions, cfg.dashcam.crop_bottom_fraction)
        band, offset = road_band(img, vp_y)
        return detect(band, (vp_x, vp_y - offset), params)

    results.append(time_stage(
        "classical stripe baseline (CPU, OpenCV)", stripe_stage, timed, args.warmup,
        "redact + road band + detect. 0 geometric true positives: see BASELINE_RESULT.md",
    ))

    # --- the student stand-in -----------------------------------------------
    from ped_lane.detector import SegDetector

    for device, imgsz in (("cuda:0", 640), ("cuda:0", 960), ("cpu", 640)):
        try:
            det = SegDetector(cfg.model.weights, device=device, imgsz=imgsz,
                              conf=cfg.model.conf)
        except Exception as exc:
            print(f"[warn] skipping {device}/{imgsz}: {exc}")
            continue
        det.warmup(height=540, width=1920, runs=3)

        def loop_stage(frame, det=det):
            prepared = prepare_frame(frame, cfg.dashcam)
            result = det.predict(prepared).restricted_to(cfg.active_classes)
            return draw_polygons(
                prepared, [d.polygon for d in result.detections],
                [d.class_id for d in result.detections], cfg.output.overlay_alpha,
            )

        n = len(timed) if device != "cpu" else min(len(timed), 150)
        results.append(time_stage(
            f"video loop + overlay: {Path(det.weights).name} @ {device}, imgsz {imgsz}",
            loop_stage, timed[:n], min(args.warmup, 10),
            "crop -> infer -> overlay. GENERIC COCO WEIGHTS, not a crosswalk model",
        ))

    _write_report(Path(args.out), Path(args.json), results, args, width, height, src_fps)
    print()
    for r in results:
        print(f"{r.name:58} {r.fps:7.1f} FPS  mean {r.mean_ms:7.2f} ms  p95 {r.p95_ms:7.2f} ms")
    print(f"\n[info] wrote {args.out} and {args.json}")
    return 0


def _write_report(out: Path, json_path: Path, results, args, width, height, src_fps) -> None:
    import torch

    out.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps([r.__dict__ for r in results], indent=2), encoding="utf-8"
    )

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    lines = [
        "# Benchmarks",
        "",
        f"Measured on this PC. Source `{args.source}` ({width}x{height}, {src_fps:.2f} fps). "
        f"{args.frames} timed frames after {args.warmup} warm-up frames, frames held in RAM "
        "so the decoder is not counted twice.",
        "",
        f"- GPU: **{gpu}**",
        f"- CPU: {platform.processor() or 'unknown'}",
        f"- torch {torch.__version__}, OpenCV {cv2.__version__}, Python "
        f"{platform.python_version()}",
        "",
        "`cpu %` is this process against one core (so >100% means multiple threads). "
        "`gpu %` and VRAM are whole-device readings from `nvidia-smi`, sampled at 10 Hz, "
        "and include anything else using the GPU.",
        "",
        "| stage | frames | FPS | mean ms | p50 | p95 | max | cpu % mean | gpu % mean | VRAM MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        res = r.resources or {}
        lines.append(
            f"| {r.name} | {r.frames} | {r.fps} | {r.mean_ms} | {r.p50_ms} | {r.p95_ms} | "
            f"{r.max_ms} | {res.get('cpu_pct_mean', '-')} | {res.get('gpu_util_mean', '-')} | "
            f"{res.get('gpu_mem_mb_max', '-')} |"
        )
    by_name = {r.name: r for r in results}
    decode = next((r for r in results if r.name.startswith("video decode")), None)
    lines += ["", "## End-to-end live loop", ""]
    for key in [k for k in by_name if k.startswith("video loop")]:
        stage = by_name[key]
        total = stage.mean_ms + (decode.mean_ms if decode else 0.0)
        lines.append(
            f"- `{key}` + decode = **{total:.2f} ms/frame ({1000.0 / total:.1f} FPS)** "
            f"end to end, against targets of >= 10 FPS and < 300 ms."
        )
    lines += [
        "",
        "The classical baseline at ~19.6 ms is **slower than the neural model on the "
        "GPU** (~8 ms) while detecting nothing correctly, so it is not a cheap "
        "fallback either.",
        "",
        "### Superseded: the milestone-1 numbers",
        "",
        "An earlier report gave 49.8 FPS / 15.0 ms for `cuda:0, imgsz 640` and 11.1 FPS "
        "/ 97.4 ms for CPU. Those are not comparable to the table above: they were "
        "measured on 1920x810 frames (`crop_top_fraction` was 0.15, now 0.40, so the "
        "model sees 1920x540), and they included decode, the HUD and video writing in "
        "the timed section. Use this file, not those figures.",
    ]
    lines += ["", "## Notes", ""]
    for r in results:
        if r.note:
            lines.append(f"- **{r.name}** — {r.note}")
    lines += [
        "",
        "## Not benchmarked yet",
        "",
        "| model | status | why |",
        "|---|---|---|",
        "| **Trained crosswalk student** (`yolo11n-seg` fine-tuned) | **does not exist** | "
        "no training has been run; there are only 9 labelled-positive seconds, which is "
        "not a trainable dataset. See `docs/FOOTAGE_PLAN.md`. |",
        "| **Mask2Former Swin-L Mapillary teacher** | **not downloaded** | weights are "
        "CC BY-NC 4.0 and not cleared; 826 MB. Gated behind `teacher.enabled: false`. "
        "See `LICENSING.md`. |",
        "| **ONNX / TensorRT / OpenVINO exports** | **not built** | nothing worth "
        "exporting until a trained model exists. |",
        "",
        "The `video loop + overlay` rows use the **generic COCO `yolo11n-seg`** fallback. "
        "They measure the pipeline's speed, not crosswalk accuracy — that model cannot "
        "detect a crosswalk at all. Treat them as an upper bound on what a same-size "
        "crosswalk student would cost.",
        "",
        "Regenerate with `python scripts/benchmark.py`.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
