# Crosswalk Detector (dashcam)

Pedestrian-crossing (zebra crosswalk) detection from forward-facing dashcam footage.

> **Licensing — read before this goes anywhere near a product.**
> Ultralytics (`yolo11n-seg`) is **AGPL-3.0**, which has network-copyleft
> obligations, and the Mask2Former teacher weights are **CC BY-NC 4.0**
> (non-commercial) and are **not downloaded**, pending legal clearance.
> See **[LICENSING.md](LICENSING.md)**.

**Scope:** one class, `crosswalk` (`active_classes` in `config.yaml`).
`sidewalk`, `person` and `vehicle` remain implemented and can be switched back
on without code changes. All frames and outputs stay on this PC: the dashcam
burns a GPS/date overlay into every frame.

## Quick start (Windows, PowerShell)
```powershell
py -3 -m venv .venv                 # no `py` on this PC: use
                                    # & "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # NVIDIA GPU
pip install -r requirements.txt

# put a dashcam clip in samples\, then:
$env:PYTHONPATH = "src"
python -m ped_lane.run --source samples\dashcam.mp4
# live stream
$env:CAMERA_URL = "rtsp://CAMERA_IP:554/stream1"
python -m ped_lane.run
```
Press `q` in the window to quit. `--help` lists every flag; the useful ones are
`--device`, `--imgsz`, `--conf`, `--save-video`, `--no-window`, `--max-frames`.
Without custom weights it falls back to a generic COCO model that detects
person/vehicle only and will NOT detect crosswalks or sidewalks; that is
expected until training is done.

Tests: `.venv\Scripts\python.exe -m pytest` (no PYTHONPATH needed, see pyproject.toml).

## Measured on this PC (milestone 1)
RTX A4000 16 GB / i9-13900K, clip `samples/dashcam.mp4` (1920x1080, 29.97 fps),
after the hood/sky crop the model sees 1920x810. Generic `yolo11n-seg`,
headless, no video writing:

| device | imgsz | loop FPS | inference mean / p95 | end-to-end mean / p95 |
|---|---|---|---|---|
| cuda:0 | 640 | 49.8 | 15.0 / 16.7 ms | 17.1 / 21.1 ms |
| cuda:0 | 960 | 47.9 | 15.6 / 17.0 ms | 17.8 / 21.5 ms |
| cpu    | 640 | 11.1 | 97.4 / 109.5 ms | 99.5 / 111.5 ms |

Both GPU targets (>= 10 FPS, < 300 ms) are met with a wide margin. Going from
640 to 960 costs almost nothing on the GPU because the time is dominated by
CPU-side pre/post-processing rather than convolution, so the larger input is
effectively free for the distant, thin crosswalks -- worth re-checking once the
real student model is in place. The CPU fallback clears 10 FPS, but only just.

## Documentation

- `docs/DEVELOPMENT.md` — approach, data discipline, dashcam specifics, stack,
  layout and conventions
- `docs/FOOTAGE_PLAN.md` — what the next footage must contain
- `docs/BASELINE_RESULT.md` — the classical baseline's measured result
- `docs/BENCHMARKS.md` — measured FPS, latency and resource use
- `LICENSING.md` — licence position (AGPL, CC BY-NC) and data protection

## Local reference files (not in the repo)

`data/` is gitignored in full, so a fresh clone has no `data/reference/`. Two
small files live there and are regenerated locally rather than committed:

- `data/reference/mapillary_vistas_labels.json` — the 65-class Mapillary Vistas
  label list. Rebuild it from the public checkpoint config without downloading
  any weights:

  ```powershell
  .venv\Scripts\python.exe -c @'
  import json, pathlib, urllib.request
  url = "https://huggingface.co/facebook/mask2former-swin-large-mapillary-vistas-panoptic/resolve/main/config.json"
  cfg = json.loads(urllib.request.urlopen(url, timeout=30).read().decode())
  out = pathlib.Path("data/reference/mapillary_vistas_labels.json")
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps({int(k): v for k, v in cfg["id2label"].items()}, indent=2, sort_keys=True), encoding="utf-8")
  '@
  ```

  `src/ped_lane/mapillary.py` works without it; only the human-readable Vistas
  class names are unavailable.

- `data/reference/licenses/` — upstream licence texts, fetched from
  `facebookresearch/Mask2Former` (`LICENSE`, `MODEL_ZOO.md`) and
  `creativecommons.org/licenses/by-nc/4.0/legalcode.txt`. See `LICENSING.md`.

## Classical baseline: measured result

The OpenCV zebra-stripe baseline **does not work on this footage**. Measured
geometrically against labelled crosswalk polygons: **precision 0.250, recall
0.111, F1 0.154, mean IoU 0.191** — one true positive out of nine labelled
crossings, and that one overlaps under a quarter of the crossing.

> **Superseded figures.** An earlier report quoted "P 0.182 / R 0.222 /
> F1 0.200" and "catches 2 of 9 crossings". **Those numbers are invalid.** They
> were computed temporally — a firing counted as correct if the *second* was
> labelled, with no check that the detected region was on the crossing. Do not
> quote them. Detection quality is only ever
> `ped_lane.evaluation.evaluate_geometric`, which needs labelled polygons.

See `docs/BASELINE_RESULT.md`.

## Roadmap (teacher-free)
1. Live/video loop with overlay, FPS and latency — **done**
2. Extract frames at 1 fps, crop/mask applied before writing, survey how many
   contain a crosswalk
3. OpenCV polygon labelling tool + classical zebra-stripe baseline
4. Fine-tune `yolo11n-seg` on the hand-labelled set, single class
5. Evaluate by crosswalk precision/recall/IoU, train/val split **by time
   segment** (never random frames — consecutive frames are near-duplicates)
6. Export (ONNX / TensorRT) and benchmark

The Mask2Former teacher path is coded and tested but gated behind
`teacher.enabled: false`; see [LICENSING.md](LICENSING.md).
