# Development notes

Working conventions, measured facts and constraints for this project.

## Current scope: one class

`active_classes: ["crosswalk"]` in `config.yaml`. Only crosswalk is trained on
and drawn. `sidewalk`, `person` and `vehicle` remain fully defined in
`src/ped_lane/classes.py` and in the Mapillary mapping, and are re-enabled by
editing `active_classes` — that code should not be deleted. The longer-term
panoptic goal (stuff classes plus instance ids for pedestrians and vehicles) is
deferred, not abandoned.

## Blocked pending legal clearance

- **Mask2Former teacher weights**
  (`facebook/mask2former-swin-large-mapillary-vistas-panoptic`, 826 MB) are
  **CC BY-NC 4.0, non-commercial**. This project's purpose is not confirmed to
  be non-commercial.
- **Mapillary Vistas dataset** — same restriction.
- Neither is downloaded. The teacher code sits behind `teacher.enabled` in
  `config.yaml`, **default false**; nothing downloads while it is false.
  `scripts/teacher_eval.py` additionally refuses to run without an explicit
  `--confirm-download` flag.
- **Ultralytics is AGPL-3.0**, which affects any shipped product regardless of
  the teacher question. See `LICENSING.md`.

## Approach

1. Extract frames at 1 fps from dashcam clips, **with the dashcam crop and
   ignore-region masking already applied**, so the burned-in GPS/date/speed
   overlay is never written to disk.
2. Hand-label crosswalk polygons with the project's OpenCV labelling tool
   (`scripts/label_crosswalk.py`, no extra dependencies).
3. A classical OpenCV zebra-stripe detector exists as a baseline and candidate
   proposer. **Measured result: it does not work.** Geometric scoring against
   labelled polygons gives precision 0.250, recall 0.111, F1 0.154, mean IoU
   0.191 — one true positive out of nine labelled crossings, covering under a
   quarter of that crossing. It is a near-zero floor, not a working proposer or
   a fallback. See `docs/BASELINE_RESULT.md`.
4. Fine-tune Ultralytics `yolo11n-seg` on the hand-labelled set, single class.
5. Teacher-based pseudo-labelling stays designed and coded but switched off, as
   the route to take if and when the licence position clears.

## Data discipline

- **Split train/validation by time segment, never by random frame.**
  Consecutive dashcam frames are near-duplicates; a random split leaks
  validation frames into training and reports a meaningless score.
- One clip is one place, one camera, one time of day, one weather condition. A
  model trained on it will overfit; results from it should say so plainly, and
  more varied footage is the fix.
- Report positives honestly: how many frames actually contain a crosswalk, not
  how many frames were extracted.
- **Only geometric metrics count.** A detection is a true positive when its
  region overlaps a labelled crosswalk polygon (IoU >= 0.10 by default), never
  because it fired during a second that happens to contain a crossing.
  `ped_lane.evaluation.evaluate_geometric` is the only source of precision,
  recall and F1.
- **The temporal counter is not detection quality** and is always printed under
  that label (`evaluate_temporal`). It only reports how often the detector
  fired during a labelled second. It deliberately exposes no
  `precision`/`recall` attributes, so it cannot be quoted as though it were.
- Never judge by raw detection counts.
- **Negatives are part of the dataset, not an afterthought.** A frame reviewed
  and found to contain no crossing is training signal; a frame not yet reviewed
  is not. The labelling tool records that distinction explicitly (`x` marks a
  reviewed negative).
- **Two cases are mandatory in every dataset build.** The first clip had the
  blue pedestrian-crossing sign visible at all four of its crossings, so a model
  trained on it would learn the sign rather than the paint:
    1. **Crossings with no blue sign** — included as positives.
    2. **The blue sign with no crossing in view** — included as negatives.
  A dataset build containing neither is not fit to train on. See
  `docs/FOOTAGE_PLAN.md`.
- Other hard negatives worth collecting deliberately: stop lines, give-way
  triangles, lane arrows, bike-lane and bus-lane markings, hatched areas,
  manhole covers and drain gratings, and railing shadows across the road.

## Privacy (hard requirement)

Frames carry a burned-in GPS/date overlay, so a frame is personal data.

- Everything stays on the local machine. No uploads, no hosted labelling tools,
  no dataset sync, no sharing of frames or derived crops.
- `src/ped_lane/privacy.py::enforce_local_only()` runs first in every entry
  point that touches frames; it disables Hugging Face, Ultralytics, W&B, Comet,
  ClearML and Neptune telemetry, and Ultralytics hosted-dataset sync.
- Saved frames must have the crop and `dashcam.ignore_regions` applied first.
- `data/`, `models/`, `runs/` and `samples/` are gitignored as whole
  directories. See `LICENSING.md` for the data-protection position.

## Dashcam specifics

- The camera moves, so there is no fixed ROI and no background subtraction, and
  temporal smoothing must stay short (2–4 frames).
- Reference clip `samples/dashcam.mp4`: 1920x1080, 29.97 fps, 180 s. The road
  vanishing point was measured by optical-flow focus of expansion over 149 frame
  pairs: median 0.678 of frame height, p01 0.557, p99 0.761, pitch swing 0.20H.
  `crop_top_fraction` is set from that measurement, not guessed.
- Crop the car hood and the burned-in text bar
  (`dashcam.crop_bottom_fraction`), and blank fixed obstructions
  (`dashcam.ignore_regions`, fractions of the ORIGINAL full frame, applied
  before the crop).
- `extraction.exclude_ranges` drops whole seconds that geometric redaction
  cannot clean, such as a vehicle occupant's hand entering frame above the
  mask.
- Expect motion blur, glare, night and rain; include these in training and
  evaluation.
- Crosswalks seen at shallow angles are thin slivers; evaluate by distance
  bucket where possible.
- `imgsz` 640 vs 960 should be re-benchmarked only once a crosswalk model
  exists, and judged by crosswalk metrics. The current figures come from a
  generic COCO model and say nothing about crosswalk accuracy.

## Stack

Python 3.11 in a project-local `.venv`. Ultralytics 8.4, OpenCV, PyYAML, torch
2.6.0+cu124. `transformers` is pinned `>=4.40,<5` (teacher only; 5.x changed the
Mask2Former API). Reference hardware: NVIDIA RTX A4000 16 GB, Intel i9-13900K,
Windows 11.

Use PowerShell syntax on Windows, `pathlib` for paths, and no POSIX-only shell
commands. Assume no admin rights: no system-wide installs, everything inside the
project venv. Other Python environments may exist on the machine; leave them
alone. Keep the CPU fallback working.

## Layout

- `src/ped_lane/` — `classes.py` (taxonomy and active scope), `config.py`,
  `devices.py`, `metrics.py`, `overlay.py` (crop/mask/HUD), `detector.py`,
  `stream.py` (file and live sources), `privacy.py`, `run.py` (live loop),
  `stripes.py` (classical baseline), `tracking.py` (display smoothing),
  `evaluation.py` (geometric and temporal metrics), `teacher.py` and
  `mapillary.py` (gated, unused while `teacher.enabled: false`)
- `config.yaml` — runtime config; stream credentials via the `CAMERA_URL`
  environment variable, never committed
- `scripts/` — `extract_frames.py` (1 fps, redacted before writing),
  `label_crosswalk.py` (OpenCV polygon labelling, single class),
  `propose_candidates.py` (baseline proposer and evaluation),
  `render_demo.py` (overlay videos), `benchmark.py`, `teacher_eval.py` (gated).
  Dataset conversion, training and export are still to come.
- `docs/FOOTAGE_PLAN.md` — what the next footage must contain
- `docs/BASELINE_RESULT.md` — the classical baseline's measured result
- `docs/BENCHMARKS.md` — measured FPS, latency and resource use
- `LICENSING.md` — licence and data-protection position
- `data/`, `models/`, `runs/`, `samples/` are gitignored

## Conventions

- Type hints, small functions, no global state; every script runnable with
  `--help`.
- pytest for pure logic: crop and mask maths, class scoping, config parsing,
  label format round-trips, metric definitions, dataset conversion and
  time-based splitting.
- Verify on a recorded dashcam clip first, then live.
- Report measured FPS and latency for every model/device combination and keep
  `docs/BENCHMARKS.md` current. Models that have not been benchmarked are listed
  there as such, never silently omitted.
- Large downloads (>1 GB), heavy new dependencies and system-level installs are
  decisions to raise before acting, not defaults.

## Evaluation render rules

Renders that exist to show what a model found must not flatter it:

- a translucent filled polygon **following the detected region**; no bounding
  boxes, no score text
- **no temporal smoothing and no region held over from an earlier frame**.
  Smoothing stays off by default for evaluation renders; `tracking.py` is for
  presentation, not assessment.
- if nothing is detected on a frame, **draw nothing**
- write to a new folder per model (`runs/teacher_eval/` and so on), never over
  an existing `runs/` output
