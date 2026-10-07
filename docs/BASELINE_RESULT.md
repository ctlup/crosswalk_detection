# Classical stripe baseline: measured result

**Verdict: the baseline is not usable. Measured geometrically against labelled
polygons: precision 0.250, recall 0.111, F1 0.154, mean IoU 0.191.**

## Measured geometric result (the number to quote)

`scripts/propose_candidates.py --frames data/frames/dashcam --labels data/labels/dashcam
--threshold 0.40`, over the 174 extracted 1 fps frames, 29 of which are labelled
(9 positive polygons + 20 explicit negatives):

| | |
|---|---|
| frames with labels | 29 |
| true positives (IoU >= 0.10) | **1** |
| false positives | **3** |
| false negatives | **8** |
| precision | **0.250** |
| recall | **0.111** |
| F1 | **0.154** |
| mean IoU of the matched detection | **0.191** |

Frame by frame:

| second | score | detections | best IoU | outcome |
|---|---:|---:|---:|---|
| 13 | 0.000 | 0 | – | missed |
| **14** | **0.482** | 1 | **0.191** | **true positive** |
| 42 | 0.000 | 0 | – | missed |
| 43 | 0.000 | 0 | – | missed |
| 44 | 0.000 | 0 | – | missed |
| 99 | 0.000 | 0 | – | missed |
| 100 | 0.433 | 1 | 0.000 | fired, but on something else |
| 115 | 0.781 | 1 | – | false positive (labelled empty) |
| 148 | 0.000 | 0 | – | missed |
| 149 | 0.000 | 0 | – | missed |
| 158 | 0.731 | 1 | – | false positive (labelled empty) |

The single true positive clears only the deliberately lenient 0.10 IoU bar at
0.191, i.e. the detected region covers well under a quarter of the crossing.
It would not count at the 0.5 IoU usual in segmentation work.

### Correction to an earlier statement

An earlier report said "**0 of 30** temporally on-truth detections overlap a
crosswalk". That was a by-eye audit of the **30 fps frames inside the demo
windows**, which is a different sample from the 1 fps extracted frames measured
here. On the measured set there is **one** genuine overlap, at t=14 s. The
"zero" was too strong a generalisation; the measured figures above supersede
it. The conclusion — that the baseline is not usable — is unchanged.

### About the labels

The labels are **drafts traced by the assistant** from gridded screenshots, not
by a human (`data/labels/dashcam/README.txt`). They are approximate, and the
faint ones (13, 42, 43, 99, 148 s) least reliable. Re-run the command above
after correcting them in `scripts/label_crosswalk.py` and update this table.


Measured on the five demo clips (four windows around the known crossings in
`samples/dashcam.mp4`, plus the Pexels stock clip), at score threshold 0.40,
with `StripeParams` exactly as shipped. The detector was not tuned to produce
these numbers.

## Headline

| | |
|---|---|
| Frames scored | 1511 |
| Frames the detector fired on | **99** |
| Of those, inspected by eye for overlap | 30 |
| Judged by eye to overlap a crosswalk | **0** (see the correction above) |

## Raw per-frame, by clip

| clip | frames | labelled-positive frames | fired | temporal on-truth | temporal off-truth |
|---|---|---|---|---|---|
| A (13–14 s) | 285 | 60 | 5 | 2 | 3 |
| B (42–44 s) | 360 | 90 | 39 | 15 | 24 |
| C (99–100 s) | 329 | 59 | 19 | 3 | 16 |
| D (148–149 s) | 330 | 60 | 27 | 1 | 26 |
| Pexels | 207 | 204 | 9 | 9 | 0 |
| **total** | **1511** | **473** | **99** | **30** | **69** |

The "temporal on-truth" column is **not** a true-positive count. It counts
firings during a second labelled as containing a crossing somewhere in frame,
with no check that the detected region is on the crossing.

## Why the temporal column is worthless here

All 30 temporally on-truth detections were inspected individually. **None
overlapped a crosswalk.** They landed on:

- kerb lines and the edge of the carriageway (most of clip B)
- a parked white car (clip C, t=99.37 s)
- grass verges and roadside scrub (clips A, B, C)
- railings, fences and a bus-shelter frame (clip D, all 12 shown frames)
- shop frontages and a doorway (clip B, t=43.01–43.48 s)
- **a manhole grating** (all 9 Pexels firings, while a crossing filled the frame)

The nearest misses are clip B at t=43.74 s and t=44.18 s, where the detected
band sits directly *above* the zebra with the stripes falling outside its
lower edge.

Pexels scores a temporal precision of 1.000 purely because the whole clip is
labelled positive, so every firing is "on-truth" by construction. That single
number is the clearest demonstration of why the temporal metric must never be
quoted as detection quality.

## Superseded figures — do not quote

An earlier report gave:

> "P 0.182 / R 0.222 / F1 0.200 at threshold 0.40", "catches 2 of 9 crossings",
> and per-clip "hits / misses / false positives" tables.

**All of these are invalid.** Every one was computed temporally. The correct
geometric figures are precision 0.000, recall 0.000, F1 0.000.

## What changed as a result

- `ped_lane.evaluation` now provides `evaluate_geometric` (the only source of
  precision / recall / F1, requires labelled polygons) and `evaluate_temporal`
  (prints under the heading `TEMPORAL (not detection quality)` and has no
  precision/recall attributes at all).
- `scripts/propose_candidates.py` takes `--labels` for geometric scoring and
  warns loudly when no labels are supplied, so a run cannot quietly produce a
  temporal-only number that reads like detection quality.
- `scripts/render_demo.py` labels its HIT/MISS captions and printed counts as
  temporal.
- `stripes.py` and `tracking.py` are unchanged. The result is a property of
  the detector, not of a bug in it.

## Caveats

- Ground truth is labelled at **1 fps**; intermediate frames inherit a label
  by rounding and were not individually reviewed.
- The geometric audit was done by eye because no crosswalk polygons have been
  labelled yet. Once `scripts/label_crosswalk.py` has been run on the nine
  positive seconds, `--labels` will produce the number automatically and this
  document should be updated with it.
- Nine positive seconds from one clip is far too small a sample to characterise
  any detector. The conclusion here is strong only because the result is zero.
