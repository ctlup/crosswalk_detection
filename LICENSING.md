# Licensing position

Summary for review before this project goes any further than internal
experimentation. Nothing here is legal advice; it records what the upstream
licences say and where the open questions are. Full licence texts are saved
locally under `data/reference/licenses/`.

## Short version

| Component | Role | Licence | Status |
|---|---|---|---|
| **Ultralytics** (`yolo11n-seg`) | the model we train and would ship | **AGPL-3.0** | **Needs a decision** |
| Mask2Former **weights** | optional teacher / pseudo-labeller | **CC BY-NC 4.0** (non-commercial) | **Blocked, not downloaded** |
| Mapillary Vistas dataset | what the teacher was trained on | research / non-commercial | **Blocked, not downloaded** |
| Mask2Former **code** | — | MIT (Meta) | Fine |
| PyTorch | runtime | BSD-3-Clause | Fine |
| transformers | teacher loader only | Apache-2.0 | Fine |
| OpenCV (`opencv-python`) | video and image handling | Apache-2.0 | Fine |

## 1. Ultralytics is AGPL-3.0 — the one that affects the shipped product

The student model is `yolo11n-seg`, trained with the Ultralytics package, which
is licensed **AGPL-3.0** (confirmed from the installed package metadata, not
from memory).

AGPL-3.0 is a strong copyleft licence. In broad terms it requires that software
which incorporates or links to AGPL code be released under AGPL as well,
including its source — and, unlike GPL, that obligation is triggered by making
the software available **over a network**, not only by distributing binaries.

This applies **regardless of which teacher we use or whether the licence
question below is resolved**, because Ultralytics is in the training and
inference path either way.

Practical options, in rough order of cost:

1. **Internal use only** — if the detector never leaves the organisation and is
   not offered to third parties over a network, the obligations are far lighter.
2. **Buy an Ultralytics Enterprise licence** — the vendor sells one precisely to
   remove the AGPL obligation. This is the normal route for a commercial product.
3. **Comply with AGPL** — release the product's source under AGPL.
4. **Switch framework** — retrain on a permissively licensed detector/segmenter.
   Cheapest to do *now*, before the dataset and tooling are built around
   Ultralytics; expensive later.

**This should be decided before significant labelling effort goes in**, because
option 4 gets more costly with every week of work.

## 2. Mask2Former teacher weights are CC BY-NC 4.0 — currently blocked

The checkpoint `facebook/mask2former-swin-large-mapillary-vistas-panoptic`
(826 MB) would be used to pseudo-label dashcam frames, saving most of the manual
labelling work.

Its Hugging Face repository declares `license: other` and **ships no licence
file**. The operative terms are upstream, in
`facebookresearch/Mask2Former/MODEL_ZOO.md`, verbatim:

> #### License
> All models available for download through this document are licensed under the
> [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

CC BY-NC 4.0 §1(i) defines the restriction:

> **NonCommercial** means not primarily intended for or directed towards
> commercial advantage or monetary compensation.

The *code* in the same repository is MIT and is not the issue; the **weights**
are.

### The open question

The plan was teacher → pseudo-labels → student. Whether a model's **outputs**
(the pseudo-labels) are "Adapted Material" of its **weights**, and therefore
carry the NC restriction into a student trained on them, is **not settled law**.
A conservative reading says the restriction follows; an permissive reading says
outputs are facts about our own footage. We are not in a position to pick.

### What we are doing about it

Nothing is downloaded. The teacher code (`src/ped_lane/teacher.py`,
`src/ped_lane/mapillary.py`) is written and tested but sits behind
`teacher.enabled: false` in `config.yaml`; the download happens only when that
flag is turned on. The project is proceeding on a **teacher-free path**:
hand-labelled crosswalk polygons plus a classical OpenCV baseline.

If clearance comes through, two routes reopen:

- **Full use** — teacher pseudo-labels the training data (big time saving).
- **Reference only** — teacher is used solely to measure our accuracy and tune
  the crop, and never contributes labels to the shipped model. This sidesteps
  the derivative-work question while keeping most of the evaluation value.

## 3. Mapillary Vistas dataset — also blocked

The dataset the teacher was trained on is itself released for research /
non-commercial use. Relevant if we ever download it directly (it is also the
project's >1 GB download that requires explicit approval).

## 4. Data protection (GDPR) — needs explicit confirmation

**This is a separate question from licensing, and it needs a decision from
whoever is accountable for data protection.** The material is urban dashcam
footage of public roads, recorded from a moving vehicle; location and date are
withheld here deliberately, and the camera's burned-in location/time overlay is
removed before any frame is written to disk.

What the frames contain:

- **A burned-in GPS / date / speed overlay.** Latitude, longitude, timestamp
  and speed on every frame — precise location data tied to a specific vehicle
  at a specific moment. **This is removed before any frame is written to disk**
  (measured at rows 0.9565–0.9769 of frame height; the extraction crop cuts
  61 px above it, verified across all 180 extracted frames).
- **Identifiable pedestrians and other people.** Faces and whole bodies of
  members of the public who did not consent and are not aware of the
  recording — for example a pedestrian standing beside a parked car at
  t=42s, clearly identifiable.
- **A vehicle occupant's hand**, close to the lens and above the masked
  region, at roughly t=8s and t=29s. Geometric masking cannot remove this
  without discarding most of the right-hand roadway, so
  `extraction.exclude_ranges` in config.yaml **drops those seconds entirely**
  (6.5–9.5s and 27.5–30.5s; 6 of 180 frames, none containing a crosswalk).
  Found by reviewing every extracted frame individually — colour-based detection
  was unreliable, firing on sunlit tarmac.
- **Vehicle number plates**, which are personal data in the EU because they
  identify a registered keeper.
- Building frontages, shop signage and house numbers, which can identify
  residents indirectly.

Removing the GPS overlay does **not** by itself make these frames
non-personal-data. Pedestrians and number plates remain, and they are not
masked by the current crop.

Points for the accountable person to confirm:

1. **Lawful basis** for processing this footage for model development
   (legitimate interest, with a balancing test, is the usual route — it needs
   documenting, not assuming).
2. **Who owns the recordings** and whether that owner has authorised this use.
3. Whether **pedestrian faces and number plates must be blurred** in the
   stored training frames. This is cheap to add to `redact()` now and
   expensive to retrofit after labelling, so it is worth deciding before
   step 4. Note it would need a face/plate detector, and the obvious
   off-the-shelf ones carry their own licence questions.
4. **Retention**: how long the extracted frames and trained weights may be
   kept, and what happens to them at project end.
5. Whether a **DPIA** is required. Systematic monitoring of a publicly
   accessible area is one of the criteria that commonly triggers one.

### Rendered videos in `runs/demo/` and `runs/demo_clean/`

Overlay videos of the classical baseline are written to `runs/demo/*.mp4`
(`--style eval`: raw detections with hit/miss captions) and
`runs/demo_clean/*.mp4` (`--style clean`: a smoothed translucent region and a
status badge, intended for showing to people). They are produced by
`scripts/render_demo.py`, which applies
`redact()` to dashcam sources, so the **burned-in GPS/date/speed bar is not in
them** and the seconds in `extraction.exclude_ranges` (the vehicle occupant's
hand) are skipped.

They nonetheless **contain identifiable pedestrians and legible vehicle number
plates**, because those are in the footage and nothing masks them. Treat these
files exactly like the raw footage:

- They stay on this PC. `runs/` is gitignored. Do not attach them to a ticket,
  a chat, a slide deck or an email, and do not upload them for review.
- If any of this work has to be shown outside the team, the frames need faces
  and plates blurred first — see point 3 above, which is still open.
- The `clean` style is a *presentation* filter: it smooths and gates what is
  drawn, it does not change what was detected, and it does not remove anything
  from the frame. A clean video is exactly as sensitive as an eval one.
- `pexels_web_test.mp4` in either folder is the exception: it is public stock footage
  (Pexels Licence) and carries no privacy constraint. It still shows
  pedestrians and plates belonging to members of the public, so it is not a
  substitute for consent, only for licensing.

What is already enforced in code, pending those answers:

- All frames and derived data stay on this PC. No cloud labelling tools, no
  dataset sync, no telemetry (`src/ped_lane/privacy.py`, with tests).
- The burned-in GPS/date overlay never reaches disk (`src/ped_lane/overlay.py`
  `redact()`, with a deterministic test that fails if the crop is weakened).
- `data/` is gitignored, so frames cannot be committed by accident.
