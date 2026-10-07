# Footage recording plan

For whoever records the next batch of dashcam footage.

**Why this exists:** the first clip (180 s, one town, one sunny morning)
contains **4 distinct crossings**. That is a demo, not a dataset. A model
trained on it would memorise four locations rather than learn what a crossing
looks like. What follows is what the next recordings need to contain.

---

## Target

| | |
|---|---|
| **Distinct crossings** | **100 minimum, 300 preferred** |
| Distinct locations | 30+ (a crossing filmed twice is not two crossings) |
| Night / low light | at least 20% of crossings |
| Wet or rain | at least 15% of crossings |
| Low sun / glare | at least 10% of crossings |

**Count crossings, not minutes.** An hour of empty ring road adds nothing. Ten
minutes through a town centre with twelve crossings is worth more than the hour.

---

## Keep constant

- **Same camera, same mount, same position on the windscreen.** The crop
  settings, the ignore region and the measured vanishing point in
  `config.yaml` are all calibrated to this exact rig. Move the camera and
  every one of them has to be re-measured.
- **Same resolution and frame rate** (1920x1080, ~30 fps).
- **Windscreen clean.** Smears and glare cost more than they look like they do.

## Vary deliberately

Tick these off; do not assume a long drive covers them by accident.

- [ ] **Time of day** — morning, midday, late afternoon, dusk, **after dark**
- [ ] **Weather** — dry, overcast, **rain**, wet road after rain, low winter sun
- [ ] **Sun angle** — including driving *into* a low sun, and deep shade under trees
- [ ] **Location type** — town centre, residential, school zone, retail park,
      village high street, industrial estate
- [ ] **Approach geometry** — straight-on, approaching round a bend,
      **turning across a crossing**, stopped at one, passing one on a side road
- [ ] **Crossing condition** — fresh paint, **heavily faded**, partly resurfaced,
      patched, wet and reflective, partly covered by a vehicle or pedestrian
- [ ] **Crossing type** — plain zebra, raised/speed-table, with central island,
      staggered, at a signalled junction, with cycle crossing alongside
- [ ] **Speed** — slow town speeds and faster approaches (motion blur differs)

## The two cases that matter most

The first clip had the blue pedestrian-crossing sign in view at *every single*
crossing. A model trained on that will learn the sign, not the paint. Record
both of these deliberately:

- [ ] **Crossings with NO blue sign** — unsigned, obscured, missing or
      vandalised sign. Target at least 20 of these.
- [ ] **The blue sign with NO crossing in view** — the sign on approach before
      the paint is visible, a sign facing a side road, a sign where the paint
      has worn away entirely. Target at least 20 of these.

Also worth collecting on purpose, as hard negatives:

- [ ] Stop lines and give-way triangles
- [ ] Lane arrows, dashed centre lines, hatched areas
- [ ] Cycle-lane markings and bus-lane markings
- [ ] Road works with temporary white marking
- [ ] Manhole covers and drain gratings (the classical baseline fired on one
      while ignoring a crossing that filled the frame — see
      `docs/BASELINE_RESULT.md`)
- [ ] Shadows of railings across the road — these look like zebra stripes

## Privacy — please follow

- [ ] **Turn the GPS/date/speed stamp OFF in the camera settings** if it can be
      turned off. We currently crop it away, which costs the bottom 10% of
      every frame. Turning it off at source gets that back.
- [ ] **Keep vehicle occupants out of shot** — hands, arms, phone mounts,
      air fresheners, anything hanging from the mirror. Two frames in the
      first clip had to be discarded because a hand entered the top of frame.
- [ ] Do not record private land, driveways or anywhere not a public road.
- [ ] Hand the footage over on physical media or a local share. **Do not upload
      it anywhere.** See `LICENSING.md` section 4.

Frames will still contain identifiable pedestrians and legible number plates.
That is unavoidable when filming public roads, and it is why the data
protection questions in `LICENSING.md` need answering before this scales up.

## Practical notes

- Record in continuous runs and keep the original files; we extract at 1 fps
  and keep timestamps, so a continuous clip is more useful than pre-cut ones.
- Note roughly where and when each run was recorded. Train/validation splits
  are made **by time segment and by location**, never by random frame, so we
  need to know which footage came from where.
- A run that turns out to contain no crossings is still worth keeping: it is
  negative data, and we have far less of it than we need.
