"""OpenCV polygon labelling tool for the single class "crosswalk".

No new dependencies, no web service, nothing leaves this PC. Writes YOLO-seg
label files next to the frames.

Coordinates are normalised against the **saved (redacted) frame**, which is
what you see here. `crop_top_fraction` is applied later, during dataset
conversion, which shifts and clips the polygons -- that is deliberate, so
retuning the crop never invalidates this labelling.

Keys
  left click        add a vertex
  right click / u   undo last vertex
  Enter / c         close the current polygon and keep it
  d                 delete the last kept polygon
  x                 mark frame reviewed with NO crosswalk (an explicit negative)
  n / Space / ->    next frame        p / <-   previous frame
  j                 jump to the next unreviewed frame
  a                 adopt the classical baseline's proposal as a polygon
  h                 toggle help      q / Esc  save and quit

    python scripts/label_crosswalk.py --help
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ped_lane import classes  # noqa: E402
from ped_lane.config import load_config  # noqa: E402
from ped_lane.privacy import enforce_local_only  # noqa: E402
from ped_lane.stripes import (  # noqa: E402
    StripeParams,
    detect,
    road_band,
    vanishing_point_for,
)

CROSSWALK_CLASS = classes.CROSSWALK
STATE_NAME = "label_state.json"
WINDOW = "label crosswalk"


# --- pure logic, unit tested ------------------------------------------------

def to_yolo_seg(polygon: np.ndarray, width: int, height: int, class_id: int = 0) -> str:
    """One YOLO-seg label line: `class x1 y1 x2 y2 ...`, normalised and clipped."""
    if len(polygon) < 3:
        raise ValueError("a polygon needs at least 3 points")
    parts = [str(class_id)]
    for x, y in np.asarray(polygon, dtype=np.float64).reshape(-1, 2):
        nx = min(max(x / width, 0.0), 1.0)
        ny = min(max(y / height, 0.0), 1.0)
        parts.append(f"{nx:.6f}")
        parts.append(f"{ny:.6f}")
    return " ".join(parts)


def from_yolo_seg(line: str, width: int, height: int) -> tuple[int, np.ndarray]:
    """Inverse of `to_yolo_seg`, for reopening a frame that was already labelled."""
    fields = line.split()
    if len(fields) < 7 or len(fields) % 2 == 0:
        raise ValueError(f"malformed YOLO-seg line: {line!r}")
    class_id = int(fields[0])
    values = [float(v) for v in fields[1:]]
    points = np.array(
        [[values[i] * width, values[i + 1] * height] for i in range(0, len(values), 2)],
        dtype=np.float32,
    )
    return class_id, points


def label_path_for(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / (image_path.stem + ".txt")


@dataclass
class FrameLabels:
    polygons: list[np.ndarray] = field(default_factory=list)
    reviewed: bool = False

    @property
    def is_negative(self) -> bool:
        return self.reviewed and not self.polygons


# --- the tool ---------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python scripts/label_crosswalk.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__[__doc__.index("Keys"):],
    )
    ap.add_argument("--frames", required=True, help="directory of extracted frames")
    ap.add_argument("--labels", default=None, help="label dir (default <frames>/../labels/<name>)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--start", type=int, default=0, help="frame index to open at")
    ap.add_argument(
        "--only-candidates",
        type=float,
        default=None,
        metavar="SCORE",
        help="only show frames the baseline scores at or above SCORE",
    )
    ap.add_argument("--width", type=int, default=1600, help="display width")
    return ap.parse_args(argv)


def _baseline_polygon(image: np.ndarray, source_height: int) -> np.ndarray | None:
    vp_x, vp_y = vanishing_point_for(image.shape[1], source_height)
    band, offset = road_band(image, vp_y)
    result = detect(band, (vp_x, vp_y - offset), StripeParams())
    if not result.found:
        return None
    polygon = result.groups[0].polygon.astype(np.float32).copy()
    polygon[:, 1] += offset  # back into full saved-frame coordinates
    return polygon


def _draw(canvas, labels: FrameLabels, pending: list, scale: float, info: str, show_help: bool):
    for polygon in labels.polygons:
        pts = (np.asarray(polygon, np.float32) * scale).astype(np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], (60, 220, 255))
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)
        cv2.polylines(canvas, [pts], True, (60, 220, 255), 2)
    if pending:
        pts = (np.asarray(pending, np.float32) * scale).astype(np.int32)
        if len(pts) > 1:
            cv2.polylines(canvas, [pts], False, (90, 255, 90), 2)
        for point in pts:
            cv2.circle(canvas, tuple(point), 4, (90, 255, 90), -1)
    bar = canvas.shape[0] - 30
    cv2.rectangle(canvas, (0, bar - 8), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
    cv2.putText(canvas, info, (10, bar + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    if show_help:
        for i, line in enumerate(__doc__[__doc__.index("Keys"):].strip().splitlines()):
            y = 26 + i * 20
            cv2.rectangle(canvas, (8, y - 15), (560, y + 5), (0, 0, 0), -1)
            cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (120, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main(argv: list[str] | None = None) -> int:
    enforce_local_only()
    args = parse_args(argv)
    load_config(args.config)  # fail fast on a broken config

    frames_dir = Path(args.frames)
    manifest_path = frames_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[error] no manifest.json in {frames_dir}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_height = manifest["source"]["height"]

    labels_dir = Path(args.labels) if args.labels else frames_dir.parent.parent / "labels" / frames_dir.name
    labels_dir.mkdir(parents=True, exist_ok=True)
    state_path = labels_dir / STATE_NAME
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}

    records = manifest["frames"]
    if args.only_candidates is not None:
        print(f"[info] filtering to baseline score >= {args.only_candidates} ...")
        keep = []
        for record in records:
            image = cv2.imread(str(frames_dir / record["file"]))
            if image is None:
                continue
            polygon = _baseline_polygon(image, source_height)
            if polygon is not None:
                keep.append(record)
        records = keep or records
        print(f"[info] {len(records)} frame(s) to review")

    index = max(0, min(args.start, len(records) - 1))
    show_help = True
    pending: list[tuple[float, float]] = []
    scale = 1.0
    loaded: dict[str, FrameLabels] = {}

    def load(record) -> FrameLabels:
        name = record["file"]
        if name in loaded:
            return loaded[name]
        image_path = frames_dir / name
        path = label_path_for(image_path, labels_dir)
        entry = FrameLabels(reviewed=bool(state.get(name, {}).get("reviewed")))
        if path.exists():
            img = cv2.imread(str(image_path))
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    _, pts = from_yolo_seg(line, img.shape[1], img.shape[0])
                    entry.polygons.append(pts)
            entry.reviewed = True
        loaded[name] = entry
        return entry

    def save(record, image_shape) -> None:
        name = record["file"]
        entry = loaded[name]
        path = label_path_for(frames_dir / name, labels_dir)
        height, width = image_shape[:2]
        lines = [to_yolo_seg(p, width, height, CROSSWALK_CLASS) for p in entry.polygons]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        state[name] = {"reviewed": entry.reviewed, "polygons": len(entry.polygons)}

    def on_mouse(event, x, y, flags, _):
        nonlocal pending
        if event == cv2.EVENT_LBUTTONDOWN:
            pending.append((x / scale, y / scale))
        elif event == cv2.EVENT_RBUTTONDOWN and pending:
            pending.pop()

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)

    try:
        while True:
            record = records[index]
            image = cv2.imread(str(frames_dir / record["file"]))
            if image is None:
                index = (index + 1) % len(records)
                continue
            entry = load(record)
            scale = args.width / image.shape[1]
            canvas = cv2.resize(image, (args.width, int(image.shape[0] * scale)))

            reviewed = sum(1 for r in records if state.get(r["file"], {}).get("reviewed"))
            status = "NEGATIVE" if entry.is_negative else (
                f"{len(entry.polygons)} polygon(s)" if entry.polygons else "unreviewed"
            )
            info = (
                f"[{index + 1}/{len(records)}] t={record['seconds']:.0f}s  {status}  "
                f"| reviewed {reviewed}/{len(records)}  | h=help  q=save+quit"
            )
            cv2.imshow(WINDOW, _draw(canvas, entry, pending, scale, info, show_help))

            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                save(record, image.shape)
                break
            elif key in (13, ord("c")):
                if len(pending) >= 3:
                    entry.polygons.append(np.array(pending, dtype=np.float32))
                    entry.reviewed = True
                    pending = []
                    save(record, image.shape)
            elif key == ord("u") and pending:
                pending.pop()
            elif key == ord("d") and entry.polygons:
                entry.polygons.pop()
                save(record, image.shape)
            elif key == ord("x"):
                entry.polygons.clear()
                entry.reviewed = True
                pending = []
                save(record, image.shape)
            elif key == ord("a"):
                polygon = _baseline_polygon(image, source_height)
                if polygon is not None:
                    entry.polygons.append(polygon)
                    entry.reviewed = True
                    save(record, image.shape)
            elif key == ord("h"):
                show_help = not show_help
            elif key in (ord("n"), ord(" "), 83):
                save(record, image.shape)
                pending = []
                index = min(index + 1, len(records) - 1)
            elif key in (ord("p"), 81):
                save(record, image.shape)
                pending = []
                index = max(index - 1, 0)
            elif key == ord("j"):
                save(record, image.shape)
                pending = []
                nxt = next(
                    (i for i in range(index + 1, len(records))
                     if not state.get(records[i]["file"], {}).get("reviewed")),
                    index,
                )
                index = nxt
    finally:
        cv2.destroyAllWindows()
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    positives = sum(1 for v in state.values() if v.get("polygons"))
    negatives = sum(1 for v in state.values() if v.get("reviewed") and not v.get("polygons"))
    print(f"\n[info] labels in {labels_dir}")
    print(f"[info] reviewed {positives + negatives}/{len(records)}  "
          f"positives {positives}  explicit negatives {negatives}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
