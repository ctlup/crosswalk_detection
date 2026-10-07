import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

_SPEC = importlib.util.spec_from_file_location(
    "extract_frames", Path(__file__).resolve().parents[1] / "scripts" / "extract_frames.py"
)
extract_frames = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(extract_frames)


def test_frame_step_for_common_rates():
    assert extract_frames.frame_step(29.97, 1.0) == 30
    assert extract_frames.frame_step(25.0, 1.0) == 25
    assert extract_frames.frame_step(30.0, 2.0) == 15
    assert extract_frames.frame_step(30.0, 0.5) == 60


def test_frame_step_never_below_one():
    # Asking for more frames than the source has must not stall or rewind.
    assert extract_frames.frame_step(10.0, 30.0) == 1


def test_frame_step_rejects_nonsense():
    with pytest.raises(ValueError):
        extract_frames.frame_step(0, 1.0)
    with pytest.raises(ValueError):
        extract_frames.frame_step(30.0, 0)


def test_frame_name_is_sortable_and_carries_the_timestamp():
    name = extract_frames.frame_name("dashcam", 3600, 120.12)
    assert name == "dashcam_f003600_t00120.12s.jpg"
    names = [extract_frames.frame_name("c", i * 30, i) for i in (0, 5, 12, 120)]
    assert names == sorted(names)  # lexical order == temporal order


@pytest.fixture
def tiny_clip(tmp_path):
    """A 90-frame 30 fps clip with a bright bar in the burned-in overlay band."""
    path = tmp_path / "tiny.mp4"
    h, w = 1080, 320
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    assert writer.isOpened()
    for i in range(90):
        f = np.full((h, w, 3), 30 + i, dtype=np.uint8)
        f[int(h * 0.9565):int(h * 0.9769), :] = 255   # stand-in overlay text
        f[int(h * 0.76):, int(w * 0.87):] = 240       # stand-in windscreen clutter
        writer.write(f)
    writer.release()
    return path


def test_extraction_redacts_and_writes_a_manifest(tmp_path, tiny_clip):
    out = tmp_path / "frames"
    rc = extract_frames.main(
        ["--source", str(tiny_clip), "--out", str(out), "--fps", "1", "--config", "config.yaml"]
    )
    assert rc == 0

    jpgs = sorted(out.glob("*.jpg"))
    assert len(jpgs) == 3  # 90 frames at 30 fps -> 3 s

    for jpg in jpgs:
        img = cv2.imread(str(jpg))
        assert img.shape[0] == 1080 - int(1080 * 0.10)   # bottom crop applied
        assert img[:, :, :].max() < 250                  # the bright overlay bar is gone
        assert img[int(1080 * 0.80), int(320 * 0.95)].sum() == 0  # ignore region blanked


def test_manifest_records_what_is_needed_to_apply_crop_top_later(tmp_path, tiny_clip):
    out = tmp_path / "frames"
    assert extract_frames.main(
        ["--source", str(tiny_clip), "--out", str(out), "--fps", "1", "--config", "config.yaml"]
    ) == 0

    manifest = json.loads((out / extract_frames.MANIFEST_NAME).read_text(encoding="utf-8"))

    # The original height is the thing that must survive: top_crop_rows needs it.
    assert manifest["source"]["height"] == 1080
    assert manifest["frame_size"]["height"] == 972
    later = manifest["apply_at_train_and_inference"]
    assert later["crop_top_fraction"] == 0.40
    assert later["top_crop_rows"] == 432
    assert len(manifest["frames"]) == 3
    assert manifest["redaction_applied"]["crop_bottom_fraction"] == 0.10


def test_refuses_to_overwrite_frames_without_the_flag(tmp_path, tiny_clip):
    out = tmp_path / "frames"
    args = ["--source", str(tiny_clip), "--out", str(out), "--fps", "1", "--config", "config.yaml"]
    assert extract_frames.main(args) == 0
    assert extract_frames.main(args) == 2            # second run blocked
    assert extract_frames.main(args + ["--overwrite"]) == 0


def test_missing_source_is_an_error_not_a_crash(tmp_path):
    assert extract_frames.main(
        ["--source", str(tmp_path / "nope.mp4"), "--out", str(tmp_path / "o")]
    ) == 2


def test_exclude_ranges_drop_frames_and_purge_stale_ones(tmp_path, tiny_clip, monkeypatch):
    """A newly added exclude_range must remove a frame an earlier run wrote.

    Without the purge, the frame we are trying to delete survives on disk --
    the exact failure this guards against.
    """
    import yaml

    out = tmp_path / "frames"
    base = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

    open_cfg = tmp_path / "open.yaml"
    base["extraction"] = {"exclude_ranges": []}
    open_cfg.write_text(yaml.safe_dump(base), encoding="utf-8")
    assert extract_frames.main(
        ["--source", str(tiny_clip), "--out", str(out), "--fps", "1", "--config", str(open_cfg)]
    ) == 0
    assert len(list(out.glob("*.jpg"))) == 3

    tight_cfg = tmp_path / "tight.yaml"
    base["extraction"] = {"exclude_ranges": [[0.5, 1.5]]}
    tight_cfg.write_text(yaml.safe_dump(base), encoding="utf-8")
    assert extract_frames.main(
        ["--source", str(tiny_clip), "--out", str(out), "--fps", "1",
         "--config", str(tight_cfg), "--overwrite"]
    ) == 0

    remaining = sorted(p.name for p in out.glob("*.jpg"))
    assert len(remaining) == 2
    manifest = json.loads((out / extract_frames.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert {r["file"] for r in manifest["frames"]} == set(remaining)  # nothing stale
    assert manifest["extraction"]["excluded_seconds"] == [1.0]
