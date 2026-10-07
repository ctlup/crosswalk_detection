from pathlib import Path

from ped_lane.config import Config, resolve_source


def test_defaults_when_sections_missing():
    cfg = Config.from_dict({})
    assert cfg.model.imgsz == 640
    assert cfg.model.device == "auto"
    assert cfg.output.save_video is None
    assert cfg.temporal.smoothing_frames == 3


def test_none_sections_do_not_crash():
    cfg = Config.from_dict({"model": None, "output": None, "dashcam": None})
    assert cfg.model.conf == 0.35
    assert cfg.dashcam.crop_bottom_fraction == 0.18


def test_save_video_null_forms_become_none():
    for value in (None, "", "null"):
        assert Config.from_dict({"output": {"save_video": value}}).output.save_video is None


def test_save_video_path_is_parsed():
    cfg = Config.from_dict({"output": {"save_video": "runs/out.mp4"}})
    assert cfg.output.save_video == Path("runs/out.mp4")


def test_types_are_coerced_from_yaml_strings():
    cfg = Config.from_dict({"model": {"imgsz": "512", "conf": "0.5"}})
    assert cfg.model.imgsz == 512
    assert cfg.model.conf == 0.5


def test_resolve_source_prefers_cli_then_env_then_config(monkeypatch):
    cfg = Config.from_dict({"source": "samples/from_config.mp4"})

    monkeypatch.delenv("CAMERA_URL", raising=False)
    assert resolve_source(None, cfg) == "samples/from_config.mp4"

    monkeypatch.setenv("CAMERA_URL", "rtsp://host/stream")
    assert resolve_source(None, cfg) == "rtsp://host/stream"
    assert resolve_source("samples/cli.mp4", cfg) == "samples/cli.mp4"


def test_resolve_source_digits_become_camera_index(monkeypatch):
    monkeypatch.delenv("CAMERA_URL", raising=False)
    cfg = Config.from_dict({})
    assert resolve_source("0", cfg) == 0
    assert isinstance(resolve_source("1", cfg), int)
    assert resolve_source("samples/a.mp4", cfg) == "samples/a.mp4"
