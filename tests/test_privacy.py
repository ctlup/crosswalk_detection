"""Frames carry a burned-in GPS/date overlay, so leaving this PC is a defect."""
import os

import pytest

from ped_lane.privacy import enforce_local_only


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in list(os.environ):
        if key.split("_")[0] in {"HF", "WANDB", "COMET", "CLEARML", "NEPTUNE", "MLFLOW", "YOLO", "DVC", "SENTRY", "TRANSFORMERS"}:
            monkeypatch.delenv(key, raising=False)


def test_telemetry_is_disabled():
    enforce_local_only()
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert os.environ["WANDB_MODE"] == "disabled"
    assert os.environ["COMET_MODE"] == "DISABLED"
    assert os.environ["CLEARML_OFFLINE_MODE"] == "1"
    assert os.environ["NEPTUNE_MODE"] == "offline"


def test_hub_stays_reachable_by_default():
    # Blocking the hub outright would break a legitimate first download; only
    # telemetry is off unless offline mode is asked for.
    enforce_local_only()
    assert "HF_HUB_OFFLINE" not in os.environ


def test_offline_mode_blocks_the_hub_entirely():
    enforce_local_only(offline_hub=True)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_an_explicit_user_setting_is_not_overridden(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "online")
    enforce_local_only()
    assert os.environ["WANDB_MODE"] == "online"


def test_calling_it_twice_is_harmless():
    enforce_local_only()
    enforce_local_only()
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
