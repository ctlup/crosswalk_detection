"""Keep frames and derived data on this machine.

The dashcam burns a GPS/date overlay into every frame, so a frame is personal
data. Model libraries default to phoning home with usage events, and
Ultralytics additionally offers to sync datasets to a hosted account. None of
that should ever carry this footage off the PC, so the defaults are turned off
before those libraries are imported.

Call `enforce_local_only()` first thing in any entry point that touches frames.
It only changes this process (and the local Ultralytics settings file); nothing
here needs admin rights.
"""
from __future__ import annotations

import os

_ENV_DEFAULTS = {
    # Hugging Face: no telemetry pings; downloads of weights still work.
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    # Ultralytics / third-party analytics.
    "YOLO_OFFLINE": "True",
    "SENTRY_DSN": "",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
    "COMET_MODE": "DISABLED",
    "CLEARML_OFFLINE_MODE": "1",
    "NEPTUNE_MODE": "offline",
    "MLFLOW_TRACKING_URI": "",
    "DVC_NO_ANALYTICS": "1",
}


def enforce_local_only(offline_hub: bool = False) -> None:
    """Disable telemetry and experiment-tracker uploads for this process.

    `offline_hub=True` additionally blocks every Hugging Face network call, for
    runs after the weights are cached. It is off by default so the first
    download still works.
    """
    for key, value in _ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    if offline_hub:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    _disable_ultralytics_sync()


def _disable_ultralytics_sync() -> None:
    """Turn off Ultralytics analytics and hosted-dataset sync, if it is installed."""
    try:
        from ultralytics import settings
    except Exception:
        return
    try:
        wanted = {"sync": False}
        if settings.get("sync") is not False:
            settings.update(wanted)
    except Exception:
        # A settings file we cannot write is not a reason to abort a run; the
        # env vars above already cover the network paths.
        pass
