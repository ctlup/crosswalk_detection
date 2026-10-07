"""Inference device selection.

`config.model.device: auto` must become a concrete device string: Ultralytics
treats `None` as "decide for me", which silently moved work to the CPU on this
machine. Keeping the choice explicit also means the benchmark tables can name
the device they measured.
"""
from __future__ import annotations

from typing import Callable


def resolve_device(
    spec: str,
    cuda_available: Callable[[], bool] | None = None,
) -> str:
    """Turn a config device spec into a device string Ultralytics accepts.

    ``auto`` becomes ``cuda:0`` when a CUDA GPU is usable and ``cpu`` otherwise.
    Explicit specs pass through normalised: ``"0"`` -> ``"cuda:0"``.
    `cuda_available` is injectable so this stays testable without a GPU.
    """
    text = (spec or "auto").strip().lower()

    if text in ("auto", ""):
        if cuda_available is None:
            cuda_available = _torch_cuda_available
        return "cuda:0" if cuda_available() else "cpu"

    if text in ("cpu", "mps"):
        return text
    if text == "cuda":
        return "cuda:0"
    if text.isdigit():
        return f"cuda:{int(text)}"
    if text.startswith("cuda:") and text[5:].isdigit():
        return text

    raise ValueError(
        f"unsupported device {spec!r}; use 'auto', 'cpu', 'cuda', "
        f"a GPU index like '0', or 'cuda:0'"
    )


def _torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def describe_device(device: str) -> str:
    """Human-readable device label for the HUD and benchmark output."""
    if device == "cpu":
        import platform

        return f"CPU ({platform.processor() or 'unknown'})"
    try:
        import torch

        index = int(device.split(":")[1]) if ":" in device else 0
        return f"{device} ({torch.cuda.get_device_name(index)})"
    except Exception:
        return device
