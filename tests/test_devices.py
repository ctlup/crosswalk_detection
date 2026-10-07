import pytest

from ped_lane.devices import resolve_device


def test_auto_picks_cuda_when_available():
    assert resolve_device("auto", cuda_available=lambda: True) == "cuda:0"


def test_auto_falls_back_to_cpu():
    assert resolve_device("auto", cuda_available=lambda: False) == "cpu"


def test_auto_never_returns_none():
    # Ultralytics treats None as "decide for me", which silently chose the CPU
    # on this machine; the device must always be explicit.
    for available in (True, False):
        assert resolve_device("auto", cuda_available=lambda: available) is not None


def test_explicit_specs_are_normalised():
    assert resolve_device("cpu", cuda_available=lambda: True) == "cpu"
    assert resolve_device("cuda", cuda_available=lambda: False) == "cuda:0"
    assert resolve_device("0", cuda_available=lambda: False) == "cuda:0"
    assert resolve_device("1") == "cuda:1"
    assert resolve_device("cuda:1") == "cuda:1"


def test_case_and_whitespace_tolerated():
    assert resolve_device("  CPU  ", cuda_available=lambda: True) == "cpu"
    assert resolve_device("AUTO", cuda_available=lambda: True) == "cuda:0"


def test_empty_spec_is_auto():
    assert resolve_device("", cuda_available=lambda: False) == "cpu"


def test_unknown_spec_raises():
    with pytest.raises(ValueError, match="unsupported device"):
        resolve_device("tpu")
