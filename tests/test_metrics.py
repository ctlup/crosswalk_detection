import pytest

from ped_lane.metrics import FpsCounter, RollingTimer, percentile


def test_percentile_bounds():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 1.0) == 5.0


def test_percentile_is_order_independent():
    assert percentile([5.0, 1.0, 3.0], 0.5) == 3.0


def test_percentile_of_empty_is_zero():
    assert percentile([], 0.95) == 0.0


def test_empty_timer_reports_zeroes():
    stats = RollingTimer().stats()
    assert stats.count == 0
    assert stats.mean_ms == 0.0
    assert stats.p95_ms == 0.0


def test_timer_statistics():
    timer = RollingTimer(window=10)
    for value in (10.0, 20.0, 30.0):
        timer.add_ms(value)

    stats = timer.stats()
    assert stats.count == 3
    assert stats.mean_ms == pytest.approx(20.0)
    assert stats.p50_ms == 20.0
    assert stats.max_ms == 30.0
    assert timer.last_ms == 30.0


def test_timer_window_drops_old_samples():
    timer = RollingTimer(window=3)
    for value in (1.0, 2.0, 3.0, 4.0):
        timer.add_ms(value)

    stats = timer.stats()
    assert stats.count == 3
    assert stats.mean_ms == pytest.approx(3.0)


def test_add_seconds_converts_to_milliseconds():
    timer = RollingTimer()
    timer.add_seconds(0.25)
    assert timer.last_ms == pytest.approx(250.0)


def test_window_must_be_positive():
    with pytest.raises(ValueError):
        RollingTimer(window=0)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_fps_needs_two_ticks():
    clock = FakeClock()
    counter = FpsCounter(window_seconds=5.0, clock=clock)
    assert counter.fps == 0.0
    counter.tick()
    assert counter.fps == 0.0


def test_fps_over_even_spacing():
    clock = FakeClock()
    counter = FpsCounter(window_seconds=10.0, clock=clock)
    for _ in range(11):
        counter.tick()
        clock.now += 0.1  # 10 FPS
    assert counter.fps == pytest.approx(10.0, rel=1e-6)


def test_fps_window_forgets_an_old_stall():
    clock = FakeClock()
    counter = FpsCounter(window_seconds=1.0, clock=clock)

    counter.tick()        # old tick, should fall out of the window
    clock.now += 5.0
    for _ in range(5):
        counter.tick()
        clock.now += 0.05  # 20 FPS

    assert counter.fps == pytest.approx(20.0, rel=1e-6)


def test_fps_window_must_be_positive():
    with pytest.raises(ValueError):
        FpsCounter(window_seconds=0.0)
