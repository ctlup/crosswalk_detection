"""Rolling FPS and latency measurement.

Every model/device combination has to be reported with measured numbers, so the
same meter is used by the live loop and by the benchmark script.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Stats:
    count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    def as_line(self, label: str) -> str:
        return (
            f"{label}: mean {self.mean_ms:.1f} ms | p50 {self.p50_ms:.1f} | "
            f"p95 {self.p95_ms:.1f} | max {self.max_ms:.1f} (n={self.count})"
        )


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; `fraction` in [0, 1]. Empty input -> 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round(fraction * (len(ordered) - 1))
    return ordered[max(0, min(index, len(ordered) - 1))]


class RollingTimer:
    """Keeps the last `window` durations in milliseconds."""

    def __init__(self, window: int = 60):
        if window < 1:
            raise ValueError("window must be >= 1")
        self._samples: deque[float] = deque(maxlen=window)

    def add_ms(self, duration_ms: float) -> None:
        self._samples.append(float(duration_ms))

    def add_seconds(self, duration_s: float) -> None:
        self.add_ms(duration_s * 1000.0)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def last_ms(self) -> float:
        return self._samples[-1] if self._samples else 0.0

    def stats(self) -> Stats:
        values = list(self._samples)
        if not values:
            return Stats(0, 0.0, 0.0, 0.0, 0.0)
        return Stats(
            count=len(values),
            mean_ms=sum(values) / len(values),
            p50_ms=percentile(values, 0.5),
            p95_ms=percentile(values, 0.95),
            max_ms=max(values),
        )


class FpsCounter:
    """Throughput over a sliding wall-clock window.

    Measures the loop's real output rate, which is what the >= 10 FPS target
    means -- distinct from inference latency, which `RollingTimer` covers.
    """

    def __init__(self, window_seconds: float = 2.0, clock=time.perf_counter):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.window_seconds = window_seconds
        self._clock = clock
        self._ticks: deque[float] = deque()

    def tick(self) -> None:
        now = self._clock()
        self._ticks.append(now)
        cutoff = now - self.window_seconds
        while self._ticks and self._ticks[0] < cutoff:
            self._ticks.popleft()

    @property
    def fps(self) -> float:
        if len(self._ticks) < 2:
            return 0.0
        span = self._ticks[-1] - self._ticks[0]
        if span <= 0:
            return 0.0
        return (len(self._ticks) - 1) / span
