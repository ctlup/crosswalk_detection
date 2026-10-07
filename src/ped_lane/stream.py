"""Frame sources.

Live sources (USB, RTSP, dashcam Wi-Fi) need the latest frame and must drop
anything stale, otherwise OpenCV buffers frames while inference runs and latency
grows without bound. Recorded clips need the opposite: every frame, in order,
exactly once, so pseudo-labelling and evaluation are reproducible. Those are two
different behaviours, so they are two classes behind one protocol.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np


@runtime_checkable
class FrameSource(Protocol):
    """A source of frames. `read` returns ``(sequence_number, frame_or_None)``."""

    is_live: bool

    def read(self) -> tuple[int, np.ndarray | None]: ...

    def stop(self) -> None: ...

    @property
    def exhausted(self) -> bool: ...


class VideoFileSource:
    """Sequential reader for a recorded clip.

    Blocking and single-threaded on purpose: the loop should process every frame
    of a clip rather than skip ahead, so results on a sample clip do not depend
    on how fast the machine happens to be.
    """

    is_live = False

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"cannot open video file: {self.path}")
        self._seq = 0
        self._exhausted = False

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def source_fps(self) -> float:
        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        return fps if fps > 0 else 0.0

    def read(self) -> tuple[int, np.ndarray | None]:
        if self._exhausted:
            return self._seq, None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._exhausted = True
            return self._seq, None
        self._seq += 1
        return self._seq, frame

    def stop(self) -> None:
        self._cap.release()


class LatestFrameGrabber:
    """Threaded reader for live sources; keeps only the most recent frame.

    Reconnects on failure, so a dashcam Wi-Fi drop or an RTSP hiccup does not
    end the run.
    """

    is_live = True

    def __init__(self, source: str | int, reconnect_seconds: float = 3.0):
        self.source = source
        self.reconnect_seconds = reconnect_seconds
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._captured_at = 0.0
        self._reconnects = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "LatestFrameGrabber":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    @property
    def exhausted(self) -> bool:
        return False  # a live source is never done; it reconnects instead

    @property
    def reconnects(self) -> int:
        return self._reconnects

    def read(self) -> tuple[int, np.ndarray | None]:
        """Latest frame and its sequence number; the sequence lets callers skip
        duplicates when inference is faster than the camera."""
        with self._lock:
            if self._frame is None:
                return self._seq, None
            return self._seq, self._frame.copy()

    def read_with_age(self) -> tuple[int, np.ndarray | None, float]:
        """As `read`, plus how long ago the frame was captured, in seconds.

        This is the capture-to-display part of end-to-end latency, which the
        inference timer alone does not see.
        """
        with self._lock:
            if self._frame is None:
                return self._seq, None, 0.0
            return self._seq, self._frame.copy(), time.perf_counter() - self._captured_at

    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self) -> None:
        cap = self._open()
        while not self._stop.is_set():
            if not cap.isOpened():
                cap.release()
                self._reconnects += 1
                time.sleep(self.reconnect_seconds)
                cap = self._open()
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                self._reconnects += 1
                time.sleep(self.reconnect_seconds)
                cap = self._open()
                continue
            with self._lock:
                self._frame = frame
                self._captured_at = time.perf_counter()
                self._seq += 1
        cap.release()


def is_file_source(source: str | int) -> bool:
    """True when `source` names a readable local file rather than a live device."""
    if isinstance(source, int):
        return False
    text = str(source)
    if "://" in text:
        return False
    return Path(text).is_file()


def open_source(source: str | int, reconnect_seconds: float = 3.0) -> FrameSource:
    """Build the right `FrameSource` for `source`."""
    if is_file_source(source):
        return VideoFileSource(source)
    return LatestFrameGrabber(source, reconnect_seconds).start()
