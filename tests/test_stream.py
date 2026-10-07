import cv2
import numpy as np
import pytest

from ped_lane.stream import VideoFileSource, is_file_source, open_source


@pytest.fixture
def clip(tmp_path):
    """A tiny 6-frame mp4 so the file path is exercised without a real dashcam clip."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    assert writer.isOpened()
    for i in range(6):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()
    return path


def test_is_file_source_distinguishes_the_three_input_kinds(clip):
    assert is_file_source(str(clip))
    assert not is_file_source(0)
    assert not is_file_source("rtsp://camera:554/stream1")
    assert not is_file_source("samples/does_not_exist.mp4")


def test_file_source_reads_every_frame_in_order_then_reports_eof(clip):
    source = VideoFileSource(clip)
    try:
        assert not source.is_live
        seqs = []
        while True:
            seq, frame = source.read()
            if frame is None:
                break
            seqs.append(seq)
        assert seqs == [1, 2, 3, 4, 5, 6]
        assert source.exhausted
    finally:
        source.stop()


def test_reading_past_eof_keeps_returning_none(clip):
    source = VideoFileSource(clip)
    try:
        while source.read()[1] is not None:
            pass
        assert source.read()[1] is None
        assert source.read()[1] is None
    finally:
        source.stop()


def test_file_source_exposes_fps_for_the_video_writer(clip):
    source = VideoFileSource(clip)
    try:
        assert source.source_fps == pytest.approx(10.0, abs=0.5)
    finally:
        source.stop()


def test_missing_file_raises_rather_than_hanging_on_reconnect():
    with pytest.raises(FileNotFoundError):
        VideoFileSource("samples/definitely_not_here.mp4")


def test_open_source_picks_the_sequential_reader_for_files(clip):
    source = open_source(str(clip))
    try:
        assert isinstance(source, VideoFileSource)
    finally:
        source.stop()
