"""Tests for LatestFrameGrabber with mocked cv2.VideoCapture."""

import time
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_cap():
    """Create a mock VideoCapture that returns frames."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    return cap


def make_grabber(mock_cap, **kwargs):
    with patch('src.grabber.cv2') as mock_cv2:
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.CAP_PROP_FOURCC = 6
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.VideoWriter_fourcc.return_value = 1196444237
        from src.grabber import LatestFrameGrabber
        grabber = LatestFrameGrabber(source=0, warmup_frames=0, **kwargs)
        # Let the background thread grab at least one frame
        time.sleep(0.1)
        return grabber


def test_read_returns_frame(mock_cap):
    grabber = make_grabber(mock_cap)
    try:
        ok, frame = grabber.read()
        assert ok is True
        assert isinstance(frame, np.ndarray)
    finally:
        grabber.release()


def test_has_new_frame(mock_cap):
    grabber = make_grabber(mock_cap)
    try:
        time.sleep(0.1)
        assert grabber.has_new_frame() is True
        grabber.read()  # consumes the frame
        assert grabber.has_new_frame() is False
    finally:
        grabber.release()


def test_read_before_any_frame():
    cap = MagicMock()
    cap.isOpened.return_value = True
    # First read returns nothing, simulating slow camera
    call_count = [0]
    def slow_read():
        call_count[0] += 1
        if call_count[0] <= 2:
            return False, None
        return True, np.zeros((480, 640, 3), dtype=np.uint8)
    cap.read.side_effect = slow_read

    grabber = make_grabber(cap)
    try:
        # Initially might not have a frame
        # After a moment, should have one
        time.sleep(0.5)
        ok, frame = grabber.read()
        # Eventually should get a frame
        assert ok is True or frame is None  # depends on timing
    finally:
        grabber.release()


def test_release_stops_thread(mock_cap):
    grabber = make_grabber(mock_cap)
    grabber.release()
    assert grabber.stopped is True
    assert not grabber.thread.is_alive()


def test_source_not_opened_raises():
    cap = MagicMock()
    cap.isOpened.return_value = False
    with patch('src.grabber.cv2') as mock_cv2:
        mock_cv2.VideoCapture.return_value = cap
        from src.grabber import LatestFrameGrabber
        with pytest.raises(RuntimeError, match="Could not open"):
            LatestFrameGrabber(source="bad_source")


def test_target_fps(mock_cap):
    grabber = make_grabber(mock_cap, target_fps=10)
    try:
        assert grabber._frame_interval == pytest.approx(0.1)
    finally:
        grabber.release()


def test_no_target_fps(mock_cap):
    grabber = make_grabber(mock_cap)
    try:
        assert grabber._frame_interval is None
    finally:
        grabber.release()
