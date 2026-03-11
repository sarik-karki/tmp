"""Extended tracker tests: external detector mode, frame skipping, IoU."""

import time
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import numpy as np
import pytest
from src.tracker import VehicleTracker


# Fake detection result matching VehicleDet interface
@dataclass
class FakeDet:
    bbox: tuple
    conf: float
    cls: int


class FakeDetector:
    """Mock external detector (e.g. Hailo)."""
    def __init__(self):
        self.detections = []

    def detect(self, frame):
        return self.detections


@pytest.fixture
def ext_tracker():
    """Tracker with external detector (no YOLO dependency)."""
    detector = FakeDetector()
    t = VehicleTracker(
        model_path='unused.pt',
        detector=detector,
    )
    t._fake_detector = detector
    return t


def frame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# External detector mode
# ---------------------------------------------------------------------------

class TestExternalDetector:

    def test_new_vehicle_enters(self, ext_tracker):
        ext_tracker._fake_detector.detections = [
            FakeDet(bbox=(10, 10, 50, 50), conf=0.9, cls=2),
        ]
        vehicles = ext_tracker.update(frame())
        assert len(vehicles) == 1
        assert len(ext_tracker.get_entered()) == 1

    def test_tracked_vehicle_persists(self, ext_tracker):
        ext_tracker._fake_detector.detections = [
            FakeDet(bbox=(10, 10, 50, 50), conf=0.9, cls=2),
        ]
        ext_tracker.update(frame())
        # Same position — should match by IoU
        ext_tracker.update(frame())
        assert ext_tracker.get_entered() == []  # not new
        assert ext_tracker.get_active_count() == 1

    def test_iou_matching(self, ext_tracker):
        ext_tracker._fake_detector.detections = [
            FakeDet(bbox=(10, 10, 100, 100), conf=0.9, cls=2),
        ]
        ext_tracker.update(frame())
        tid = ext_tracker.get_entered()[0]['track_id']

        # Move slightly — should match same track
        ext_tracker._fake_detector.detections = [
            FakeDet(bbox=(15, 15, 105, 105), conf=0.9, cls=2),
        ]
        ext_tracker.update(frame())
        assert ext_tracker.get_entered() == []
        assert ext_tracker.get_active_count() == 1

    def test_new_detection_gets_new_id(self, ext_tracker):
        ext_tracker._fake_detector.detections = [
            FakeDet(bbox=(10, 10, 50, 50), conf=0.9, cls=2),
        ]
        ext_tracker.update(frame())
        tid1 = ext_tracker.get_entered()[0]['track_id']

        ext_tracker._fake_detector.detections = [
            FakeDet(bbox=(10, 10, 50, 50), conf=0.9, cls=2),
            FakeDet(bbox=(400, 400, 450, 450), conf=0.8, cls=5),
        ]
        ext_tracker.update(frame())
        entered = ext_tracker.get_entered()
        assert len(entered) == 1
        assert entered[0]['track_id'] != tid1

    def test_no_detections(self, ext_tracker):
        ext_tracker._fake_detector.detections = []
        vehicles = ext_tracker.update(frame())
        assert vehicles == []
        assert ext_tracker.get_entered() == []

    def test_vehicle_exit_after_timeout(self, ext_tracker):
        ext_tracker._fake_detector.detections = [
            FakeDet(bbox=(10, 10, 50, 50), conf=0.9, cls=2),
        ]
        ext_tracker.update(frame())

        # Remove detection and simulate time passing
        ext_tracker._fake_detector.detections = []
        # Fast-forward timestamp
        for tid in ext_tracker.active_tracks:
            ext_tracker.active_tracks[tid]['timestamp'] = time.time() - 999
        ext_tracker.update(frame())
        assert len(ext_tracker.get_exited()) == 1
        assert ext_tracker.get_active_count() == 0


# ---------------------------------------------------------------------------
# Frame skipping
# ---------------------------------------------------------------------------

class TestFrameSkipping:

    def test_process_every_n_skips_frames(self):
        detector = FakeDetector()
        detector.detections = [
            FakeDet(bbox=(10, 10, 50, 50), conf=0.9, cls=2),
        ]
        t = VehicleTracker(
            model_path='unused.pt',
            detector=detector,
            process_every_n=3,
        )
        # _frame_count starts at 0, increments before check
        # Frame 1: count=1, 1%3!=0 → skipped (no last vehicles yet)
        v1 = t.update(frame())
        assert len(v1) == 0  # skipped, _last_vehicles is empty

        # Frame 2: count=2, 2%3!=0 → skipped
        v2 = t.update(frame())
        assert len(v2) == 0

        # Frame 3: count=3, 3%3==0 → processed!
        v3 = t.update(frame())
        assert len(v3) == 1
        assert len(t.get_entered()) == 1

        # Frame 4: count=4, skipped — reuses last
        v4 = t.update(frame())
        assert len(v4) == 1
        assert t.get_entered() == []  # no new entries on skip

    def test_process_every_1_never_skips(self):
        detector = FakeDetector()
        detector.detections = [
            FakeDet(bbox=(10, 10, 50, 50), conf=0.9, cls=2),
        ]
        t = VehicleTracker(
            model_path='unused.pt',
            detector=detector,
            process_every_n=1,
        )
        t.update(frame())
        assert len(t.get_entered()) == 1

        detector.detections = [
            FakeDet(bbox=(10, 10, 50, 50), conf=0.9, cls=2),
            FakeDet(bbox=(300, 300, 350, 350), conf=0.8, cls=5),
        ]
        t.update(frame())
        assert len(t.get_entered()) == 1  # only the new one


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

class TestIoU:

    def test_identical_boxes(self):
        iou = VehicleTracker._compute_iou([0, 0, 100, 100], [0, 0, 100, 100])
        assert iou == 1.0

    def test_no_overlap(self):
        iou = VehicleTracker._compute_iou([0, 0, 50, 50], [100, 100, 200, 200])
        assert iou == 0.0

    def test_partial_overlap(self):
        iou = VehicleTracker._compute_iou([0, 0, 100, 100], [50, 50, 150, 150])
        # Intersection: 50x50=2500, Union: 10000+10000-2500=17500
        assert abs(iou - 2500 / 17500) < 1e-6

    def test_contained_box(self):
        iou = VehicleTracker._compute_iou([0, 0, 200, 200], [50, 50, 100, 100])
        # Intersection: 50x50=2500, Union: 40000+2500-2500=40000
        assert abs(iou - 2500 / 40000) < 1e-6

    def test_touching_boxes_no_overlap(self):
        iou = VehicleTracker._compute_iou([0, 0, 50, 50], [50, 50, 100, 100])
        assert iou == 0.0


# ---------------------------------------------------------------------------
# Track info and class names
# ---------------------------------------------------------------------------

class TestTrackInfo:

    def test_get_track_info(self, ext_tracker):
        ext_tracker._fake_detector.detections = [
            FakeDet(bbox=(10, 10, 50, 50), conf=0.9, cls=2),
        ]
        ext_tracker.update(frame())
        tid = ext_tracker.get_entered()[0]['track_id']
        info = ext_tracker.get_track_info(tid)
        assert 'first_seen' in info
        assert 'entry_position' in info
        assert info['class_name'] == 'car'  # cls=2 -> car (COCO)

    def test_get_track_info_nonexistent(self, ext_tracker):
        assert ext_tracker.get_track_info(9999) is None

    def test_class_names(self, ext_tracker):
        ext_tracker._fake_detector.detections = [
            FakeDet(bbox=(10, 10, 50, 50), conf=0.9, cls=5),
        ]
        ext_tracker.update(frame())
        assert ext_tracker.get_entered()[0]['class_name'] == 'bus'

    def test_unknown_class(self, ext_tracker):
        # cls=7 is truck in COCO mapping
        ext_tracker._fake_detector.detections = [
            FakeDet(bbox=(10, 10, 50, 50), conf=0.9, cls=7),
        ]
        ext_tracker.update(frame())
        assert ext_tracker.get_entered()[0]['class_name'] == 'truck'

    def test_unknown_class_name_mapping(self):
        # Test _get_class_name directly for an unmapped class
        detector = FakeDetector()
        t = VehicleTracker(model_path='unused.pt', detector=detector)
        assert t._get_class_name(99) == 'vehicle'

    def test_reset(self, ext_tracker):
        ext_tracker._fake_detector.detections = [
            FakeDet(bbox=(10, 10, 50, 50), conf=0.9, cls=2),
        ]
        ext_tracker.update(frame())
        ext_tracker.reset()
        assert ext_tracker.get_active_count() == 0
        assert ext_tracker.track_history == {}
        assert ext_tracker._frame_count == 0
