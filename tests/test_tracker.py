from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from src.tracker import VehicleTracker


def make_fake_result(track_ids, boxes, class_ids, confs):
    result = MagicMock()
    result.boxes.id.cpu().numpy.return_value = np.array(track_ids, dtype=float)
    result.boxes.xyxy.cpu().numpy.return_value = np.array(boxes, dtype=float)
    result.boxes.cls.cpu().numpy.return_value = np.array(class_ids, dtype=float)
    result.boxes.conf.cpu().numpy.return_value = np.array(confs, dtype=float)
    return result


@pytest.fixture
def tracker():
    with patch('src.tracker.YOLO') as MockYOLO:
        mock_model = MagicMock()
        MockYOLO.return_value = mock_model
        t = VehicleTracker()
        t._mock_model = mock_model
        yield t


def update_with(tracker, track_ids, boxes, class_ids, confs):
    tracker._mock_model.track.return_value = [make_fake_result(track_ids, boxes, class_ids, confs)]
    return tracker.update(np.zeros((480, 640, 3), dtype=np.uint8))


def test_vehicles_enter(tracker):
    update_with(tracker, [1, 2], [[10,10,50,50],[200,200,250,250]], [1, 0], [0.9, 0.85])
    assert tracker.get_active_count() == 2
    assert [v['track_id'] for v in tracker.get_entered()] == [1, 2]


def test_no_new_entries_on_second_frame(tracker):
    update_with(tracker, [1, 2], [[10,10,50,50],[200,200,250,250]], [1, 0], [0.9, 0.85])
    update_with(tracker, [1, 2], [[10,10,50,50],[200,200,250,250]], [1, 0], [0.9, 0.85])
    assert tracker.get_entered() == []


def test_vehicle_disappears_not_yet_exited(tracker):
    update_with(tracker, [1, 2], [[10,10,50,50],[200,200,250,250]], [1, 0], [0.9, 0.85])
    update_with(tracker, [1], [[10,10,50,50]], [1], [0.9])
    assert tracker.get_active_count() == 2
    assert tracker.get_exited() == []


def test_new_vehicle_enters(tracker):
    update_with(tracker, [1, 2], [[10,10,50,50],[200,200,250,250]], [1, 0], [0.9, 0.85])
    update_with(tracker, [1, 2, 3], [[10,10,50,50],[200,200,250,250],[300,300,350,350]], [1, 0, 2], [0.9, 0.85, 0.8])
    assert tracker.get_active_count() == 3
    assert [v['track_id'] for v in tracker.get_entered()] == [3]


def test_vehicle_class_name(tracker):
    update_with(tracker, [1], [[10,10,50,50]], [0], [0.9])
    vehicles = tracker.get_entered()
    assert vehicles[0]['class_name'] == 'bus'


def test_reset(tracker):
    update_with(tracker, [1, 2], [[10,10,50,50],[200,200,250,250]], [1, 0], [0.9, 0.85])
    tracker.reset()
    assert tracker.get_active_count() == 0
    assert tracker.track_history == {}
    assert tracker.get_entered() == []
    assert tracker.get_exited() == []
