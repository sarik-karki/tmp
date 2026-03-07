"""Extended space manager tests: save/load, draw, edge cases."""

import json
import numpy as np
import pytest
from src.space_manager import SpaceManager


@pytest.fixture
def sm():
    s = SpaceManager()
    s.add_space('A1', [[0, 0], [100, 0], [100, 100], [0, 100]])
    s.add_space('A2', [[110, 0], [210, 0], [210, 100], [110, 100]])
    return s


def test_save_and_load(tmp_path, sm):
    filepath = str(tmp_path / "spaces.json")
    sm.save_spaces(filepath)

    sm2 = SpaceManager(filepath)
    assert set(sm2.spaces.keys()) == {'A1', 'A2'}
    assert sm2.get_space((50, 50)) == 'A1'


def test_load_nonexistent_file():
    sm = SpaceManager("nonexistent_file.json")
    assert sm.spaces == {}


def test_get_space_center(sm):
    center = sm.get_space_center('A1')
    assert center is not None
    assert abs(center[0] - 50) <= 1
    assert abs(center[1] - 50) <= 1


def test_get_space_center_nonexistent(sm):
    assert sm.get_space_center('Z99') is None


def test_add_space_updates_caches(sm):
    sm.add_space('C1', [[300, 300], [400, 300], [400, 400], [300, 400]])
    assert 'C1' in sm._space_centers
    assert 'C1' in sm._space_bboxes
    assert sm.get_space((350, 350)) == 'C1'


def test_remove_space_cleans_caches(sm):
    sm.remove_space('A1')
    assert 'A1' not in sm.spaces
    assert 'A1' not in sm._space_centers
    assert 'A1' not in sm._space_bboxes
    assert 'A1' not in sm.space_status
    assert 'A1' not in sm.space_occupants


def test_remove_nonexistent_space(sm):
    sm.remove_space('NOPE')  # should not raise


def test_occupancy_summary_empty():
    sm = SpaceManager()
    summary = sm.get_occupancy_summary()
    assert summary['total'] == 0
    assert summary['percent_full'] == 0


def test_draw_spaces_returns_frame(sm):
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    result = sm.draw_spaces(frame)
    assert result is not None
    assert result.shape == frame.shape


def test_draw_spaces_no_labels(sm):
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    result = sm.draw_spaces(frame, show_labels=False)
    assert result is not None


def test_draw_spaces_no_status(sm):
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    result = sm.draw_spaces(frame, show_status=False)
    assert result is not None


def test_bbox_prefilter_speeds_up(sm):
    # Point far outside bounding rects should be filtered fast
    assert sm.get_space((9999, 9999)) is None


def test_boundary_point(sm):
    # Point on the polygon edge (0,0) should be inside (pointPolygonTest >= 0)
    result = sm.get_space((0, 0))
    assert result == 'A1'


def test_update_occupancy_overwrites(sm):
    """Two vehicles in same space — last one wins."""
    vehicles = [
        {'track_id': 1, 'center': (50, 50)},
        {'track_id': 2, 'center': (50, 50)},
    ]
    result = sm.update_occupancy(vehicles)
    assert result['A1']['status'] == 'occupied'
    assert result['A1']['track_id'] == 2  # last one overwrites


def test_triangle_space():
    sm = SpaceManager()
    sm.add_space('TRI', [[0, 0], [200, 0], [100, 200]])
    assert sm.get_space((100, 50)) == 'TRI'
    assert sm.get_space((0, 200)) is None
