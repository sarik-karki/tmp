import pytest
from src.space_manager import SpaceManager


@pytest.fixture
def sm():
    s = SpaceManager()
    s.add_space('A1', [[0,0],[100,0],[100,100],[0,100]])
    s.add_space('A2', [[110,0],[210,0],[210,100],[110,100]])
    s.add_space('B1', [[0,110],[100,110],[100,210],[0,210]])
    s.add_space('B2', [[110,110],[210,110],[210,210],[110,210]])
    return s


def test_point_inside_space(sm):
    assert sm.get_space((50, 50)) == 'A1'
    assert sm.get_space((160, 50)) == 'A2'
    assert sm.get_space((50, 160)) == 'B1'
    assert sm.get_space((160, 160)) == 'B2'


def test_point_outside_all_spaces(sm):
    assert sm.get_space((300, 300)) is None


def test_update_occupancy(sm):
    vehicles = [
        {'track_id': 1, 'center': (50, 50)},
        {'track_id': 2, 'center': (160, 160)},
    ]
    result = sm.update_occupancy(vehicles)
    assert result['A1'] == {'status': 'occupied', 'track_id': 1}
    assert result['B2'] == {'status': 'occupied', 'track_id': 2}
    assert result['A2'] == {'status': 'empty', 'track_id': None}
    assert result['B1'] == {'status': 'empty', 'track_id': None}


def test_empty_and_occupied_spaces(sm):
    sm.update_occupancy([{'track_id': 1, 'center': (50, 50)}])
    assert sm.get_occupied_spaces() == ['A1']
    assert set(sm.get_empty_spaces()) == {'A2', 'B1', 'B2'}


def test_get_vehicle_space(sm):
    sm.update_occupancy([
        {'track_id': 1, 'center': (50, 50)},
        {'track_id': 2, 'center': (160, 160)},
    ])
    assert sm.get_vehicle_space(1) == 'A1'
    assert sm.get_vehicle_space(2) == 'B2'
    assert sm.get_vehicle_space(99) is None


def test_occupancy_summary(sm):
    sm.update_occupancy([
        {'track_id': 1, 'center': (50, 50)},
        {'track_id': 2, 'center': (160, 160)},
    ])
    summary = sm.get_occupancy_summary()
    assert summary['total'] == 4
    assert summary['occupied'] == 2
    assert summary['empty'] == 2
    assert summary['percent_full'] == 50.0


def test_remove_space(sm):
    sm.update_occupancy([{'track_id': 1, 'center': (50, 50)}])
    sm.remove_space('A1')
    assert 'A1' not in sm.spaces
    assert sm.get_occupancy_summary()['total'] == 3


def test_reset_occupancy_on_update(sm):
    sm.update_occupancy([{'track_id': 1, 'center': (50, 50)}])
    sm.update_occupancy([])
    assert sm.get_occupied_spaces() == []
    assert len(sm.get_empty_spaces()) == 4
