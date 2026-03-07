import time
import pytest
from src.match import PlateMatcher


def make_config(polygon=None):
    return {
        'entry_zone': {
            'polygon': polygon or [[0, 0], [200, 0], [200, 200], [0, 200]]
        },
        'plate_reader': {
            'poll_interval': 1
        }
    }


@pytest.fixture
def matcher():
    return PlateMatcher(make_config())


# --- push_plate ---

def test_push_plate_adds_to_queue(matcher):
    matcher.push_plate("ABC123")
    assert matcher.queue_size() == 1


def test_push_plate_empty_string_ignored(matcher):
    matcher.push_plate("")
    assert matcher.queue_size() == 0


def test_push_plate_none_ignored(matcher):
    matcher.push_plate(None)
    assert matcher.queue_size() == 0


def test_push_plate_uppercases(matcher):
    matcher.push_plate("abc123")
    # Assign it to a track to verify it was uppercased
    matcher.try_assign(1, (100, 100))
    assert matcher.get_plate(1) == "ABC123"


def test_push_plate_strips_whitespace(matcher):
    matcher.push_plate("  ABC123  ")
    matcher.try_assign(1, (100, 100))
    assert matcher.get_plate(1) == "ABC123"


# --- try_assign ---

def test_try_assign_in_zone_with_plate(matcher):
    matcher.push_plate("PLATE1")
    matcher.try_assign(1, (100, 100))  # inside entry zone
    assert matcher.get_plate(1) == "PLATE1"
    assert matcher.queue_size() == 0


def test_try_assign_outside_zone(matcher):
    matcher.push_plate("PLATE1")
    matcher.try_assign(1, (500, 500))  # outside entry zone
    assert matcher.get_plate(1) is None
    assert matcher.queue_size() == 1  # plate stays in queue


def test_try_assign_no_plate_in_queue(matcher):
    matcher.try_assign(1, (100, 100))
    assert matcher.get_plate(1) is None


def test_try_assign_already_assigned(matcher):
    matcher.push_plate("FIRST")
    matcher.try_assign(1, (100, 100))
    matcher.push_plate("SECOND")
    matcher.try_assign(1, (100, 100))  # already has plate
    assert matcher.get_plate(1) == "FIRST"
    assert matcher.queue_size() == 1  # SECOND still in queue


def test_multiple_vehicles_multiple_plates(matcher):
    matcher.push_plate("PLATE_A")
    matcher.push_plate("PLATE_B")
    matcher.try_assign(1, (100, 100))
    matcher.try_assign(2, (100, 100))
    assert matcher.get_plate(1) == "PLATE_A"
    assert matcher.get_plate(2) == "PLATE_B"
    assert matcher.queue_size() == 0


# --- late matching (vehicle arrives before plate) ---

def test_late_match_vehicle_then_plate(matcher):
    # Vehicle enters zone first, no plate yet
    matcher.try_assign(1, (100, 100))
    assert matcher.get_plate(1) is None

    # Plate arrives later — should auto-match
    matcher.push_plate("LATE_PLATE")
    assert matcher.get_plate(1) == "LATE_PLATE"


def test_late_match_oldest_vehicle_gets_plate(matcher):
    matcher.try_assign(1, (100, 100))
    matcher.try_assign(2, (100, 100))
    matcher.push_plate("PLATE1")
    assert matcher.get_plate(1) == "PLATE1"  # oldest gets it
    assert matcher.get_plate(2) is None


# --- release ---

def test_release_removes_assignment(matcher):
    matcher.push_plate("PLATE1")
    matcher.try_assign(1, (100, 100))
    assert matcher.get_plate(1) == "PLATE1"
    matcher.release(1)
    assert matcher.get_plate(1) is None


def test_release_nonexistent_track(matcher):
    matcher.release(999)  # should not raise


# --- get_all ---

def test_get_all(matcher):
    matcher.push_plate("A")
    matcher.push_plate("B")
    matcher.try_assign(1, (100, 100))
    matcher.try_assign(2, (100, 100))
    result = matcher.get_all()
    assert result == {1: "A", 2: "B"}


def test_get_all_empty(matcher):
    assert matcher.get_all() == {}


# --- entry zone ---

def test_entry_zone_boundary():
    config = make_config([[0, 0], [100, 0], [100, 100], [0, 100]])
    m = PlateMatcher(config)
    m.push_plate("EDGE")
    m.try_assign(1, (0, 0))  # on the boundary
    assert matcher_plate_or_waiting(m, 1)


def test_small_entry_zone_rejects_outside():
    config = make_config([[0, 0], [10, 0], [10, 10], [0, 10]])
    m = PlateMatcher(config)
    m.push_plate("OUTSIDE")
    m.try_assign(1, (500, 500))  # well outside the 10x10 zone
    assert m.get_plate(1) is None
    assert m.queue_size() == 1  # plate still in queue


# --- FIFO order ---

def test_fifo_order(matcher):
    matcher.push_plate("FIRST")
    matcher.push_plate("SECOND")
    matcher.push_plate("THIRD")
    matcher.try_assign(10, (100, 100))
    matcher.try_assign(20, (100, 100))
    matcher.try_assign(30, (100, 100))
    assert matcher.get_plate(10) == "FIRST"
    assert matcher.get_plate(20) == "SECOND"
    assert matcher.get_plate(30) == "THIRD"


# helper
def matcher_plate_or_waiting(m, track_id):
    """Returns True if track has a plate or is in unmatched_tracks."""
    return m.get_plate(track_id) is not None or track_id in m.unmatched_tracks
