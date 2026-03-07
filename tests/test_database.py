import os
import time
import pytest
from src.database import VehicleDatabase


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    database = VehicleDatabase(db_path)
    yield database
    database.close()


def test_creates_db_file(tmp_path):
    db_path = str(tmp_path / "subdir" / "test.db")
    database = VehicleDatabase(db_path)
    assert os.path.isfile(db_path)
    database.close()


def test_log_exit_and_get_sessions(db):
    now = time.time()
    db.log_exit(track_id=1, plate="ABC123", space="A1",
                entry_time=now - 60, exit_time=now)
    rows = db.get_sessions()
    assert len(rows) == 1
    row = rows[0]
    assert row[1] == "ABC123"  # plate
    assert row[2] == 1         # track_id
    assert row[3] == "A1"      # space
    assert abs(row[6] - 60.0) < 0.1  # duration


def test_log_exit_null_entry_time(db):
    now = time.time()
    db.log_exit(track_id=1, plate="XYZ789", space=None,
                entry_time=None, exit_time=now)
    rows = db.get_sessions()
    assert len(rows) == 1
    assert rows[0][6] is None  # duration is None


def test_get_sessions_filter_by_plate(db):
    now = time.time()
    db.log_exit(track_id=1, plate="ABC123", space="A1",
                entry_time=now - 60, exit_time=now)
    db.log_exit(track_id=2, plate="XYZ789", space="A2",
                entry_time=now - 30, exit_time=now)
    rows = db.get_sessions(plate="abc123")  # lowercase — should match
    assert len(rows) == 1
    assert rows[0][1] == "ABC123"


def test_get_sessions_limit(db):
    now = time.time()
    for i in range(10):
        db.log_exit(track_id=i, plate=f"PLATE{i}", space="A1",
                    entry_time=now - 60, exit_time=now)
    rows = db.get_sessions(limit=3)
    assert len(rows) == 3


def test_get_sessions_empty(db):
    rows = db.get_sessions()
    assert rows == []


def test_log_exit_no_plate(db):
    now = time.time()
    db.log_exit(track_id=1, plate=None, space="A1",
                entry_time=now - 10, exit_time=now)
    rows = db.get_sessions()
    assert len(rows) == 1
    assert rows[0][1] is None


def test_multiple_exits_same_plate(db):
    now = time.time()
    db.log_exit(track_id=1, plate="AAA111", space="A1",
                entry_time=now - 120, exit_time=now - 60)
    db.log_exit(track_id=2, plate="AAA111", space="B2",
                entry_time=now - 50, exit_time=now)
    rows = db.get_sessions(plate="AAA111")
    assert len(rows) == 2
