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
    with VehicleDatabase(db_path) as database:
        assert os.path.isfile(db_path)


def test_context_manager(tmp_path):
    db_path = str(tmp_path / "ctx.db")
    with VehicleDatabase(db_path) as database:
        database.add_vehicle("CTX111", "Test")
    # Connection should be closed after with block
    # Re-open to verify data persisted
    with VehicleDatabase(db_path) as database:
        row = database.conn.execute(
            "SELECT license_plate FROM vehicles WHERE license_plate = ?",
            ("CTX111",)
        ).fetchone()
        assert row is not None


def test_log_exit_and_get_sessions(db):
    now = time.time()
    db.log_exit(track_id=1, plate="ABC123", space="A1",
                entry_time=now - 60, exit_time=now)
    rows = db.get_sessions()
    assert len(rows) == 1
    row = rows[0]
    assert row[1] == "ABC123"  # plate (normalized)
    assert row[2] == 1         # track_id
    assert row[3] == "A1"      # space
    assert row[4] is not None  # entry_time (ISO string)
    assert row[5] is not None  # exit_time (ISO string)
    assert abs(row[6] - 60.0) < 0.1  # duration


def test_log_exit_normalizes_plate(db):
    now = time.time()
    db.log_exit(track_id=1, plate="abc-123", space="A1",
                entry_time=now - 10, exit_time=now)
    rows = db.get_sessions()
    assert rows[0][1] == "ABC123"


def test_log_exit_null_entry_time(db):
    now = time.time()
    db.log_exit(track_id=1, plate="XYZ789", space=None,
                entry_time=None, exit_time=now)
    rows = db.get_sessions()
    assert len(rows) == 1
    assert rows[0][4] is None  # entry_time is None
    assert rows[0][6] is None  # duration is None


def test_get_sessions_filter_by_plate(db):
    now = time.time()
    db.log_exit(track_id=1, plate="ABC123", space="A1",
                entry_time=now - 60, exit_time=now)
    db.log_exit(track_id=2, plate="XYZ789", space="A2",
                entry_time=now - 30, exit_time=now)
    rows = db.get_sessions(plate="abc-123")  # denormalized — should still match
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


# ---------------------------------------------------------------------------
# Permits
# ---------------------------------------------------------------------------

def test_check_permit_valid(db):
    db.add_permit("ABC123", "monthly", "2099-12-31")
    assert db.check_permit("ABC123") is True

def test_check_permit_expired(db):
    db.add_permit("XYZ789", "daily", "2020-01-01")
    assert db.check_permit("XYZ789") is False

def test_check_permit_missing(db):
    assert db.check_permit("NOPE000") is False

def test_check_permit_normalizes(db):
    db.add_permit("abc-123", "monthly", "2099-12-31")
    assert db.check_permit("  ABC 123 ") is True

def test_get_active_permit(db):
    db.add_permit("TEST111", "annual", "2099-06-30")
    permit = db.get_active_permit("TEST111")
    assert permit is not None
    assert permit["permit_type"] == "annual"
    assert permit["expiration_date"] == "2099-06-30"

def test_get_active_permit_none(db):
    assert db.get_active_permit("NOPE999") is None

def test_deactivate_permits(db):
    db.add_permit("DEAC111", "monthly", "2099-12-31")
    assert db.check_permit("DEAC111") is True
    db.deactivate_permits("DEAC111")
    assert db.check_permit("DEAC111") is False


# ---------------------------------------------------------------------------
# Violations
# ---------------------------------------------------------------------------

def test_record_violation(db):
    db.record_violation("BAD123", "No valid parking permit")
    rows = db.conn.execute(
        "SELECT license_plate, reason FROM violations WHERE license_plate = ?",
        ("BAD123",)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][1] == "No valid parking permit"


# ---------------------------------------------------------------------------
# Vehicle registry
# ---------------------------------------------------------------------------

def test_add_vehicle(db):
    db.add_vehicle("REG111", "John Doe")
    row = db.conn.execute(
        "SELECT license_plate, owner_name FROM vehicles WHERE license_plate = ?",
        ("REG111",)
    ).fetchone()
    assert row == ("REG111", "John Doe")

def test_add_vehicle_upsert(db):
    db.add_vehicle("REG222", "Alice")
    db.add_vehicle("REG222", "Bob")
    row = db.conn.execute(
        "SELECT owner_name FROM vehicles WHERE license_plate = ?",
        ("REG222",)
    ).fetchone()
    assert row[0] == "Bob"
