import logging
import sqlite3
from datetime import date, datetime
from pathlib import Path

log = logging.getLogger(__name__)


def _normalize_plate(plate):
    """Normalize plate strings for consistent matching/storage."""
    if not plate:
        return plate
    return plate.strip().upper().replace(" ", "").replace("-", "")


class VehicleDatabase:

    def __init__(self, db_path='data/database.db'):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute('PRAGMA journal_mode=WAL;')
        self._init_schema()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _init_schema(self):
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                plate       TEXT,
                track_id    INTEGER,
                space       TEXT,
                entry_time  TEXT,
                exit_time   TEXT,
                duration    REAL
            );
            CREATE TABLE IF NOT EXISTS vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                license_plate TEXT UNIQUE NOT NULL,
                owner_name TEXT
            );
            CREATE TABLE IF NOT EXISTS permits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                license_plate TEXT,
                permit_type TEXT,
                expiration_date TEXT,
                is_active INTEGER
            );
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                license_plate TEXT,
                timestamp TEXT,
                reason TEXT
            );
        ''')
        self.conn.commit()

    # --- Session tracking ---

    def log_exit(self, track_id, plate, space, entry_time, exit_time):
        plate = _normalize_plate(plate)
        duration = exit_time - entry_time if entry_time and exit_time else None
        entry_iso = datetime.fromtimestamp(entry_time).isoformat(timespec="seconds") if entry_time else None
        exit_iso = datetime.fromtimestamp(exit_time).isoformat(timespec="seconds") if exit_time else None
        try:
            self.conn.execute(
                '''INSERT INTO sessions (plate, track_id, space, entry_time, exit_time, duration)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (plate, track_id, space, entry_iso, exit_iso, duration)
            )
            self.conn.commit()
        except sqlite3.Error as e:
            log.error("Failed to log exit for track %s: %s", track_id, e)

    def get_sessions(self, plate=None, limit=100):
        if plate:
            plate = _normalize_plate(plate)
            rows = self.conn.execute(
                'SELECT * FROM sessions WHERE plate=? ORDER BY entry_time DESC LIMIT ?',
                (plate, limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                'SELECT * FROM sessions ORDER BY entry_time DESC LIMIT ?',
                (limit,)
            ).fetchall()
        return rows

    # --- Vehicle registry ---

    def add_vehicle(self, license_plate, owner_name=None):
        license_plate = _normalize_plate(license_plate)
        try:
            self.conn.execute(
                '''INSERT INTO vehicles (license_plate, owner_name)
                   VALUES (?, ?)
                   ON CONFLICT(license_plate) DO UPDATE SET owner_name = excluded.owner_name''',
                (license_plate, owner_name),
            )
            self.conn.commit()
        except sqlite3.Error as e:
            log.error("Failed to add vehicle %s: %s", license_plate, e)

    # --- Permits ---

    def add_permit(self, license_plate, permit_type, expiration_date):
        """Add an active permit. expiration_date: YYYY-MM-DD."""
        license_plate = _normalize_plate(license_plate)
        try:
            self.conn.execute(
                '''INSERT INTO permits (license_plate, permit_type, expiration_date, is_active)
                   VALUES (?, ?, ?, 1)''',
                (license_plate, permit_type, expiration_date),
            )
            self.conn.commit()
        except sqlite3.Error as e:
            log.error("Failed to add permit for %s: %s", license_plate, e)

    def check_permit(self, license_plate):
        """Return True if the plate has an active, non-expired permit."""
        license_plate = _normalize_plate(license_plate)
        today_iso = date.today().isoformat()
        row = self.conn.execute(
            '''SELECT 1 FROM permits
               WHERE license_plate = ? AND is_active = 1 AND expiration_date >= ?
               LIMIT 1''',
            (license_plate, today_iso),
        ).fetchone()
        return row is not None

    def get_active_permit(self, license_plate):
        """Return active permit details dict, or None."""
        license_plate = _normalize_plate(license_plate)
        today_iso = date.today().isoformat()
        row = self.conn.execute(
            '''SELECT permit_type, expiration_date FROM permits
               WHERE license_plate = ? AND is_active = 1 AND expiration_date >= ?
               ORDER BY expiration_date DESC LIMIT 1''',
            (license_plate, today_iso),
        ).fetchone()
        if row is None:
            return None
        return {"permit_type": row[0], "expiration_date": row[1]}

    def deactivate_permits(self, license_plate):
        """Set is_active = 0 for all permits for the given plate."""
        license_plate = _normalize_plate(license_plate)
        try:
            self.conn.execute(
                'UPDATE permits SET is_active = 0 WHERE license_plate = ?',
                (license_plate,),
            )
            self.conn.commit()
        except sqlite3.Error as e:
            log.error("Failed to deactivate permits for %s: %s", license_plate, e)

    # --- Violations ---

    def record_violation(self, license_plate, reason):
        """Insert a violation with current timestamp."""
        license_plate = _normalize_plate(license_plate)
        timestamp_iso = datetime.now().isoformat(timespec="seconds")
        try:
            self.conn.execute(
                '''INSERT INTO violations (license_plate, timestamp, reason)
                   VALUES (?, ?, ?)''',
                (license_plate, timestamp_iso, reason),
            )
            self.conn.commit()
        except sqlite3.Error as e:
            log.error("Failed to record violation for %s: %s", license_plate, e)

    def close(self):
        self.conn.close()
