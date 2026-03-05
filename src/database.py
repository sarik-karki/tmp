import sqlite3
from pathlib import Path


class VehicleDatabase:

    def __init__(self, db_path='data/database.db'):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                plate       TEXT,
                track_id    INTEGER,
                space       TEXT,
                entry_time  REAL,
                exit_time   REAL,
                duration    REAL
            )
        ''')
        self.conn.commit()

    def log_exit(self, track_id, plate, space, entry_time, exit_time):
        duration = exit_time - entry_time if entry_time else None
        self.conn.execute(
            '''INSERT INTO sessions (plate, track_id, space, entry_time, exit_time, duration)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (plate, track_id, space, entry_time, exit_time, duration)
        )
        self.conn.commit()

    def get_sessions(self, plate=None, limit=100):
        if plate:
            rows = self.conn.execute(
                'SELECT * FROM sessions WHERE plate=? ORDER BY entry_time DESC LIMIT ?',
                (plate.upper(), limit)
            ).fetchall()
        else:
            rows = self.conn.execute(
                'SELECT * FROM sessions ORDER BY entry_time DESC LIMIT ?',
                (limit,)
            ).fetchall()
        return rows

    def close(self):
        self.conn.close()
