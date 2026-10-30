import sqlite3
from datetime import datetime, timezone
from config import DATA_DIR
class DatabaseManager:
    def __init__(self, path=None):
        self.path = str(path or DATA_DIR / "screening.db")
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS screenings (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, candidate TEXT, job_title TEXT, score REAL, recommendation TEXT)")
    def _connect(self): return sqlite3.connect(self.path)
    def save_screening(self, candidate, job_title, score, recommendation):
        with self._connect() as conn:
            conn.execute("INSERT INTO screenings (created_at, candidate, job_title, score, recommendation) VALUES (?, ?, ?, ?, ?)", (datetime.now(timezone.utc).isoformat(), candidate, job_title, float(score), recommendation))
    def get_history(self):
        with self._connect() as conn:
            return conn.execute("SELECT created_at, candidate, job_title, score, recommendation FROM screenings ORDER BY id DESC LIMIT 500").fetchall()
    def clear_history(self):
        with self._connect() as conn: conn.execute("DELETE FROM screenings")
