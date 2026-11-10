"""Local SQLite persistence for DriveGuard safety events."""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class EventStore:
    """Stores event-start records locally; video frames are never persisted."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.path = Path(db_path) if db_path else project_root / "data" / "driveguard.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), timeout=5.0)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("""
            CREATE TABLE IF NOT EXISTS safety_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                safety_score INTEGER NOT NULL,
                message TEXT
            )
        """)
        self._connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_safety_events_occurred_at "
            "ON safety_events(occurred_at)"
        )
        self._connection.commit()
        log.info("Event history database: %s", self.path)

    def record_event(self, event_type: str, risk_level: str,
                     safety_score: int, message: Optional[str] = None) -> None:
        try:
            occurred_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            with self._connection:
                self._connection.execute(
                    """INSERT INTO safety_events
                       (occurred_at, event_type, risk_level, safety_score, message)
                       VALUES (?, ?, ?, ?, ?)""",
                    (occurred_at, event_type, risk_level, int(safety_score), message),
                )
        except sqlite3.Error:
            log.exception("Could not save safety event to %s", self.path)

    def close(self) -> None:
        self._connection.close()
