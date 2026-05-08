"""Standalone commons signal emitter — no Chicory dependency.

Writes signals to the shared ``pending_signals`` table in the commons
SQLite DB.  Any project (Chicory, GPT-GU, or others) can import this
module and emit signals with only stdlib dependencies.

Usage::

    emitter = SignalEmitter(
        db_path="~/.chicory/commons.db",
        project_id="gpt-gu",
    )
    emitter.start()

    emitter.emit_store(["memory", "transformation"])
    emitter.emit_retrieve(["time", "recursion"])
    emitter.emit_synchronicity(["memory", "time"], strength=0.85, event_type="convergence")

    emitter.stop()
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS pending_signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id      TEXT NOT NULL,
    op_type         TEXT NOT NULL,
    tags            TEXT NOT NULL,
    strength        REAL,
    event_type      TEXT,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    processed       INTEGER NOT NULL DEFAULT 0,
    tag_set_hash    TEXT NOT NULL,
    UNIQUE(project_id, op_type, tag_set_hash, created_at)
)
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_pending_signals_unprocessed
    ON pending_signals(processed) WHERE processed = 0
"""

_INSERT = """
INSERT OR IGNORE INTO pending_signals
    (project_id, op_type, tags, strength, event_type, tag_set_hash)
VALUES (?, ?, ?, ?, ?, ?)
"""


def _tag_set_hash(tags: list[str]) -> str:
    canonical = json.dumps(sorted(tags), separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class SignalEmitter:
    """Buffers and flushes signals to the commons DB.

    Parameters
    ----------
    db_path : str or Path
        Path to the commons SQLite database.
    project_id : str
        Identifier for the emitting project (e.g. ``"gpt-gu"``).
    buffer_size : int
        Flush the buffer after this many signals accumulate.
    flush_interval : float
        Seconds between automatic flush sweeps.
    """

    def __init__(
        self,
        db_path: str | Path = "",
        project_id: str = "",
        buffer_size: int = 10,
        flush_interval: float = 5.0,
    ) -> None:
        if not db_path:
            db_path = Path.home() / ".chicory" / "commons.db"
        self._db_path = str(db_path)
        self._project_id = project_id
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval

        self._buffer: list[tuple[str, list[str], float | None, str | None]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._flush_thread: threading.Thread | None = None
        self._conn: sqlite3.Connection | None = None

    def start(self) -> None:
        """Open the DB connection and start the periodic flush thread."""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_INDEX)
        self._conn.commit()

        self._stop_event.clear()
        self._flush_thread = threading.Thread(
            target=self._periodic_flush, daemon=True,
        )
        self._flush_thread.start()

    def stop(self) -> None:
        """Flush remaining signals and close the connection."""
        self._stop_event.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=5.0)
            self._flush_thread = None
        self.flush()
        if self._conn:
            self._conn.close()
            self._conn = None

    def emit_store(self, tags: list[str]) -> None:
        self._buffer_signal("store", tags)

    def emit_retrieve(self, tags: list[str]) -> None:
        self._buffer_signal("retrieve", tags)

    def emit_synchronicity(
        self,
        tags: list[str],
        strength: float = 0.0,
        event_type: str = "",
    ) -> None:
        self._buffer_signal("synchronicity", tags, strength, event_type)
        self.flush()

    def emit(self, op_type: str, tags: list[str], **kwargs) -> None:
        """Generic emission for project-specific signal types."""
        self._buffer_signal(
            op_type, tags,
            kwargs.get("strength"),
            kwargs.get("event_type"),
        )

    def flush(self) -> None:
        """Write buffered signals to the DB."""
        with self._lock:
            if not self._buffer:
                return
            batch = self._buffer[:]
            self._buffer.clear()

        if not self._conn:
            logger.warning("SignalEmitter not started — dropping %d signals", len(batch))
            return

        rows = [
            (
                self._project_id,
                op_type,
                json.dumps(sorted(tags)),
                strength,
                event_type,
                _tag_set_hash(tags),
            )
            for op_type, tags, strength, event_type in batch
        ]
        try:
            self._conn.executemany(_INSERT, rows)
            self._conn.commit()
        except sqlite3.Error:
            logger.exception("Failed to flush %d signals", len(rows))

    def _buffer_signal(
        self,
        op_type: str,
        tags: list[str],
        strength: float | None = None,
        event_type: str | None = None,
    ) -> None:
        if not tags:
            return
        with self._lock:
            self._buffer.append((op_type, tags, strength, event_type))
            should_flush = len(self._buffer) >= self._buffer_size
        if should_flush:
            self.flush()

    def _periodic_flush(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(self._flush_interval)
            if not self._stop_event.is_set():
                self.flush()
