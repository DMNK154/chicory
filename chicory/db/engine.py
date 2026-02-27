"""SQLite connection management with WAL mode."""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from chicory.config import ChicoryConfig
from chicory.exceptions import DatabaseError


class DatabaseEngine:
    """Manages a SQLite connection with WAL mode and proper pragmas."""

    def __init__(self, config: ChicoryConfig) -> None:
        self._config = config
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise DatabaseError("Database not connected. Call connect() first.")
        return self._conn

    def connect(self) -> sqlite3.Connection:
        """Open the database connection and set pragmas."""
        db_path = str(self._config.db_path)
        is_memory = db_path == ":memory:"

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if not is_memory:
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        return self._conn

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> sqlite3.Cursor:
        """Execute a single SQL statement (thread-safe)."""
        with self._lock:
            try:
                if params is not None:
                    return self.connection.execute(sql, params)
                return self.connection.execute(sql)
            except sqlite3.Error as e:
                raise DatabaseError(f"SQL execution failed: {e}") from e

    def executemany(self, sql: str, params_list: list[tuple[Any, ...]]) -> sqlite3.Cursor:
        """Execute a SQL statement against multiple parameter sets (thread-safe)."""
        with self._lock:
            try:
                return self.connection.executemany(sql, params_list)
            except sqlite3.Error as e:
                raise DatabaseError(f"SQL executemany failed: {e}") from e

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager for atomic operations (thread-safe)."""
        with self._lock:
            conn = self.connection
            try:
                conn.execute("BEGIN")
                yield conn
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
