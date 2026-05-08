"""Server-wide user database for credit tracking."""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

USER_DB_PATH = Path.home() / ".chicory" / "users.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    email           TEXT NOT NULL UNIQUE COLLATE NOCASE,
    file_credits    INTEGER NOT NULL DEFAULT 0,
    message_credits INTEGER NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);

CREATE TABLE IF NOT EXISTS credit_transactions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id           INTEGER NOT NULL REFERENCES users(id),
    transaction_type  TEXT NOT NULL,
    credit_type       TEXT NOT NULL,
    amount            INTEGER NOT NULL,
    stripe_session_id TEXT,
    session_id        TEXT,
    created_at        TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);

CREATE TABLE IF NOT EXISTS subscription_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    stripe_event_id TEXT NOT NULL UNIQUE,
    event_type      TEXT NOT NULL,
    user_id         INTEGER NOT NULL REFERENCES users(id),
    processed_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_credit_tx_user ON credit_transactions(user_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_credit_tx_stripe_unique
    ON credit_transactions(stripe_session_id)
    WHERE stripe_session_id IS NOT NULL;
"""

_MIGRATION_SQL = """
ALTER TABLE users ADD COLUMN subscription_status TEXT NOT NULL DEFAULT 'none';
ALTER TABLE users ADD COLUMN stripe_customer_id TEXT;
ALTER TABLE users ADD COLUMN stripe_subscription_id TEXT;
ALTER TABLE users ADD COLUMN subscription_expires_at TEXT;
"""


class UserDB:
    """Manages ~/.chicory/users.db for persistent credit tracking."""

    def __init__(self, db_path: Path = USER_DB_PATH) -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> None:
        """Open the database and apply schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_SCHEMA_SQL)
        self._migrate()
        self._conn.commit()
        logger.info("UserDB connected: %s", self._db_path)

    def _migrate(self) -> None:
        """Add subscription columns if they don't exist yet."""
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(users)").fetchall()
        }
        if "subscription_status" not in cols:
            for stmt in _MIGRATION_SQL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    self._conn.execute(stmt)
            self._conn.execute(
                """CREATE UNIQUE INDEX IF NOT EXISTS idx_users_stripe_customer
                   ON users(stripe_customer_id) WHERE stripe_customer_id IS NOT NULL"""
            )
            logger.info("UserDB: migrated users table with subscription columns")

    def get_or_create_user(self, email: str) -> dict:
        """Lookup user by email, creating a zero-balance row if new."""
        _cols = "id, email, file_credits, message_credits, subscription_status"
        with self._lock:
            conn = self._conn
            row = conn.execute(
                f"SELECT {_cols} FROM users WHERE email = ?", (email,),
            ).fetchone()
            if row:
                return dict(row)
            conn.execute("INSERT INTO users (email) VALUES (?)", (email,))
            conn.commit()
            row = conn.execute(
                f"SELECT {_cols} FROM users WHERE email = ?", (email,),
            ).fetchone()
            return dict(row)

    def add_credits(
        self, email: str, credit_type: str, amount: int, stripe_session_id: str,
    ) -> dict:
        """Add purchased credits atomically. Idempotent via stripe_session_id unique index."""
        with self._lock:
            conn = self._conn
            user = self.get_or_create_user(email)
            try:
                conn.execute(
                    """INSERT INTO credit_transactions
                       (user_id, transaction_type, credit_type, amount, stripe_session_id)
                       VALUES (?, 'purchase', ?, ?, ?)""",
                    (user["id"], credit_type, amount, stripe_session_id),
                )
            except sqlite3.IntegrityError:
                # Duplicate stripe_session_id — already processed
                logger.info("Duplicate credit transaction ignored: %s", stripe_session_id)
                return self.get_balances(email)

            col = "file_credits" if credit_type == "file" else "message_credits"
            conn.execute(
                f"UPDATE users SET {col} = {col} + ? WHERE id = ?",
                (amount, user["id"]),
            )
            conn.commit()
            logger.info("Added %d %s credits for %s", amount, credit_type, email)
            return self.get_balances(email)

    def use_credit(self, email: str, credit_type: str, session_id: str) -> bool:
        """Decrement one credit atomically. Returns False if balance is zero."""
        col = "file_credits" if credit_type == "file" else "message_credits"
        with self._lock:
            conn = self._conn
            cursor = conn.execute(
                f"UPDATE users SET {col} = {col} - 1 WHERE email = ? AND {col} > 0",
                (email,),
            )
            if cursor.rowcount != 1:
                return False
            # Record the usage transaction
            user = conn.execute(
                "SELECT id FROM users WHERE email = ?", (email,),
            ).fetchone()
            if user:
                conn.execute(
                    """INSERT INTO credit_transactions
                       (user_id, transaction_type, credit_type, amount, session_id)
                       VALUES (?, 'usage', ?, -1, ?)""",
                    (user["id"], credit_type, session_id),
                )
            conn.commit()
            return True

    def get_balances(self, email: str) -> dict:
        """Return current credit balances for a user."""
        with self._lock:
            row = self._conn.execute(
                "SELECT file_credits, message_credits FROM users WHERE email = ?",
                (email,),
            ).fetchone()
            if row:
                return {"file_credits": row["file_credits"], "message_credits": row["message_credits"]}
            return {"file_credits": 0, "message_credits": 0}

    # ── Subscription methods ────────────────────────────────────────

    def set_subscription(
        self,
        email: str,
        status: str,
        stripe_customer_id: str,
        stripe_subscription_id: str,
        expires_at: str | None,
    ) -> None:
        """Update subscription state for a user."""
        with self._lock:
            self._conn.execute(
                """UPDATE users SET subscription_status = ?, stripe_customer_id = ?,
                   stripe_subscription_id = ?, subscription_expires_at = ?
                   WHERE email = ?""",
                (status, stripe_customer_id, stripe_subscription_id, expires_at, email),
            )
            self._conn.commit()
            logger.info("Subscription updated for %s: %s", email, status)

    def get_subscription(self, email: str) -> dict:
        """Return subscription info for a user."""
        with self._lock:
            row = self._conn.execute(
                """SELECT subscription_status, stripe_customer_id,
                   stripe_subscription_id, subscription_expires_at
                   FROM users WHERE email = ?""",
                (email,),
            ).fetchone()
            if row:
                return dict(row)
            return {"subscription_status": "none", "stripe_customer_id": None,
                    "stripe_subscription_id": None, "subscription_expires_at": None}

    def is_subscriber(self, email: str) -> bool:
        """True if subscription is active or in grace period."""
        sub = self.get_subscription(email)
        return sub["subscription_status"] in ("active", "past_due")

    def record_subscription_event(
        self, stripe_event_id: str, event_type: str, user_id: int,
    ) -> bool:
        """Record a webhook event. Returns False if duplicate (idempotent)."""
        with self._lock:
            try:
                self._conn.execute(
                    """INSERT INTO subscription_events (stripe_event_id, event_type, user_id)
                       VALUES (?, ?, ?)""",
                    (stripe_event_id, event_type, user_id),
                )
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def find_user_by_stripe_customer(self, stripe_customer_id: str) -> dict | None:
        """Lookup user by Stripe customer ID."""
        with self._lock:
            row = self._conn.execute(
                """SELECT id, email, file_credits, message_credits, subscription_status
                   FROM users WHERE stripe_customer_id = ?""",
                (stripe_customer_id,),
            ).fetchone()
            return dict(row) if row else None

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
