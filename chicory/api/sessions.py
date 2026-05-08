"""Session management for the Chicory web demo."""

from __future__ import annotations

import logging
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from chicory.config import ChicoryConfig, load_config
from chicory.llm.base import BaseLLMClient
from chicory.llm.factory import create_llm_client
from chicory.orchestrator.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

SESSION_DIR = Path.home() / ".chicory" / "sessions"
NETWORK_DIR = Path.home() / ".chicory" / "networks"
DEFAULT_TIMEOUT_MINUTES = 30
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

# Per-tier limits (0 = unlimited)
TIER_LIMITS = {
    "free": {"files": 3, "chats": 5},
    "pro": {"files": 25, "chats": 100},
    "admin": {"files": 0, "chats": 0},
}


@dataclass
class Session:
    """One user's ephemeral session state."""

    session_id: str
    orchestrator: Orchestrator
    llm_client: BaseLLMClient
    tier: str = "admin"
    email: str | None = None
    file_credits: int = 0
    message_credits: int = 0
    is_subscriber: bool = False
    user_id: int | None = None
    _persistent_path: Path | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    db_path: Path = field(default_factory=Path)
    upload_dir: Path = field(default_factory=Path)
    files_uploaded: int = 0
    chat_turns: int = 0

    @property
    def max_files(self) -> int:
        return TIER_LIMITS.get(self.tier, TIER_LIMITS["free"])["files"]

    @property
    def max_chats(self) -> int:
        return TIER_LIMITS.get(self.tier, TIER_LIMITS["free"])["chats"]

    @property
    def files_limited(self) -> bool:
        if self.email and self.file_credits > 0:
            return False
        return self.max_files > 0 and self.files_uploaded >= self.max_files

    @property
    def chats_limited(self) -> bool:
        if self.email and self.message_credits > 0:
            return False
        return self.max_chats > 0 and self.chat_turns >= self.max_chats

    def touch(self) -> None:
        """Update last-accessed timestamp."""
        self.last_accessed = datetime.utcnow()

    @property
    def expires_at(self) -> datetime:
        return self.last_accessed + timedelta(minutes=DEFAULT_TIMEOUT_MINUTES)

    def close(self, preserve_for_subscriber: bool = False) -> None:
        """Close the orchestrator. For subscribers, persist DB first."""
        try:
            self.orchestrator.close()
        except Exception:
            logger.exception("Error closing orchestrator for session %s", self.session_id)

        if preserve_for_subscriber and self._persistent_path:
            self._persist_network()

        # Remove session directory (temp copy)
        session_dir = self.db_path.parent
        if session_dir.exists():
            shutil.rmtree(session_dir, ignore_errors=True)

    def _persist_network(self) -> None:
        """Copy session DB and uploads to the permanent network directory."""
        if not self._persistent_path:
            return
        dest_dir = self._persistent_path
        dest_dir.mkdir(parents=True, exist_ok=True)

        # WAL checkpoint to flush all data into the main DB file
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
        except Exception:
            logger.exception("WAL checkpoint failed for %s", self.db_path)

        # Copy DB
        dest_db = dest_dir / "chicory.db"
        shutil.copy2(self.db_path, dest_db)

        # Copy uploads
        dest_uploads = dest_dir / "uploads"
        if self.upload_dir.exists():
            if dest_uploads.exists():
                shutil.rmtree(dest_uploads)
            shutil.copytree(self.upload_dir, dest_uploads)

        logger.info("Persisted network for session %s to %s", self.session_id, dest_dir)


class SessionManager:
    """Manages ephemeral per-user sessions with auto-cleanup."""

    def __init__(self, timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._timeout = timedelta(minutes=timeout_minutes)
        self._cleanup_thread: threading.Thread | None = None
        self._running = False

    def start_cleanup_loop(self) -> None:
        """Start the background session cleanup thread."""
        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop, daemon=True,
        )
        self._cleanup_thread.start()

    def stop(self) -> None:
        """Stop cleanup and close all sessions."""
        self._running = False
        with self._lock:
            for session in list(self._sessions.values()):
                session.close(preserve_for_subscriber=session.is_subscriber)
            self._sessions.clear()

    def create_session(self) -> Session:
        """Create a new ephemeral session with its own SQLite DB."""
        session_id = uuid.uuid4().hex[:16]
        session_dir = SESSION_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        db_path = session_dir / "chicory.db"
        upload_dir = session_dir / "uploads"
        upload_dir.mkdir(exist_ok=True)

        config = load_config(
            db_path=db_path,
            commons_enabled=False,
        )
        orchestrator = Orchestrator(config)
        llm_client = create_llm_client(config)

        session = Session(
            session_id=session_id,
            orchestrator=orchestrator,
            llm_client=llm_client,
            db_path=db_path,
            upload_dir=upload_dir,
        )

        with self._lock:
            self._sessions[session_id] = session

        logger.info("Session created: %s (db=%s)", session_id, db_path)
        return session

    def get_session(self, session_id: str) -> Session | None:
        """Look up a session by ID. Returns None if expired or unknown."""
        with self._lock:
            session = self._sessions.get(session_id)
        if session is not None:
            session.touch()
        return session

    def create_session_for_subscriber(self, user_id: int) -> Session:
        """Create a session that restores from the user's persistent network DB."""
        session_id = uuid.uuid4().hex[:16]
        session_dir = SESSION_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        persistent_dir = NETWORK_DIR / str(user_id)
        persistent_db = persistent_dir / "chicory.db"
        persistent_uploads = persistent_dir / "uploads"

        db_path = session_dir / "chicory.db"
        upload_dir = session_dir / "uploads"
        upload_dir.mkdir(exist_ok=True)

        # Restore from persistent storage if it exists
        if persistent_db.exists():
            shutil.copy2(persistent_db, db_path)
            logger.info("Restored persistent DB for user %d", user_id)
        if persistent_uploads.exists():
            shutil.copytree(persistent_uploads, upload_dir, dirs_exist_ok=True)

        config = load_config(db_path=db_path, commons_enabled=False)
        orchestrator = Orchestrator(config)
        llm_client = create_llm_client(config)

        session = Session(
            session_id=session_id,
            orchestrator=orchestrator,
            llm_client=llm_client,
            db_path=db_path,
            upload_dir=upload_dir,
            is_subscriber=True,
            user_id=user_id,
            _persistent_path=persistent_dir,
        )

        with self._lock:
            self._sessions[session_id] = session

        logger.info("Subscriber session created: %s (user=%d)", session_id, user_id)
        return session

    def destroy_session(self, session_id: str) -> bool:
        """Explicitly destroy a session. Returns True if it existed."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close(preserve_for_subscriber=session.is_subscriber)
            logger.info("Session destroyed: %s (persisted=%s)", session_id, session.is_subscriber)
            return True
        return False

    def destroy_session_silent(self, session_id: str) -> None:
        """Remove a session without persisting (for replacing fresh sessions)."""
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            session.close(preserve_for_subscriber=False)

    def _cleanup_loop(self) -> None:
        """Periodically evict expired sessions."""
        while self._running:
            time.sleep(60)
            now = datetime.utcnow()
            expired: list[str] = []

            with self._lock:
                for sid, session in self._sessions.items():
                    if (now - session.last_accessed) > self._timeout:
                        expired.append(sid)

            for sid in expired:
                self.destroy_session(sid)
                logger.info("Session expired and cleaned up: %s", sid)
