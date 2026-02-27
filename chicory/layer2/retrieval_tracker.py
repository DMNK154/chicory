"""Retrieval event logging and per-tag frequency analysis."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Optional

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.layer2.time_series import sigmoid
from chicory.models.retrieval import RetrievalEvent, RetrievalResult


class RetrievalTracker:
    """Logs retrieval events and computes per-tag retrieval frequency."""

    def __init__(self, config: ChicoryConfig, db: DatabaseEngine) -> None:
        self._config = config
        self._db = db

    def log_retrieval(
        self,
        query_text: str,
        method: str,
        results: list[tuple[str, int, float]],  # (memory_id, rank, relevance_score)
        model_version: str,
        context_summary: str | None = None,
    ) -> int:
        """Log a retrieval event and its results. Returns the retrieval_id."""
        self._db.execute(
            """
            INSERT INTO retrieval_events (query_text, context_summary, method, result_count, model_version)
            VALUES (?, ?, ?, ?, ?)
            """,
            (query_text, context_summary, method, len(results), model_version),
        )
        self._db.connection.commit()

        retrieval_id = self._db.execute("SELECT last_insert_rowid()").fetchone()[0]

        for memory_id, rank, score in results:
            self._db.execute(
                """
                INSERT INTO retrieval_results (retrieval_id, memory_id, rank, relevance_score)
                VALUES (?, ?, ?, ?)
                """,
                (retrieval_id, memory_id, rank, score),
            )
        self._db.connection.commit()

        return retrieval_id

    def log_tag_hits(
        self,
        retrieval_id: int,
        tag_hits: list[tuple[int, str]],  # (tag_id, hit_type)
    ) -> None:
        """Record which tags were involved in a retrieval event."""
        for tag_id, hit_type in tag_hits:
            self._db.execute(
                """
                INSERT OR IGNORE INTO retrieval_tag_hits (retrieval_id, tag_id, hit_type)
                VALUES (?, ?, ?)
                """,
                (retrieval_id, tag_id, hit_type),
            )
        self._db.connection.commit()

    def get_tag_retrieval_frequency(
        self, tag_id: int, window_hours: float | None = None
    ) -> float:
        """Raw retrieval frequency for a tag (events per hour)."""
        W = window_hours or self._config.trend_window_hours
        cutoff = (datetime.utcnow() - timedelta(hours=W)).isoformat()

        row = self._db.execute(
            """
            SELECT COUNT(*) as cnt FROM retrieval_tag_hits rth
            JOIN retrieval_events re ON rth.retrieval_id = re.id
            WHERE rth.tag_id = ? AND re.occurred_at > ?
            """,
            (tag_id, cutoff),
        ).fetchone()

        count = row["cnt"] if row else 0
        return count / W if W > 0 else 0.0

    def get_base_rate(self, window_hours: float | None = None) -> float:
        """Base retrieval rate: total tag hits / (num_active_tags * window_hours)."""
        W = window_hours or self._config.trend_window_hours
        cutoff = (datetime.utcnow() - timedelta(hours=W)).isoformat()

        total_row = self._db.execute(
            """
            SELECT COUNT(*) as cnt FROM retrieval_tag_hits rth
            JOIN retrieval_events re ON rth.retrieval_id = re.id
            WHERE re.occurred_at > ?
            """,
            (cutoff,),
        ).fetchone()

        tag_count_row = self._db.execute(
            "SELECT COUNT(*) as cnt FROM tags WHERE is_active = 1"
        ).fetchone()

        total = total_row["cnt"] if total_row else 0
        num_tags = tag_count_row["cnt"] if tag_count_row else 1

        return total / (num_tags * W) if (num_tags * W) > 0 else 0.0

    def get_normalized_frequency(
        self, tag_id: int, window_hours: float | None = None
    ) -> float:
        """Normalized retrieval frequency (ratio to base rate), mapped to [0,1]."""
        raw = self.get_tag_retrieval_frequency(tag_id, window_hours)
        base = self.get_base_rate(window_hours)

        if base <= 0:
            # No retrievals at all — return 0 (no signal)
            return 0.0 if raw == 0 else 0.9

        ratio = raw / base
        # Map via sigmoid(log(ratio)) so ratio=1 maps to 0.5
        if ratio <= 0:
            return 0.0
        return sigmoid(math.log(ratio))

    def get_recent_retrievals(
        self, limit: int = 20, since_hours: float | None = None
    ) -> list[RetrievalEvent]:
        """Get recent retrieval events."""
        if since_hours is not None:
            cutoff = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat()
            rows = self._db.execute(
                """
                SELECT * FROM retrieval_events
                WHERE occurred_at > ? ORDER BY occurred_at DESC LIMIT ?
                """,
                (cutoff, limit),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM retrieval_events ORDER BY occurred_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [self._row_to_event(r) for r in rows]

    def get_retrieval_result_memory_ids(self, retrieval_id: int) -> list[str]:
        """Get memory IDs from a retrieval event's results."""
        rows = self._db.execute(
            "SELECT memory_id FROM retrieval_results WHERE retrieval_id = ? ORDER BY rank",
            (retrieval_id,),
        ).fetchall()
        return [r["memory_id"] for r in rows]

    def get_retrieval_tag_ids(self, retrieval_id: int) -> list[int]:
        """Get tag IDs that were hit in a retrieval event."""
        rows = self._db.execute(
            "SELECT tag_id FROM retrieval_tag_hits WHERE retrieval_id = ?",
            (retrieval_id,),
        ).fetchall()
        return [r["tag_id"] for r in rows]

    def record_usefulness(
        self, retrieval_id: int, memory_id: str, was_useful: bool
    ) -> None:
        """Record whether a retrieval result was useful."""
        self._db.execute(
            """
            UPDATE retrieval_results SET was_useful = ?
            WHERE retrieval_id = ? AND memory_id = ?
            """,
            (1 if was_useful else 0, retrieval_id, memory_id),
        )
        self._db.connection.commit()

    @staticmethod
    def _row_to_event(row) -> RetrievalEvent:
        return RetrievalEvent(
            id=row["id"],
            query_text=row["query_text"],
            context_summary=row["context_summary"],
            method=row["method"],
            occurred_at=datetime.fromisoformat(row["occurred_at"]),
            result_count=row["result_count"],
            model_version=row["model_version"],
        )
