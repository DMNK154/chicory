"""Composite salience scoring combining LLM judgment and usage metrics."""

from __future__ import annotations

import math
from datetime import datetime

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.layer2.time_series import multi_tier_decay


class SalienceScorer:
    """Computes and updates composite salience scores."""

    def __init__(self, config: ChicoryConfig, db: DatabaseEngine) -> None:
        self._config = config
        self._db = db

    def compute_usage_salience(self, memory_id: str) -> float:
        """Compute usage-based salience from access count, recency, and success rate."""
        row = self._db.execute(
            """
            SELECT access_count, last_accessed,
                   retrieval_success_count, retrieval_total_count
            FROM memories WHERE id = ?
            """,
            (memory_id,),
        ).fetchone()

        if not row:
            return 0.0

        access_count = row["access_count"]
        last_accessed = row["last_accessed"]
        success_count = row["retrieval_success_count"]
        total_count = row["retrieval_total_count"]

        # Access component: log-scaled
        w_access = 0.4
        access_score = math.log(1 + access_count) / math.log(1 + 100)  # Normalize against ~100 accesses
        access_score = min(access_score, 1.0)

        # Recency component: multi-tier exponential decay
        w_recency = 0.4
        if last_accessed:
            last_dt = datetime.fromisoformat(last_accessed)
            hours_ago = (datetime.utcnow() - last_dt).total_seconds() / 3600
            recency_score = multi_tier_decay(hours_ago, [
                (self._config.salience_recency_active_weight,
                 self._config.salience_recency_active_halflife_hours),
                (self._config.salience_recency_longterm_weight,
                 self._config.salience_recency_longterm_halflife_hours),
            ])
        else:
            recency_score = 0.0

        # Success rate component
        w_success = 0.2
        if total_count > 0:
            success_score = success_count / total_count
        else:
            success_score = 0.5  # No data yet, neutral

        raw = w_access * access_score + w_recency * recency_score + w_success * success_score
        # Sigmoid to [0, 1]
        return 1.0 / (1.0 + math.exp(-6 * (raw - 0.5)))

    def compute_composite(self, salience_model: float, salience_usage: float) -> float:
        """Weighted combination of model-judged and usage-based salience."""
        w_m = self._config.salience_model_weight
        w_u = self._config.salience_usage_weight
        return w_m * salience_model + w_u * salience_usage

    def update_on_access(self, memory_id: str) -> None:
        """Update salience scores after a memory is accessed."""
        now = datetime.utcnow().isoformat()
        self._db.execute(
            """
            UPDATE memories SET
                access_count = access_count + 1,
                last_accessed = ?,
                retrieval_total_count = retrieval_total_count + 1,
                updated_at = ?
            WHERE id = ?
            """,
            (now, now, memory_id),
        )
        self._db.connection.commit()

        # Recompute usage salience
        usage = self.compute_usage_salience(memory_id)
        row = self._db.execute(
            "SELECT salience_model FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()

        if row:
            composite = self.compute_composite(row["salience_model"], usage)
            self._db.execute(
                "UPDATE memories SET salience_usage = ?, salience_composite = ? WHERE id = ?",
                (usage, composite, memory_id),
            )
            self._db.connection.commit()

    def update_on_access_batch(self, memory_ids: list[str]) -> None:
        """Update salience scores for multiple memories in a single transaction."""
        if not memory_ids:
            return
        now = datetime.utcnow().isoformat()
        with self._db.transaction():
            # Bulk update access counts
            placeholders = ",".join("?" * len(memory_ids))
            self._db.execute(
                f"""
                UPDATE memories SET
                    access_count = access_count + 1,
                    last_accessed = ?,
                    retrieval_total_count = retrieval_total_count + 1,
                    updated_at = ?
                WHERE id IN ({placeholders})
                """,
                (now, now, *memory_ids),
            )

            # Recompute usage salience for each memory
            rows = self._db.execute(
                f"SELECT id, salience_model FROM memories WHERE id IN ({placeholders})",
                tuple(memory_ids),
            ).fetchall()

            for row in rows:
                usage = self.compute_usage_salience(row["id"])
                composite = self.compute_composite(row["salience_model"], usage)
                self._db.execute(
                    "UPDATE memories SET salience_usage = ?, salience_composite = ? WHERE id = ?",
                    (usage, composite, row["id"]),
                )

    def record_success(self, memory_id: str) -> None:
        """Record that a retrieved memory was useful."""
        self._db.execute(
            """
            UPDATE memories SET retrieval_success_count = retrieval_success_count + 1
            WHERE id = ?
            """,
            (memory_id,),
        )
        self._db.connection.commit()

    def adjust_salience(self, memory_id: str, adjustment: float) -> None:
        """Apply an external salience adjustment (e.g. from meta-pattern feedback)."""
        row = self._db.execute(
            "SELECT salience_model FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return

        new_model = max(0.0, min(1.0, row["salience_model"] + adjustment))
        usage = self.compute_usage_salience(memory_id)
        composite = self.compute_composite(new_model, usage)

        self._db.execute(
            """
            UPDATE memories SET salience_model = ?, salience_usage = ?,
                                salience_composite = ?, updated_at = ?
            WHERE id = ?
            """,
            (new_model, usage, composite, datetime.utcnow().isoformat(), memory_id),
        )
        self._db.connection.commit()
