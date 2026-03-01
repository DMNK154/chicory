"""Sliding-window tag trend computation with derivatives."""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta
from typing import Optional

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.layer2.time_series import (
    sigmoid,
    split_window_derivative,
    three_part_jerk,
    weighted_sum_with_decay,
)
from chicory.models.trend import TrendSnapshot, TrendVector


class TrendEngine:
    """Computes sliding-window tag trend signals."""

    def __init__(self, config: ChicoryConfig, db: DatabaseEngine) -> None:
        self._config = config
        self._db = db
        self._norm_factor: float | None = None

    def record_event(
        self,
        tag_id: int,
        event_type: str,
        memory_id: str | None = None,
        weight: float = 1.0,
        metadata: str | None = None,
    ) -> None:
        """Record a tag event."""
        self._db.execute(
            """
            INSERT INTO tag_events (tag_id, event_type, memory_id, weight, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (tag_id, event_type, memory_id, weight, metadata),
        )
        self._db.connection.commit()
        self._norm_factor = None  # Invalidate cache

    def record_events_batch(
        self,
        events: list[tuple[int, str, str | None, float]],
    ) -> None:
        """Record multiple tag events in a single transaction.

        Each event is (tag_id, event_type, memory_id, weight).
        """
        if not events:
            return
        self._db.executemany(
            """
            INSERT INTO tag_events (tag_id, event_type, memory_id, weight)
            VALUES (?, ?, ?, ?)
            """,
            events,
        )
        self._db.connection.commit()
        self._norm_factor = None  # Invalidate cache once

    def compute_trend(
        self, tag_id: int, window_hours: float | None = None
    ) -> TrendVector:
        """Compute the trend vector for a single tag."""
        W = window_hours or self._config.trend_window_hours
        halflife = W / 2
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=W)

        rows = self._db.execute(
            "SELECT occurred_at, weight FROM tag_events WHERE tag_id = ? AND occurred_at > ?",
            (tag_id, cutoff.isoformat()),
        ).fetchall()

        if not rows:
            return TrendVector(
                tag_id=tag_id, level=0.0, velocity=0.0,
                jerk=0.0, temperature=0.0, event_count=0,
            )

        # Convert to (age_hours, weight) pairs
        events = []
        for r in rows:
            t = datetime.fromisoformat(r["occurred_at"])
            age = (now - t).total_seconds() / 3600
            events.append((age, r["weight"]))

        level = weighted_sum_with_decay(events, halflife)
        velocity = split_window_derivative(events, W, halflife)
        jerk = three_part_jerk(events, W, halflife)

        # Temperature: composite, normalized
        w_l = self._config.trend_level_weight
        w_v = self._config.trend_velocity_weight
        w_j = self._config.trend_jerk_weight
        raw = w_l * level + w_v * max(0, velocity) + w_j * max(0, jerk)

        norm = self._get_normalization_factor()
        temperature = sigmoid(raw / norm) if norm > 0 else 0.5

        return TrendVector(
            tag_id=tag_id,
            level=level,
            velocity=velocity,
            jerk=jerk,
            temperature=temperature,
            event_count=len(events),
        )

    def compute_all_trends(self) -> dict[int, TrendVector]:
        """Compute trends for all active tags in a single query.

        Fetches all tag events in one pass and groups by tag_id in Python,
        avoiding O(t) separate queries.
        """
        W = self._config.trend_window_hours
        halflife = W / 2
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=W)

        # Single query: all events for all active tags within the window
        rows = self._db.execute(
            """
            SELECT te.tag_id, te.occurred_at, te.weight
            FROM tag_events te
            JOIN tags t ON te.tag_id = t.id
            WHERE t.is_active = 1 AND te.occurred_at > ?
            ORDER BY te.tag_id
            """,
            (cutoff.isoformat(),),
        ).fetchall()

        # Group events by tag_id
        from collections import defaultdict
        events_by_tag: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for r in rows:
            t = datetime.fromisoformat(r["occurred_at"])
            age = (now - t).total_seconds() / 3600
            events_by_tag[r["tag_id"]].append((age, r["weight"]))

        # Also include active tags with zero events
        tag_rows = self._db.execute(
            "SELECT id FROM tags WHERE is_active = 1"
        ).fetchall()

        norm = self._get_normalization_factor()
        w_l = self._config.trend_level_weight
        w_v = self._config.trend_velocity_weight
        w_j = self._config.trend_jerk_weight

        result: dict[int, TrendVector] = {}
        for tr in tag_rows:
            tag_id = tr["id"]
            events = events_by_tag.get(tag_id)
            if not events:
                result[tag_id] = TrendVector(
                    tag_id=tag_id, level=0.0, velocity=0.0,
                    jerk=0.0, temperature=0.0, event_count=0,
                )
                continue

            level = weighted_sum_with_decay(events, halflife)
            velocity = split_window_derivative(events, W, halflife)
            jerk = three_part_jerk(events, W, halflife)

            raw = w_l * level + w_v * max(0, velocity) + w_j * max(0, jerk)
            temperature = sigmoid(raw / norm) if norm > 0 else 0.5

            result[tag_id] = TrendVector(
                tag_id=tag_id, level=level, velocity=velocity,
                jerk=jerk, temperature=temperature, event_count=len(events),
            )

        return result

    def snapshot_trends(self) -> None:
        """Persist current trends for all active tags."""
        trends = self.compute_all_trends()
        W = self._config.trend_window_hours
        for tag_id, tv in trends.items():
            self._db.execute(
                """
                INSERT INTO trend_snapshots
                    (tag_id, window_hours, level, velocity, jerk, temperature, event_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (tag_id, W, tv.level, tv.velocity, tv.jerk, tv.temperature, tv.event_count),
            )
        self._db.connection.commit()

    def get_trend_history(
        self, tag_id: int, periods: int = 30
    ) -> list[TrendSnapshot]:
        """Get recent trend snapshots for a tag."""
        rows = self._db.execute(
            """
            SELECT * FROM trend_snapshots WHERE tag_id = ?
            ORDER BY computed_at DESC LIMIT ?
            """,
            (tag_id, periods),
        ).fetchall()
        return [
            TrendSnapshot(
                id=r["id"],
                tag_id=r["tag_id"],
                computed_at=datetime.fromisoformat(r["computed_at"]),
                window_hours=r["window_hours"],
                level=r["level"],
                velocity=r["velocity"],
                jerk=r["jerk"],
                temperature=r["temperature"],
                event_count=r["event_count"],
            )
            for r in rows
        ]

    def get_previous_temperature(self, tag_id: int) -> float | None:
        """Get the most recent snapshot temperature for a tag."""
        row = self._db.execute(
            """
            SELECT temperature FROM trend_snapshots
            WHERE tag_id = ? ORDER BY computed_at DESC LIMIT 1
            """,
            (tag_id,),
        ).fetchone()
        return row["temperature"] if row else None

    def _get_normalization_factor(self) -> float:
        """Compute 90th-percentile raw score across all tags for normalization.

        Uses a single query to fetch all tag events, groups by tag in Python.
        """
        if self._norm_factor is not None:
            return self._norm_factor

        from collections import defaultdict

        W = self._config.trend_window_hours
        halflife = W / 2
        now = datetime.utcnow()
        cutoff = now - timedelta(hours=W)

        tag_rows = self._db.execute(
            "SELECT id FROM tags WHERE is_active = 1"
        ).fetchall()

        if not tag_rows:
            self._norm_factor = 1.0
            return 1.0

        # Single query: all events for all active tags
        event_rows = self._db.execute(
            """
            SELECT te.tag_id, te.occurred_at, te.weight
            FROM tag_events te
            JOIN tags t ON te.tag_id = t.id
            WHERE t.is_active = 1 AND te.occurred_at > ?
            """,
            (cutoff.isoformat(),),
        ).fetchall()

        events_by_tag: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for r in event_rows:
            t = datetime.fromisoformat(r["occurred_at"])
            age = (now - t).total_seconds() / 3600
            events_by_tag[r["tag_id"]].append((age, r["weight"]))

        w_l = self._config.trend_level_weight
        w_v = self._config.trend_velocity_weight
        w_j = self._config.trend_jerk_weight

        raw_scores = []
        for tr in tag_rows:
            events = events_by_tag.get(tr["id"])
            if not events:
                raw_scores.append(0.0)
                continue

            level = weighted_sum_with_decay(events, halflife)
            velocity = split_window_derivative(events, W, halflife)
            jerk = three_part_jerk(events, W, halflife)
            raw = w_l * level + w_v * max(0, velocity) + w_j * max(0, jerk)
            raw_scores.append(raw)

        if not raw_scores or max(raw_scores) == 0:
            self._norm_factor = 1.0
        else:
            sorted_scores = sorted(raw_scores)
            idx = int(0.9 * len(sorted_scores))
            p90 = sorted_scores[min(idx, len(sorted_scores) - 1)]
            self._norm_factor = max(p90, 0.01)

        return self._norm_factor
