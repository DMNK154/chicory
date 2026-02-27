"""Synchronicity detection: dormant reactivation, cross-domain bridges, semantic convergence."""

from __future__ import annotations

import itertools
import json
import math
import statistics
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.layer1.embedding_engine import EmbeddingEngine
from chicory.layer1.tag_manager import TagManager
from chicory.layer2.retrieval_tracker import RetrievalTracker
from chicory.layer2.time_series import multi_tier_decay
from chicory.layer2.trend_engine import TrendEngine
from chicory.layer3.phase_space import PhaseSpace
from chicory.models.phase import Quadrant
from chicory.models.synchronicity import SynchronicityEvent


class SynchronicityDetector:
    """Detects meaningful coincidences from phase space anomalies."""

    def __init__(
        self,
        config: ChicoryConfig,
        db: DatabaseEngine,
        phase_space: PhaseSpace,
        trend_engine: TrendEngine,
        retrieval_tracker: RetrievalTracker,
        tag_manager: TagManager,
        embedding_engine: EmbeddingEngine,
    ) -> None:
        self._config = config
        self._db = db
        self._phase = phase_space
        self._trends = trend_engine
        self._retrieval = retrieval_tracker
        self._tags = tag_manager
        self._embeddings = embedding_engine

    def check_for_synchronicities(self) -> list[SynchronicityEvent]:
        """Run all detection methods and return any events found."""
        events: list[SynchronicityEvent] = []
        events.extend(self.detect_dormant_reactivation())
        events.extend(self.detect_cross_domain_bridges())
        events.extend(self.detect_semantic_convergence())

        for event in events:
            event_id = self.record_event(event)
            event.id = event_id

        return events

    def detect_dormant_reactivation(self) -> list[SynchronicityEvent]:
        """Detect low-trend/high-retrieval anomalies — the Jungian unconscious signal."""
        coords = self._phase.compute_all_coordinates()
        if len(coords) < 3:
            return []

        retrieval_freqs = [c.retrieval_freq for c in coords.values()]
        mean_rf = statistics.mean(retrieval_freqs)
        std_rf = statistics.stdev(retrieval_freqs) if len(retrieval_freqs) > 1 else 0.0

        if std_rf < 1e-6:
            return []

        threshold = self._config.sync_detection_sigma
        events = []

        for tag_id, coord in coords.items():
            if coord.quadrant != Quadrant.DORMANT_REACTIVATION:
                continue

            z_score = (coord.retrieval_freq - mean_rf) / std_rf
            if z_score <= threshold:
                continue
            if coord.temperature >= self._config.sync_inactive_temp_ceiling:
                continue

            # Check for quadrant transition from inactive
            prev_temp = self._trends.get_previous_temperature(tag_id)
            was_inactive = prev_temp is not None and prev_temp < 0.2

            strength = z_score
            if was_inactive:
                strength *= 1.5

            events.append(SynchronicityEvent(
                event_type="low_trend_high_retrieval",
                description=(
                    f"Tag '{coord.tag_name}' is being retrieved at {z_score:.1f}σ "
                    f"above average despite very low trend activity "
                    f"(temperature={coord.temperature:.2f}). "
                    f"{'Jumped from inactive state. ' if was_inactive else ''}"
                    f"This suggests an unconscious pull toward this topic."
                ),
                strength=strength,
                quadrant=Quadrant.DORMANT_REACTIVATION.value,
                involved_tags=json.dumps([tag_id]),
            ))

        return events

    def detect_cross_domain_bridges(self) -> list[SynchronicityEvent]:
        """Detect retrievals that bridge previously unrelated tag clusters."""
        recent = self._retrieval.get_recent_retrievals(
            limit=10, since_hours=self._config.sync_cross_domain_lookback_hours,
        )
        events = []

        for retrieval in recent:
            # Get all tag IDs from the memories returned
            memory_ids = self._retrieval.get_retrieval_result_memory_ids(retrieval.id)
            all_tags: set[int] = set()
            for mid in memory_ids:
                tag_ids = self._tags.get_tag_ids_for_memory(mid)
                all_tags.update(tag_ids)

            if len(all_tags) < 2:
                continue

            # Check all tag pairs for co-occurrence
            for tag_a, tag_b in itertools.combinations(sorted(all_tags), 2):
                co_count = self._tags.get_co_occurrence_count(tag_a, tag_b)
                if co_count > 0:
                    continue  # Already known to co-occur

                # Compute surprise under independence
                total_memories = self._db.execute(
                    "SELECT COUNT(*) as cnt FROM memories WHERE is_archived = 0"
                ).fetchone()["cnt"]

                if total_memories == 0:
                    continue

                freq_a = self._db.execute(
                    "SELECT COUNT(*) as cnt FROM memory_tags WHERE tag_id = ?",
                    (tag_a,),
                ).fetchone()["cnt"] / total_memories

                freq_b = self._db.execute(
                    "SELECT COUNT(*) as cnt FROM memory_tags WHERE tag_id = ?",
                    (tag_b,),
                ).fetchone()["cnt"] / total_memories

                expected = freq_a * freq_b * total_memories
                if expected >= 1.0:
                    continue  # Not actually surprising

                surprise = -math.log(max(expected / total_memories, 1e-10))
                if surprise < self._config.cross_domain_surprise_threshold:
                    continue

                tag_a_name = self._tags.get_by_id(tag_a).name
                tag_b_name = self._tags.get_by_id(tag_b).name

                events.append(SynchronicityEvent(
                    event_type="cross_domain_bridge",
                    description=(
                        f"Retrieval bridged '{tag_a_name}' and '{tag_b_name}' "
                        f"which have never co-occurred (surprise={surprise:.1f}). "
                        f"Query: '{retrieval.query_text[:80]}'"
                    ),
                    strength=surprise,
                    quadrant="cross_domain",
                    involved_tags=json.dumps([tag_a, tag_b]),
                    involved_memories=json.dumps(memory_ids),
                    trigger_retrieval_id=retrieval.id,
                ))

        return events

    def detect_semantic_convergence(self) -> list[SynchronicityEvent]:
        """Detect memories with no shared tags but high embedding similarity
        that were retrieved in separate events."""
        recent = self._retrieval.get_recent_retrievals(
            limit=20, since_hours=self._config.sync_semantic_convergence_lookback_hours,
        )
        if len(recent) < 2:
            return []

        # Collect (memory_id, retrieval_id) pairs
        memory_retrieval_map: dict[str, int] = {}
        for r in recent:
            for mid in self._retrieval.get_retrieval_result_memory_ids(r.id):
                if mid not in memory_retrieval_map:
                    memory_retrieval_map[mid] = r.id

        memory_ids = list(memory_retrieval_map.keys())
        if len(memory_ids) < 2:
            return []

        # Load embeddings
        all_cached = self._embeddings.get_all_cached()
        available = [(mid, all_cached[mid]) for mid in memory_ids if mid in all_cached]
        if len(available) < 2:
            return []

        events = []
        threshold = self._config.semantic_convergence_threshold

        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                mid_a, vec_a = available[i]
                mid_b, vec_b = available[j]

                # Must be from different retrieval events
                if memory_retrieval_map[mid_a] == memory_retrieval_map[mid_b]:
                    continue

                # Must have no shared tags
                tags_a = set(self._tags.get_tag_ids_for_memory(mid_a))
                tags_b = set(self._tags.get_tag_ids_for_memory(mid_b))
                if tags_a & tags_b:
                    continue

                similarity = float(np.dot(vec_a, vec_b))
                if similarity < threshold:
                    continue

                all_tag_ids = list(tags_a | tags_b)
                events.append(SynchronicityEvent(
                    event_type="unexpected_semantic_cluster",
                    description=(
                        f"Memories with no shared tags have embedding similarity "
                        f"of {similarity:.2f}, retrieved in separate events. "
                        f"This suggests an unconscious thematic thread."
                    ),
                    strength=similarity,
                    quadrant="semantic_convergence",
                    involved_tags=json.dumps(all_tag_ids),
                    involved_memories=json.dumps([mid_a, mid_b]),
                ))

        return events

    def record_event(self, event: SynchronicityEvent) -> int:
        """Persist a synchronicity event. Returns the event ID."""
        now = datetime.utcnow().isoformat()
        self._db.execute(
            """
            INSERT INTO synchronicity_events
                (event_type, description, strength, quadrant,
                 involved_tags, involved_memories, trigger_retrieval_id,
                 last_reinforced)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_type,
                event.description,
                event.strength,
                event.quadrant,
                event.involved_tags,
                event.involved_memories,
                event.trigger_retrieval_id,
                now,
            ),
        )
        self._db.connection.commit()
        row = self._db.execute("SELECT last_insert_rowid()").fetchone()
        return row[0]

    def get_recent(self, limit: int = 20, unacknowledged_only: bool = False) -> list[SynchronicityEvent]:
        """Get recent synchronicity events."""
        if unacknowledged_only:
            rows = self._db.execute(
                """
                SELECT * FROM synchronicity_events
                WHERE acknowledged = 0
                ORDER BY detected_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = self._db.execute(
                "SELECT * FROM synchronicity_events ORDER BY detected_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [
            SynchronicityEvent(
                id=r["id"],
                detected_at=datetime.fromisoformat(r["detected_at"]),
                event_type=r["event_type"],
                description=r["description"],
                strength=r["strength"],
                quadrant=r["quadrant"],
                involved_tags=r["involved_tags"],
                involved_memories=r["involved_memories"],
                trigger_retrieval_id=r["trigger_retrieval_id"],
                acknowledged=bool(r["acknowledged"]),
                last_reinforced=datetime.fromisoformat(r["last_reinforced"]) if r["last_reinforced"] else None,
                reinforcement_count=r["reinforcement_count"],
            )
            for r in rows
        ]

    def effective_strength(self, event: SynchronicityEvent) -> float:
        """Compute effective strength with decay and reinforcement boost.

        Formula: strength * min(1 + count * boost_factor, max_boost)
                          * multi_tier_decay(age, tiers)

        Age is measured from last_reinforced (not detected_at).
        """
        now = datetime.utcnow()
        reference_time = event.last_reinforced or event.detected_at or now
        age_hours = (now - reference_time).total_seconds() / 3600

        boost = min(
            1.0 + event.reinforcement_count * self._config.sync_reinforcement_boost_factor,
            self._config.sync_reinforcement_max_boost,
        )

        decay = multi_tier_decay(age_hours, [
            (self._config.sync_decay_active_weight,
             self._config.sync_decay_active_halflife_hours),
            (self._config.sync_decay_longterm_weight,
             self._config.sync_decay_longterm_halflife_hours),
        ])

        return event.strength * boost * decay

    def reinforce_event(self, event_id: int) -> None:
        """Reinforce a synchronicity event: reset decay clock and increment count."""
        now = datetime.utcnow().isoformat()
        self._db.execute(
            """
            UPDATE synchronicity_events
            SET last_reinforced = ?, reinforcement_count = reinforcement_count + 1
            WHERE id = ?
            """,
            (now, event_id),
        )
        self._db.connection.commit()

    def get_events_for_memory(self, memory_id: str) -> list[int]:
        """Return IDs of synchronicity events whose involved_memories contains memory_id."""
        return self.get_events_for_memories([memory_id]).get(memory_id, [])

    def get_events_for_memories(self, memory_ids: list[str]) -> dict[str, list[int]]:
        """Return event IDs for multiple memories in a single table scan."""
        if not memory_ids:
            return {}
        result: dict[str, list[int]] = {mid: [] for mid in memory_ids}
        memory_id_set = set(memory_ids)

        rows = self._db.execute(
            "SELECT id, involved_memories FROM synchronicity_events WHERE involved_memories IS NOT NULL"
        ).fetchall()

        for r in rows:
            involved = json.loads(r["involved_memories"])
            event_id = r["id"]
            for mid in involved:
                if mid in memory_id_set:
                    result[mid].append(event_id)
        return result

    def acknowledge(self, event_id: int) -> None:
        """Mark a synchronicity event as acknowledged."""
        self._db.execute(
            "UPDATE synchronicity_events SET acknowledged = 1 WHERE id = ?",
            (event_id,),
        )
        self._db.connection.commit()
