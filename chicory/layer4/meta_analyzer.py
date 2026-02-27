"""Meta-pattern detection over synchronicity events."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

import numpy as np

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.layer4.adaptive_thresholds import AdaptiveThresholds
from chicory.models.meta_pattern import MetaPattern
from chicory.models.synchronicity import SynchronicityEvent

if TYPE_CHECKING:
    from chicory.layer3.synchronicity_engine import SynchronicityEngine


class MetaAnalyzer:
    """Periodic batch analysis over synchronicity events to find meta-patterns."""

    def __init__(
        self,
        config: ChicoryConfig,
        db: DatabaseEngine,
        adaptive_thresholds: AdaptiveThresholds,
        sync_engine: SynchronicityEngine | None = None,
    ) -> None:
        self._config = config
        self._db = db
        self._thresholds = adaptive_thresholds
        self._sync_engine = sync_engine

    def run_analysis(self) -> list[MetaPattern]:
        """Run meta-pattern analysis. Returns newly detected patterns."""
        events = self._get_recent_sync_events()

        # Merge in lattice-resonant events (permanent, not time-windowed)
        lattice_events = self._get_lattice_resonance_events()
        seen_ids = {e.id for e in events}
        for le in lattice_events:
            if le.id not in seen_ids:
                events.append(le)
                seen_ids.add(le.id)

        if len(events) < self._config.meta_min_sync_events:
            return []

        clusters = self._cluster_sync_events(events)
        patterns = []

        total_events = len(events)
        total_tags = self._get_active_tag_count()

        for cluster in clusters:
            pattern = self._evaluate_cluster(cluster, total_events, total_tags)
            if pattern is not None:
                self._record_pattern(pattern)
                patterns.append(pattern)

        return patterns

    def _get_recent_sync_events(self) -> list[SynchronicityEvent]:
        """Get synchronicity events from the analysis window."""
        window = self._config.meta_analysis_interval_hours * 7  # Look back 7 analysis periods
        cutoff = (datetime.utcnow() - timedelta(hours=window)).isoformat()

        rows = self._db.execute(
            "SELECT * FROM synchronicity_events WHERE detected_at > ? ORDER BY detected_at",
            (cutoff,),
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

    def _get_lattice_resonance_events(self) -> list[SynchronicityEvent]:
        """Get sync events that participate in lattice resonances.

        These are NOT time-windowed — they represent permanent structural
        patterns from the Prime Ramsey Lattice.  Reads from the persisted
        resonances table instead of recomputing O(n²) pairwise comparisons.
        """
        if not self._sync_engine or not self._config.meta_use_lattice_resonances:
            return []

        rows = self._db.execute(
            """
            SELECT se.* FROM synchronicity_events se
            WHERE se.id IN (
                SELECT event_a_id FROM resonances
                UNION
                SELECT event_b_id FROM resonances
            )
            """
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

    def _cluster_sync_events(
        self, events: list[SynchronicityEvent]
    ) -> list[list[SynchronicityEvent]]:
        """Cluster synchronicity events by tag overlap using agglomerative clustering."""
        if len(events) < 2:
            return [events] if events else []

        tag_sets = [set(json.loads(e.involved_tags)) for e in events]
        n = len(events)

        # Build distance matrix (Jaccard distance)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                intersection = len(tag_sets[i] & tag_sets[j])
                union = len(tag_sets[i] | tag_sets[j])
                jaccard = intersection / max(union, 1)
                dist = 1 - jaccard
                distances[i][j] = dist
                distances[j][i] = dist

        try:
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import squareform

            condensed = squareform(distances)
            Z = linkage(condensed, method="average")
            labels = fcluster(Z, t=self._config.clustering_jaccard_threshold, criterion="distance")

            clusters: dict[int, list[SynchronicityEvent]] = defaultdict(list)
            for i, label in enumerate(labels):
                clusters[label].append(events[i])

            return list(clusters.values())
        except (ImportError, ValueError):
            # Fallback: treat each event as its own cluster
            return [[e] for e in events]

    def _evaluate_cluster(
        self,
        cluster: list[SynchronicityEvent],
        total_events: int,
        total_tags: int,
    ) -> MetaPattern | None:
        """Evaluate whether a cluster constitutes a meta-pattern."""
        n = len(cluster)
        if n < self._config.meta_min_sync_events:
            return None

        # Compute the cluster's tag share
        all_cluster_tags: set[int] = set()
        for e in cluster:
            all_cluster_tags.update(json.loads(e.involved_tags))

        if total_tags == 0:
            return None

        tag_share = len(all_cluster_tags) / total_tags

        # Expected count under uniform distribution
        expected = max(total_events * tag_share, 0.01)
        ratio = n / expected

        threshold = self._thresholds.get_threshold("meta_base_rate_multiplier")
        if ratio < threshold:
            return None

        # Cross-domain validation
        tag_clusters = self._get_tag_clusters(all_cluster_tags)
        is_cross_domain = len(tag_clusters) >= self._config.meta_cross_domain_min_clusters

        if is_cross_domain:
            pattern_type = "cross_domain_theme"
            confidence = min(1.0, ratio / (threshold * 2))
        else:
            pattern_type = "recurring_sync"
            confidence = min(1.0, ratio / (threshold * 3))  # Require stronger signal

        sync_ids = [e.id for e in cluster if e.id is not None]
        tag_cluster_list = [list(tc) for tc in tag_clusters]

        return MetaPattern(
            description=self._describe_pattern(cluster, tag_clusters, ratio),
            pattern_type=pattern_type,
            confidence=confidence,
            involved_sync_ids=json.dumps(sync_ids),
            involved_tag_clusters=json.dumps(tag_cluster_list),
        )

    def _get_tag_clusters(self, tag_ids: set[int]) -> list[set[int]]:
        """Group tags into clusters based on co-occurrence.
        Tags that frequently co-occur are in the same cluster."""
        if not tag_ids:
            return []

        tag_list = list(tag_ids)
        # Build adjacency based on co-occurrence
        adjacency: dict[int, set[int]] = {t: set() for t in tag_list}

        for i in range(len(tag_list)):
            for j in range(i + 1, len(tag_list)):
                # Check co-occurrence
                row = self._db.execute(
                    """
                    SELECT COUNT(*) as cnt FROM memory_tags a
                    JOIN memory_tags b ON a.memory_id = b.memory_id
                    WHERE a.tag_id = ? AND b.tag_id = ?
                    """,
                    (tag_list[i], tag_list[j]),
                ).fetchone()
                if row and row["cnt"] > 2:  # Threshold: co-occurred in >2 memories
                    adjacency[tag_list[i]].add(tag_list[j])
                    adjacency[tag_list[j]].add(tag_list[i])

        # Connected components
        visited: set[int] = set()
        clusters: list[set[int]] = []

        for tag in tag_list:
            if tag in visited:
                continue
            component: set[int] = set()
            stack = [tag]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                stack.extend(adjacency[node] - visited)
            clusters.append(component)

        return clusters

    def _describe_pattern(
        self,
        cluster: list[SynchronicityEvent],
        tag_clusters: list[set[int]],
        ratio: float,
    ) -> str:
        """Generate a human-readable description of a meta-pattern."""
        n = len(cluster)
        n_clusters = len(tag_clusters)
        event_types = set(e.event_type for e in cluster)

        # Get tag names for the involved tags
        all_tags: set[int] = set()
        for e in cluster:
            all_tags.update(json.loads(e.involved_tags))

        tag_names = []
        for tid in list(all_tags)[:5]:
            row = self._db.execute(
                "SELECT name FROM tags WHERE id = ?", (tid,)
            ).fetchone()
            if row:
                tag_names.append(row["name"])

        tags_str = ", ".join(tag_names)
        types_str = ", ".join(event_types)

        return (
            f"Recurring meta-pattern detected: {n} synchronicity events "
            f"({types_str}) at {ratio:.1f}x expected frequency, "
            f"spanning {n_clusters} tag cluster(s). "
            f"Key tags: {tags_str}"
        )

    def _record_pattern(self, pattern: MetaPattern) -> int:
        """Persist a meta-pattern."""
        self._db.execute(
            """
            INSERT INTO meta_patterns
                (description, pattern_type, confidence,
                 involved_sync_ids, involved_tag_clusters, actions_taken)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                pattern.description,
                pattern.pattern_type,
                pattern.confidence,
                pattern.involved_sync_ids,
                pattern.involved_tag_clusters,
                pattern.actions_taken,
            ),
        )
        self._db.connection.commit()
        row = self._db.execute("SELECT last_insert_rowid()").fetchone()
        return row[0]

    def _get_active_tag_count(self) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) as cnt FROM tags WHERE is_active = 1"
        ).fetchone()
        return row["cnt"] if row else 0

    def get_active_patterns(self) -> list[MetaPattern]:
        """Get all active meta-patterns."""
        rows = self._db.execute(
            "SELECT * FROM meta_patterns WHERE is_active = 1 ORDER BY detected_at DESC"
        ).fetchall()
        return [
            MetaPattern(
                id=r["id"],
                detected_at=datetime.fromisoformat(r["detected_at"]),
                description=r["description"],
                pattern_type=r["pattern_type"],
                confidence=r["confidence"],
                involved_sync_ids=r["involved_sync_ids"],
                involved_tag_clusters=r["involved_tag_clusters"],
                actions_taken=r["actions_taken"],
                is_active=bool(r["is_active"]),
                validated_by=r["validated_by"],
            )
            for r in rows
        ]
