"""Meta-pattern detection over synchronicity events."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

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
        events = self._get_analysis_events()

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

    _META_EVENT_CAP = 500

    def _get_analysis_events(self) -> list[SynchronicityEvent]:
        """Get deduplicated sync events for meta-pattern analysis.

        Combines time-windowed recent events with lattice-resonant events
        (wider window) using SQL UNION for server-side dedup.  Capped at
        _META_EVENT_CAP strongest events to bound downstream work.
        """
        window = self._config.meta_analysis_interval_hours * 7
        cutoff = (datetime.utcnow() - timedelta(hours=window)).isoformat()
        cap = self._META_EVENT_CAP

        use_lattice = (
            self._sync_engine is not None
            and self._config.meta_use_lattice_resonances
        )

        if use_lattice:
            lattice_cutoff = (
                datetime.utcnow() - timedelta(hours=window * 4)
            ).isoformat()
            rows = self._db.execute(
                """
                SELECT * FROM (
                    SELECT * FROM synchronicity_events
                    WHERE detected_at > ?
                    UNION
                    SELECT se.* FROM synchronicity_events se
                    WHERE se.detected_at > ?
                      AND se.id IN (
                          SELECT event_a_id FROM resonances
                          UNION ALL
                          SELECT event_b_id FROM resonances
                      )
                )
                ORDER BY strength DESC, detected_at DESC
                LIMIT ?
                """,
                (cutoff, lattice_cutoff, cap),
            ).fetchall()
        else:
            rows = self._db.execute(
                """
                SELECT * FROM synchronicity_events
                WHERE detected_at > ?
                ORDER BY strength DESC, detected_at DESC
                LIMIT ?
                """,
                (cutoff, cap),
            ).fetchall()

        return [self._row_to_event(r) for r in rows]

    @staticmethod
    def _row_to_event(r) -> SynchronicityEvent:
        return SynchronicityEvent(
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

    def _cluster_sync_events(
        self, events: list[SynchronicityEvent]
    ) -> list[list[SynchronicityEvent]]:
        """Cluster synchronicity events by tag overlap using sparse union-find.

        Uses an inverted index (tag → event indices) to find candidate pairs
        that share at least one tag, then merges pairs whose Jaccard similarity
        meets the threshold.  Memory is O(n), not O(n²).
        """
        if len(events) < 2:
            return [events] if events else []

        tag_sets = [set(json.loads(e.involved_tags)) for e in events]
        n = len(events)
        sim_threshold = 1.0 - self._config.clustering_jaccard_threshold

        # Inverted index: tag → list of event indices that contain it
        tag_to_events: dict[int, list[int]] = defaultdict(list)
        for i, ts in enumerate(tag_sets):
            for tag in ts:
                tag_to_events[tag].append(i)

        # Union-Find with path compression + union by rank
        parent = list(range(n))
        uf_rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            a, b = find(a), find(b)
            if a == b:
                return
            if uf_rank[a] < uf_rank[b]:
                a, b = b, a
            parent[b] = a
            if uf_rank[a] == uf_rank[b]:
                uf_rank[a] += 1

        for indices in tag_to_events.values():
            if len(indices) > 50:
                indices = indices[:50]
            for a in range(len(indices)):
                i = indices[a]
                for b in range(a + 1, len(indices)):
                    j = indices[b]
                    if find(i) == find(j):
                        continue
                    intersection = len(tag_sets[i] & tag_sets[j])
                    union_size = len(tag_sets[i] | tag_sets[j])
                    if intersection / max(union_size, 1) >= sim_threshold:
                        union(i, j)

        clusters: dict[int, list[SynchronicityEvent]] = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(events[i])

        return list(clusters.values())

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
        """Group tags into clusters based on co-retrieval edges.

        Uses the pre-computed centroid_edges table (maintained by
        CentroidSubgraph) instead of a quadratic self-join on memory_tags.
        """
        if not tag_ids:
            return []

        tag_list = list(tag_ids)
        adjacency: dict[int, set[int]] = {t: set() for t in tag_list}
        tag_set = tag_ids

        placeholders = ",".join("?" * len(tag_list))
        rows = self._db.execute(
            f"""
            SELECT tag_a_id, tag_b_id FROM centroid_edges
            WHERE tag_a_id IN ({placeholders})
              AND tag_b_id IN ({placeholders})
              AND strength > 0.05
            """,
            (*tag_list, *tag_list),
        ).fetchall()

        for r in rows:
            a, b = r["tag_a_id"], r["tag_b_id"]
            if a in tag_set and b in tag_set:
                adjacency[a].add(b)
                adjacency[b].add(a)

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

        sample_ids = list(all_tags)[:5]
        ph = ",".join("?" * len(sample_ids))
        tag_names = [
            r["name"]
            for r in self._db.execute(
                f"SELECT name FROM tags WHERE id IN ({ph})", sample_ids
            ).fetchall()
        ]

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
        cursor = self._db.execute(
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
        return cursor.lastrowid

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
