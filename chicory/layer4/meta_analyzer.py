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

        cluster_groups, subgraphs = self._cluster_sync_events(events)
        patterns = []

        total_events = len(events)
        total_tags = self._get_active_tag_count()

        for sg_id, cluster in cluster_groups.items():
            sg_tags = subgraphs.get(sg_id, set())
            pattern = self._evaluate_cluster(
                cluster, total_events, total_tags, sg_tags,
            )
            if pattern is not None:
                self._record_pattern(pattern)
                patterns.append(pattern)

        return patterns

    def _get_analysis_events(self) -> list[SynchronicityEvent]:
        """Get deduplicated sync events for meta-pattern analysis.

        Fetches recent events via indexed detected_at scan, then expands
        through lattice resonances using a JOIN-based subquery (avoids
        SQLite parameter limits on large event sets).
        """
        window = self._config.meta_analysis_interval_hours * 7
        cutoff = (datetime.utcnow() - timedelta(hours=window)).isoformat()

        rows = self._db.execute(
            "SELECT * FROM synchronicity_events "
            "WHERE detected_at > ? "
            "ORDER BY strength DESC, detected_at DESC",
            (cutoff,),
        ).fetchall()

        by_id = {r["id"]: r for r in rows}

        use_lattice = (
            self._sync_engine is not None
            and self._config.meta_use_lattice_resonances
            and by_id
        )

        if use_lattice:
            lattice_cutoff = (
                datetime.utcnow() - timedelta(hours=window * 4)
            ).isoformat()
            rows_a = self._db.execute(
                """SELECT r.event_b_id AS pid
                   FROM resonances r
                   JOIN synchronicity_events se ON r.event_a_id = se.id
                   WHERE se.detected_at > ?""",
                (cutoff,),
            ).fetchall()
            rows_b = self._db.execute(
                """SELECT r.event_a_id AS pid
                   FROM resonances r
                   JOIN synchronicity_events se ON r.event_b_id = se.id
                   WHERE se.detected_at > ?""",
                (cutoff,),
            ).fetchall()
            partner_pids = {r["pid"] for r in rows_a} | {r["pid"] for r in rows_b}

            missing = [pid for pid in partner_pids if pid not in by_id]
            if missing:
                for i in range(0, len(missing), 500):
                    chunk = missing[i:i + 500]
                    ph = ",".join("?" * len(chunk))
                    extra = self._db.execute(
                        f"SELECT * FROM synchronicity_events "
                        f"WHERE id IN ({ph}) AND detected_at > ?",
                        chunk + [lattice_cutoff],
                    ).fetchall()
                    for r in extra:
                        by_id[r["id"]] = r

        return [self._row_to_event(r) for r in by_id.values()]

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
        self, events: list[SynchronicityEvent],
    ) -> tuple[dict[int, list[SynchronicityEvent]], dict[int, set[int]]]:
        """Cluster sync events by tag-tensor subgraph connectivity.

        Extracts all tags involved in the events, pulls their edges from
        the tag relational tensor, finds connected subgraphs via
        union-find on tags, then maps each event to the subgraph that
        contains its tags.  Events spanning multiple subgraphs go to
        whichever subgraph has the most tag overlap.

        Returns (cluster_groups, subgraph_tags) so callers don't need
        to re-derive subgraph membership.
        """
        tag_sets = [set(json.loads(e.involved_tags)) for e in events]
        all_tags = sorted({t for ts in tag_sets for t in ts})

        if not all_tags:
            return {0: events}, {0: set()}

        tag_subgraphs = self._find_tensor_subgraphs(all_tags)

        tag_to_sg: dict[int, int] = {}
        sg_tags: dict[int, set[int]] = {}
        for sg_id, members in enumerate(tag_subgraphs):
            sg_tags[sg_id] = members
            for t in members:
                tag_to_sg[t] = sg_id

        clusters: dict[int, list[SynchronicityEvent]] = defaultdict(list)
        for i, event in enumerate(events):
            sg_votes: dict[int, int] = defaultdict(int)
            for t in tag_sets[i]:
                if t in tag_to_sg:
                    sg_votes[tag_to_sg[t]] += 1
            if sg_votes:
                best = max(sg_votes, key=sg_votes.get)
                clusters[best].append(event)
            else:
                clusters[-1].append(event)

        return dict(clusters), sg_tags

    def _find_tensor_subgraphs(self, tag_ids: list[int]) -> list[set[int]]:
        """Find connected subgraphs in the tag relational tensor.

        Queries all tensor edges among the given tags with meaningful
        composite strength and clusters via union-find.
        """
        ph = ",".join("?" * len(tag_ids))
        rows = self._db.execute(
            f"""
            SELECT tag_a_id, tag_b_id,
                   cooccurrence_strength + synchronicity_strength
                   + semantic_strength + glyph_strength
                   - inhibition_strength AS composite
            FROM tag_relational_tensor
            WHERE tag_a_id IN ({ph})
              AND tag_b_id IN ({ph})
              AND (cooccurrence_strength + synchronicity_strength
                   + semantic_strength + glyph_strength
                   - inhibition_strength) > 0.05
            """,
            (*tag_ids, *tag_ids),
        ).fetchall()

        parent: dict[int, int] = {t: t for t in tag_ids}
        rank: dict[int, int] = {t: 0 for t in tag_ids}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def merge(a: int, b: int) -> None:
            a, b = find(a), find(b)
            if a == b:
                return
            if rank[a] < rank[b]:
                a, b = b, a
            parent[b] = a
            if rank[a] == rank[b]:
                rank[a] += 1

        for r in rows:
            merge(r["tag_a_id"], r["tag_b_id"])

        components: dict[int, set[int]] = defaultdict(set)
        for t in tag_ids:
            components[find(t)].add(t)

        return list(components.values())

    def _evaluate_cluster(
        self,
        cluster: list[SynchronicityEvent],
        total_events: int,
        total_tags: int,
        subgraph_tags: set[int],
    ) -> MetaPattern | None:
        """Evaluate whether a cluster constitutes a meta-pattern."""
        n = len(cluster)
        if n < self._config.meta_min_sync_events:
            return None

        all_cluster_tags: set[int] = set()
        for e in cluster:
            all_cluster_tags.update(json.loads(e.involved_tags))

        if total_tags == 0:
            return None

        tag_share = len(all_cluster_tags) / total_tags
        expected = max(total_events * tag_share, 0.01)
        ratio = n / expected

        threshold = self._thresholds.get_threshold("meta_base_rate_multiplier")
        if ratio < threshold:
            return None

        # Cross-domain: check if the subgraph itself spans multiple
        # disconnected tensor neighborhoods (use centroid_edges as
        # a finer-grained view within the subgraph).
        inner_clusters = self._get_tag_clusters(subgraph_tags)
        is_cross_domain = len(inner_clusters) >= self._config.meta_cross_domain_min_clusters

        if is_cross_domain:
            pattern_type = "cross_domain_theme"
            confidence = min(1.0, ratio / (threshold * 2))
        else:
            pattern_type = "recurring_sync"
            confidence = min(1.0, ratio / (threshold * 3))

        sync_ids = [e.id for e in cluster if e.id is not None]
        tag_cluster_list = [list(tc) for tc in inner_clusters]

        return MetaPattern(
            description=self._describe_pattern(cluster, inner_clusters, ratio),
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
              AND edge_strength > 0.05
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
