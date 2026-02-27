"""Feedback engine: meta-patterns feed back into the base layer."""

from __future__ import annotations

import json

from chicory.db.engine import DatabaseEngine
from chicory.layer1.salience import SalienceScorer
from chicory.layer1.tag_manager import TagManager
from chicory.models.meta_pattern import MetaPattern


class FeedbackEngine:
    """Executes feedback actions from meta-patterns back into the base layer."""

    def __init__(
        self,
        db: DatabaseEngine,
        tag_manager: TagManager,
        salience: SalienceScorer,
    ) -> None:
        self._db = db
        self._tags = tag_manager
        self._salience = salience

    def apply_pattern_actions(self, pattern: MetaPattern) -> dict:
        """Apply feedback actions for a promoted meta-pattern.
        Returns a dict describing what was done."""
        actions = {}

        # If cross-domain, create an emergent tag linking the clusters
        if pattern.pattern_type == "cross_domain_theme":
            tag_clusters = json.loads(pattern.involved_tag_clusters)
            if len(tag_clusters) >= 2:
                tag = self._create_emergent_tag(pattern, tag_clusters)
                if tag:
                    actions["created_tag"] = tag.name

        # Boost salience of involved memories
        involved_sync_ids = json.loads(pattern.involved_sync_ids)
        boosted = self._boost_involved_memories(involved_sync_ids)
        if boosted:
            actions["boosted_memories"] = boosted

        # Record actions on the pattern
        self._db.execute(
            "UPDATE meta_patterns SET actions_taken = ? WHERE id = ?",
            (json.dumps(actions), pattern.id),
        )
        self._db.connection.commit()

        return actions

    def _create_emergent_tag(self, pattern: MetaPattern, tag_clusters: list[list[int]]):
        """Create a new tag that represents the emergent cross-domain theme."""
        # Get representative tag names from each cluster
        cluster_names = []
        for cluster in tag_clusters[:3]:
            if cluster:
                row = self._db.execute(
                    "SELECT name FROM tags WHERE id = ?", (cluster[0],)
                ).fetchone()
                if row:
                    cluster_names.append(row["name"])

        if len(cluster_names) < 2:
            return None

        name = f"{cluster_names[0]}-x-{cluster_names[1]}"
        description = (
            f"Emergent theme connecting {', '.join(cluster_names)}. "
            f"Auto-created from meta-pattern: {pattern.description[:100]}"
        )

        # Check for duplicates
        existing = self._tags.get_by_name(name)
        if existing:
            return existing

        return self._tags.get_or_create(
            name=name,
            created_by="meta_pattern",
            description=description,
        )

    def _boost_involved_memories(self, sync_ids: list[int]) -> int:
        """Boost salience of memories involved in the synchronicity events."""
        memory_ids: set[str] = set()
        for sid in sync_ids:
            row = self._db.execute(
                "SELECT involved_memories FROM synchronicity_events WHERE id = ?",
                (sid,),
            ).fetchone()
            if row and row["involved_memories"]:
                memory_ids.update(json.loads(row["involved_memories"]))

        for mid in memory_ids:
            self._salience.adjust_salience(mid, 0.05)  # Small boost

        return len(memory_ids)
