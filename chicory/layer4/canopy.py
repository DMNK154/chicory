"""Canopy observer: always-on graph growth layer.

Discovers emergent memory clusters from episodic tensor co-retrieval
and bridge edges.  Scores clusters by co-activation strength minus
tag-overlap inhibition (the tag tensor already captures tag-level
associations — the canopy surfaces structure tags don't predict).

The canopy never decays, prunes, or deletes growth.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Optional

from chicory.models.canopy import CanopyShape, ScoreBundle

if TYPE_CHECKING:
    from chicory.config import ChicoryConfig
    from chicory.db.engine import DatabaseEngine
    from chicory.layer4.forest import ForestReorganizer


class CanopyObserver:

    def __init__(
        self,
        db: DatabaseEngine,
        config: ChicoryConfig,
        forest: ForestReorganizer,
    ) -> None:
        self._db = db
        self._cfg = config
        self._forest = forest

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(
        self,
        source: str,
        source_id: Optional[str],
        memory_ids: list[str],
        tag_ids: list[int] | None = None,
        touched_block_keys: list[str] | None = None,
        sync_event_ids: list[int] | None = None,
        meta_pattern_ids: list[int] | None = None,
        resonance_ids: list[int] | None = None,
    ) -> list[str]:
        """Run a canopy observation from a set of co-active memories.

        Queries the episodic tensor for edges among *memory_ids*,
        finds connected components of co-retrieval/bridge edges,
        scores each cluster, and records growth.

        Returns block_keys of canopy blocks that grew.
        """
        if not self._cfg.canopy_enabled:
            return []

        memory_ids = list(dict.fromkeys(memory_ids))
        if len(memory_ids) < 2:
            return []

        shapes = self._discover_clusters(memory_ids, source, source_id)

        grown_keys: list[str] = []
        grown_shapes: list[CanopyShape] = []

        for shape in shapes:
            scores = self._score_cluster(shape)
            self._record_observation(shape, scores, source, source_id)
            block_id = self._upsert_canopy_block(shape, scores)
            self._upsert_support_edges(block_id, shape, scores)

            if scores.growth_potential > 0:
                grown_keys.append(shape.block_key)
                grown_shapes.append(shape)

        # Recursive merge: grown clusters → higher-order clusters
        max_depth = self._cfg.canopy_max_depth_per_pass
        blocks_for_recursion = grown_shapes
        for depth in range(1, max_depth + 1):
            if len(blocks_for_recursion) < 2:
                break
            blocks_for_recursion = self._recursive_merge(
                depth, blocks_for_recursion, source, source_id,
            )
            grown_keys.extend(s.block_key for s in blocks_for_recursion)

        return grown_keys

    def global_recursive_pass(self, from_depth: int = 0) -> list[str]:
        """Merge all grown blocks at *from_depth* into depth+1 clusters.

        Loads every grown canopy block at the given depth, reconstructs
        CanopyShape objects, and feeds them through _recursive_merge.
        Repeats upward until no new merges occur.

        Returns block_keys of all newly created higher-layer blocks.
        """
        all_new_keys: list[str] = []
        depth = from_depth

        while True:
            rows = self._db.execute(
                """SELECT block_key, block_type, layer_depth, tag_ids,
                          memory_ids, parent_block_keys, source_event_types
                   FROM canopy_blocks
                   WHERE layer_depth = ? AND first_growth_at IS NOT NULL""",
                (depth,),
            ).fetchall()

            if len(rows) < 2:
                break

            shapes = []
            for r in rows:
                shapes.append(CanopyShape(
                    block_key=r["block_key"],
                    block_type=r["block_type"],
                    layer_depth=r["layer_depth"],
                    tag_ids=json.loads(r["tag_ids"]),
                    memory_ids=json.loads(r["memory_ids"]),
                    parent_block_keys=json.loads(r["parent_block_keys"]),
                    source="global_recursive",
                    source_id=None,
                    source_event_types=json.loads(r["source_event_types"]),
                ))

            merged = self._recursive_merge(
                depth + 1, shapes, "global_recursive", None,
            )

            if not merged:
                break

            all_new_keys.extend(s.block_key for s in merged)
            depth += 1

        return all_new_keys

    def observe_single_memory(
        self,
        source: str,
        source_id: str,
        memory_id: str,
    ) -> list[str]:
        """Observe a single memory's episodic neighborhood.

        Pulls the memory's top neighbors from the episodic tensor
        and runs a cluster observation on that neighborhood.
        """
        if not self._cfg.canopy_enabled:
            return []

        neighbors = self._db.execute(
            """SELECT memory_a_id, memory_b_id, co_retrieval_strength, bridge_strength
               FROM memory_relational_tensor
               WHERE (memory_a_id = ? OR memory_b_id = ?)
                 AND (co_retrieval_strength > 0 OR bridge_strength > 0)""",
            (memory_id, memory_id),
        ).fetchall()

        neighbor_ids = {memory_id}
        for r in neighbors:
            neighbor_ids.add(r["memory_a_id"])
            neighbor_ids.add(r["memory_b_id"])

        return self.observe(
            source=source,
            source_id=source_id,
            memory_ids=list(neighbor_ids),
        )

    # ------------------------------------------------------------------
    # Cluster discovery from episodic tensor
    # ------------------------------------------------------------------

    def _discover_clusters(
        self,
        memory_ids: list[str],
        source: str,
        source_id: Optional[str],
    ) -> list[CanopyShape]:
        """Find connected components of co-retrieval/bridge edges."""
        if len(memory_ids) < 2:
            return []

        placeholders = ",".join("?" * len(memory_ids))
        edges = self._db.execute(
            f"""SELECT memory_a_id, memory_b_id,
                       co_retrieval_strength, bridge_strength
                FROM memory_relational_tensor
                WHERE memory_a_id IN ({placeholders})
                  AND memory_b_id IN ({placeholders})
                  AND (co_retrieval_strength > 0 OR bridge_strength > 0)""",
            memory_ids + memory_ids,
        ).fetchall()

        if not edges:
            return []

        edge_pairs: list[tuple[str, str]] = []
        for e in edges:
            edge_pairs.append((e["memory_a_id"], e["memory_b_id"]))

        components = _connected_components(memory_ids, edge_pairs)

        shapes: list[CanopyShape] = []
        for component in components:
            if len(component) < 2:
                continue

            sorted_mids = sorted(component)
            tag_ids = self._get_tag_ids_for_memories(sorted_mids)

            shapes.append(CanopyShape(
                block_key=_canopy_key(0, "memory_cluster", sorted_mids),
                block_type="memory_cluster",
                layer_depth=0,
                tag_ids=tag_ids,
                memory_ids=sorted_mids,
                parent_block_keys=[],
                source=source,
                source_id=source_id,
                source_event_types=[source],
            ))

        return shapes

    # ------------------------------------------------------------------
    # Scoring: pressure = co-activation, inhibition = tag overlap
    # ------------------------------------------------------------------

    def _score_cluster(self, shape: CanopyShape) -> ScoreBundle:
        mids = shape.memory_ids
        if len(mids) < 2:
            return ScoreBundle()

        placeholders = ",".join("?" * len(mids))
        edges = self._db.execute(
            f"""SELECT co_retrieval_strength, bridge_strength,
                       semantic_strength, tag_projected_strength
                FROM memory_relational_tensor
                WHERE memory_a_id IN ({placeholders})
                  AND memory_b_id IN ({placeholders})""",
            mids + mids,
        ).fetchall()

        if not edges:
            return ScoreBundle()

        co_ret = [e["co_retrieval_strength"] for e in edges if e["co_retrieval_strength"] > 0]
        bridge = [e["bridge_strength"] for e in edges if e["bridge_strength"] > 0]
        semantic = [e["semantic_strength"] for e in edges]

        mean_co_ret = sum(co_ret) / len(co_ret) if co_ret else 0.0
        mean_bridge = sum(bridge) / len(bridge) if bridge else 0.0
        mean_semantic = sum(semantic) / len(semantic) if semantic else 0.0

        # Pressure: memories that fire together (normalize by active channels)
        coact_w = self._cfg.canopy_pressure_coactivation_weight if co_ret else 0.0
        bridge_w = self._cfg.canopy_pressure_bridge_weight if bridge else 0.0
        total_w = coact_w + bridge_w
        if total_w > 0:
            pressure = (coact_w * mean_co_ret + bridge_w * mean_bridge) / total_w
        else:
            pressure = 0.0

        # Inhibition: tag Jaccard overlap (suppress what tags already explain)
        tag_jaccard = self._cluster_tag_jaccard(mids)
        inhibition = tag_jaccard * self._cfg.canopy_inhibition_tag_overlap_weight

        growth_potential = pressure - inhibition

        if self._cfg.canopy_use_soft_growth:
            tau = self._cfg.canopy_growth_temperature
            canopy_growth = 1.0 / (1.0 + math.exp(-growth_potential / max(tau, 0.01)))
        else:
            canopy_growth = max(0.0, growth_potential)

        return ScoreBundle(
            bridge=_clamp01(mean_bridge),
            cooccurrence=_clamp01(mean_co_ret),
            similarity=_clamp01(mean_semantic),
            pressure=_clamp01(pressure),
            threshold=_clamp01(inhibition),
            growth_potential=growth_potential,
            canopy_growth=canopy_growth,
        )


    def _cluster_tag_jaccard(self, memory_ids: list[str]) -> float:
        """Mean pairwise tag Jaccard similarity across memories in the cluster."""
        tag_sets: list[set[int]] = []
        for mid in memory_ids:
            rows = self._db.execute(
                "SELECT tag_id FROM memory_tags WHERE memory_id = ?", (mid,),
            ).fetchall()
            tag_sets.append({r["tag_id"] for r in rows})

        if len(tag_sets) < 2:
            return 0.0

        total = 0.0
        count = 0
        for i in range(len(tag_sets)):
            for j in range(i + 1, len(tag_sets)):
                a, b = tag_sets[i], tag_sets[j]
                union = len(a | b)
                if union > 0:
                    total += len(a & b) / union
                count += 1

        return total / count if count > 0 else 0.0

    # ------------------------------------------------------------------
    # Recursive merge: grown clusters → higher-order clusters
    # ------------------------------------------------------------------

    def _recursive_merge(
        self,
        depth: int,
        grown_shapes: list[CanopyShape],
        source: str,
        source_id: Optional[str],
    ) -> list[CanopyShape]:
        """Merge grown clusters via episodic tensor edges with depth-scaled threshold."""
        if len(grown_shapes) < 2:
            return []

        # Gather all unique memories across grown blocks
        all_mids_set: set[str] = set()
        for shape in grown_shapes:
            all_mids_set.update(shape.memory_ids)
        all_mids = sorted(all_mids_set)

        # Discover connected components via episodic tensor (same as depth 0)
        discovered = self._discover_clusters(all_mids, source, source_id)

        # Growth threshold increases with depth
        growth_threshold = depth * self._cfg.canopy_inhibition_tag_overlap_weight

        result: list[CanopyShape] = []
        for proto_shape in discovered:
            # Re-key at the correct depth with parent tracking
            parent_keys = []
            for shape in grown_shapes:
                if set(shape.memory_ids) & set(proto_shape.memory_ids):
                    parent_keys.append(shape.block_key)

            shape = CanopyShape(
                block_key=_canopy_key(depth, "memory_cluster", proto_shape.memory_ids),
                block_type="memory_cluster",
                layer_depth=depth,
                tag_ids=proto_shape.tag_ids,
                memory_ids=proto_shape.memory_ids,
                parent_block_keys=sorted(parent_keys),
                source="recursive_canopy",
                source_id=source_id,
                source_event_types=["recursive_canopy"],
            )

            scores = self._score_cluster(shape)
            self._record_observation(shape, scores, "recursive_canopy", source_id)
            self._upsert_canopy_block(shape, scores)

            if scores.growth_potential > growth_threshold:
                result.append(shape)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_tag_ids_for_memories(self, memory_ids: list[str]) -> list[int]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" * len(memory_ids))
        rows = self._db.execute(
            f"SELECT DISTINCT tag_id FROM memory_tags WHERE memory_id IN ({placeholders})",
            memory_ids,
        ).fetchall()
        return sorted(r["tag_id"] for r in rows)

    # ------------------------------------------------------------------
    # DB persistence (kept from original)
    # ------------------------------------------------------------------

    def _record_observation(
        self, shape: CanopyShape, scores: ScoreBundle,
        source: str, source_id: Optional[str],
    ) -> None:
        self._db.execute(
            """INSERT INTO canopy_observations
               (block_key, source, source_id, layer_depth,
                tag_ids, memory_ids, source_canopy_block_ids,
                bridge, heat, recurrence, cooccurrence, similarity,
                relevance, semantics, pressure, threshold,
                growth_potential, canopy_growth)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                shape.block_key, source, source_id, shape.layer_depth,
                json.dumps(shape.tag_ids),
                json.dumps(shape.memory_ids),
                json.dumps(shape.parent_block_keys),
                scores.bridge, scores.heat, scores.recurrence,
                scores.cooccurrence, scores.similarity,
                scores.relevance, scores.semantics,
                scores.pressure, scores.threshold,
                scores.growth_potential, scores.canopy_growth,
            ),
        )

    def _upsert_canopy_block(self, shape: CanopyShape, scores: ScoreBundle) -> int:
        self._db.execute(
            """INSERT INTO canopy_blocks
               (block_key, block_type, layer_depth, tag_ids, memory_ids,
                parent_block_keys, source_event_types,
                peak_bridge, peak_heat, peak_recurrence,
                peak_cooccurrence, peak_similarity,
                peak_relevance, peak_semantics,
                peak_pressure, peak_threshold,
                peak_growth_potential, peak_canopy_growth,
                evidence_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
               ON CONFLICT(block_key) DO UPDATE SET
                   peak_bridge = MAX(peak_bridge, ?),
                   peak_heat = MAX(peak_heat, ?),
                   peak_recurrence = MAX(peak_recurrence, ?),
                   peak_cooccurrence = MAX(peak_cooccurrence, ?),
                   peak_similarity = MAX(peak_similarity, ?),
                   peak_relevance = MAX(peak_relevance, ?),
                   peak_semantics = MAX(peak_semantics, ?),
                   peak_pressure = MAX(peak_pressure, ?),
                   peak_threshold = MAX(peak_threshold, ?),
                   peak_growth_potential = MAX(peak_growth_potential, ?),
                   peak_canopy_growth = MAX(peak_canopy_growth, ?),
                   evidence_count = evidence_count + 1,
                   last_observed_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')""",
            (
                shape.block_key, shape.block_type, shape.layer_depth,
                json.dumps(shape.tag_ids), json.dumps(shape.memory_ids),
                json.dumps(shape.parent_block_keys),
                json.dumps(shape.source_event_types),
                scores.bridge, scores.heat, scores.recurrence,
                scores.cooccurrence, scores.similarity,
                scores.relevance, scores.semantics,
                scores.pressure, scores.threshold,
                scores.growth_potential, scores.canopy_growth,
                # UPDATE values
                scores.bridge, scores.heat, scores.recurrence,
                scores.cooccurrence, scores.similarity,
                scores.relevance, scores.semantics,
                scores.pressure, scores.threshold,
                scores.growth_potential, scores.canopy_growth,
            ),
        )

        if scores.growth_potential > 0:
            self._db.execute(
                """UPDATE canopy_blocks SET first_growth_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                   WHERE block_key = ? AND first_growth_at IS NULL""",
                (shape.block_key,),
            )

        row = self._db.execute(
            "SELECT id FROM canopy_blocks WHERE block_key=?",
            (shape.block_key,),
        ).fetchone()
        return row["id"]

    def _upsert_support_edges(
        self, block_id: int, shape: CanopyShape, scores: ScoreBundle,
    ) -> None:
        for mid in shape.memory_ids:
            self._db.execute(
                """INSERT INTO canopy_support_edges
                   (canopy_block_id, target_type, target_id, edge_type, strength, evidence_count)
                   VALUES (?, 'memory', ?, 'contains', ?, 1)
                   ON CONFLICT(canopy_block_id, target_type, target_id, edge_type) DO UPDATE SET
                       strength = MAX(strength, ?),
                       evidence_count = evidence_count + 1,
                       last_observed_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')""",
                (block_id, mid, scores.pressure, scores.pressure),
            )

        for pkey in shape.parent_block_keys:
            self._db.execute(
                """INSERT INTO canopy_support_edges
                   (canopy_block_id, target_type, target_id, edge_type, strength, evidence_count)
                   VALUES (?, 'canopy_block', ?, 'derived_from', ?, 1)
                   ON CONFLICT(canopy_block_id, target_type, target_id, edge_type) DO UPDATE SET
                       strength = MAX(strength, ?),
                       evidence_count = evidence_count + 1,
                       last_observed_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')""",
                (block_id, pkey, scores.pressure, scores.pressure),
            )

    # ------------------------------------------------------------------
    # Inspection API
    # ------------------------------------------------------------------

    def get_canopy_summary(self, top_k: int = 20) -> dict:
        total = self._db.execute("SELECT COUNT(*) as c FROM canopy_blocks").fetchone()["c"]
        grown = self._db.execute(
            "SELECT COUNT(*) as c FROM canopy_blocks WHERE first_growth_at IS NOT NULL"
        ).fetchone()["c"]
        obs_count = self._db.execute("SELECT COUNT(*) as c FROM canopy_observations").fetchone()["c"]

        top_blocks = self._db.execute(
            """SELECT block_key, block_type, layer_depth,
                      peak_pressure, peak_threshold, peak_canopy_growth,
                      evidence_count, first_growth_at
               FROM canopy_blocks
               ORDER BY peak_canopy_growth DESC
               LIMIT ?""",
            (top_k,),
        ).fetchall()

        return {
            "total_blocks": total,
            "grown_blocks": grown,
            "total_observations": obs_count,
            "top_blocks": [dict(b) for b in top_blocks],
        }

    def get_block_detail(self, block_key: str) -> dict | None:
        block = self._db.execute(
            "SELECT * FROM canopy_blocks WHERE block_key=?", (block_key,)
        ).fetchone()
        if not block:
            return None

        support = self._db.execute(
            "SELECT * FROM canopy_support_edges WHERE canopy_block_id=?",
            (block["id"],),
        ).fetchall()

        recent_obs = self._db.execute(
            """SELECT * FROM canopy_observations
               WHERE block_key=? ORDER BY observed_at DESC LIMIT 10""",
            (block_key,),
        ).fetchall()

        return {
            "block": dict(block),
            "support_edges": [dict(s) for s in support],
            "recent_observations": [dict(o) for o in recent_obs],
        }


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _canopy_key(layer_depth: int, block_type: str, sorted_ids: list[str]) -> str:
    raw = f"{layer_depth}:{block_type}:{','.join(sorted_ids)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _connected_components(keys, edges):
    """Union-find connected components. Keys can be any hashable type."""
    parent = {k: k for k in keys}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups: dict = {}
    for k in keys:
        root = find(k)
        groups.setdefault(root, []).append(k)
    return sorted(groups.values(), key=len, reverse=True)
