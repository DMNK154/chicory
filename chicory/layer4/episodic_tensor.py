"""Episodic relational tensor: sparse, forest-scoped, lazy-cached memory edges.

The tag tensor is global because tags are few and abstract.
The episodic tensor is local and episodic because memories are many and concrete.

Tag tensor:  "How do concepts relate globally?"
Episodic tensor: "Why did these two specific memories become meaningfully adjacent?"

Gateway creation rules — a pair is materialized only when:
  1. semantic_similarity(A, B) >= similarity_threshold
  2. co_retrieved(A, B)
  3. same_forest_block(A, B)
  4. tag_projected_affinity(A, B) >= episodic_tag_affinity_threshold

Cheap fields are computed eagerly at creation.
Expensive fields (supersession, contradiction, narrative) are filled lazily.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from chicory.config import ChicoryConfig
    from chicory.db.engine import DatabaseEngine

log = logging.getLogger(__name__)


class EpisodicTensor:

    def __init__(self, db: DatabaseEngine, config: ChicoryConfig) -> None:
        self._db = db
        self._cfg = config

    # ------------------------------------------------------------------
    # Bulk bootstrap: build the tensor from existing data
    # ------------------------------------------------------------------

    def bootstrap(self, batch_size: int = 500) -> int:
        """Build the episodic tensor from existing memories.

        Gateways:
          1. FAISS top-K semantic neighbors
          2. Co-retrieval pairs
          3. Same forest block membership
          4. High tag-projected affinity

        Returns total edges created.
        """
        log.info("Episodic tensor: loading embeddings...")
        mem_ids, vectors = self._load_all_embeddings()
        if len(mem_ids) == 0:
            log.warning("No embeddings found — skipping episodic tensor bootstrap")
            return 0

        log.info("Episodic tensor: %d memories with embeddings", len(mem_ids))

        candidates: set[tuple[str, str]] = set()

        # Gateway 1: FAISS semantic neighbors above similarity_threshold
        log.info("Gateway 1: FAISS semantic (threshold=%.2f)...", self._cfg.similarity_threshold)
        sem_pairs = self._gateway_semantic(mem_ids, vectors)
        candidates.update(sem_pairs)
        log.info("  %d pairs from semantic threshold", len(sem_pairs))

        # Gateway 2: Co-retrieval pairs
        log.info("Gateway 2: co-retrieval pairs...")
        coret_pairs = self._gateway_co_retrieval()
        candidates.update(coret_pairs)
        log.info("  %d pairs from co-retrieval", len(coret_pairs))

        # Gateway 3: Same forest block
        log.info("Gateway 3: same forest block...")
        block_pairs = self._gateway_same_block()
        candidates.update(block_pairs)
        log.info("  %d pairs from forest blocks", len(block_pairs))

        # Tag tensor data contributes to edge scores via tag_projected_strength
        # in _compute_eager_fields — no separate candidate gateway needed.

        log.info("Total unique candidate pairs: %d", len(candidates))

        # Build lookup structures for eager field computation
        id_to_idx = {mid: i for i, mid in enumerate(mem_ids)}
        mem_tags = self._load_all_memory_tags()
        tag_tensor = self._load_tag_tensor_cache(mem_tags)
        mem_times = self._load_memory_timestamps()
        mem_sources = self._load_memory_sources()
        coret_counts = self._load_co_retrieval_counts()

        # Compute eager fields and insert in batches
        rows: list[tuple] = []
        for a, b in candidates:
            edge = self._compute_eager_fields(
                a, b, id_to_idx, vectors, mem_tags, tag_tensor,
                mem_times, mem_sources, coret_counts,
            )
            if edge is not None:
                rows.append(edge)

            if len(rows) >= batch_size:
                self._batch_insert(rows)
                rows.clear()

        if rows:
            self._batch_insert(rows)

        prune_stats = self.prune()

        total = self._db.execute(
            "SELECT COUNT(*) as c FROM memory_relational_tensor WHERE edge_status != 'archived'"
        ).fetchone()["c"]
        log.info(
            "Episodic tensor bootstrap complete: %d active edges (pruned %d weak)",
            total, prune_stats["weak_pruned"],
        )
        return total

    # ------------------------------------------------------------------
    # Gateway methods
    # ------------------------------------------------------------------

    def _gateway_semantic(
        self, mem_ids: list[str], vectors: np.ndarray,
    ) -> set[tuple[str, str]]:
        """Find semantic neighbors above similarity_threshold via FAISS."""
        import faiss

        threshold = self._cfg.similarity_threshold
        n, dim = vectors.shape

        if n < 78:
            index = faiss.IndexFlatIP(dim)
        else:
            nlist = max(1, int(math.sqrt(n)))
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(vectors)
            index.nprobe = min(8, nlist)

        index.add(vectors)

        lims, D, I = index.range_search(vectors, threshold)

        pairs: set[tuple[str, str]] = set()
        for i in range(n):
            for pos in range(lims[i], lims[i + 1]):
                j = int(I[pos])
                if j == i:
                    continue
                a, b = mem_ids[i], mem_ids[j]
                pairs.add((min(a, b), max(a, b)))

        return pairs

    def _gateway_co_retrieval(self) -> set[tuple[str, str]]:
        rows = self._db.execute(
            """SELECT r1.memory_id as a, r2.memory_id as b
               FROM retrieval_results r1
               JOIN retrieval_results r2
                 ON r1.retrieval_id = r2.retrieval_id
                AND r1.memory_id < r2.memory_id
               GROUP BY r1.memory_id, r2.memory_id
               HAVING COUNT(*) >= 1"""
        ).fetchall()
        return {(r["a"], r["b"]) for r in rows}

    def _gateway_same_block(self) -> set[tuple[str, str]]:
        rows = self._db.execute(
            """SELECT bm1.target_id as a, bm2.target_id as b
               FROM block_memberships bm1
               JOIN block_memberships bm2
                 ON bm1.block_id = bm2.block_id
                AND bm1.target_type = 'memory'
                AND bm2.target_type = 'memory'
                AND bm1.target_id < bm2.target_id
               GROUP BY bm1.target_id, bm2.target_id"""
        ).fetchall()
        return {(r["a"], r["b"]) for r in rows}


    # ------------------------------------------------------------------
    # Eager field computation
    # ------------------------------------------------------------------

    def _compute_eager_fields(
        self,
        a: str,
        b: str,
        id_to_idx: dict[str, int],
        vectors: np.ndarray,
        mem_tags: dict[str, list[int]],
        tag_tensor: dict[tuple[int, int], dict],
        mem_times: dict[str, str],
        mem_sources: dict[str, str],
        coret_counts: dict[tuple[str, str], int],
    ) -> tuple | None:
        # Semantic strength (cosine similarity from embeddings)
        idx_a = id_to_idx.get(a)
        idx_b = id_to_idx.get(b)
        if idx_a is not None and idx_b is not None:
            semantic = float(np.dot(vectors[idx_a], vectors[idx_b]))
        else:
            semantic = 0.0

        # Tag-projected strengths (multiplex projection from tag tensor)
        tags_a = set(mem_tags.get(a, []))
        tags_b = set(mem_tags.get(b, []))

        tag_sem, tag_sync, tag_cooc, tag_inhib, tag_glyph = 0.0, 0.0, 0.0, 0.0, 0.0
        pair_count = 0

        for ta in tags_a:
            for tb in tags_b:
                key = (min(ta, tb), max(ta, tb))
                t = tag_tensor.get(key)
                if t:
                    tag_sem += t["semantic_strength"]
                    tag_sync += t["synchronicity_strength"]
                    tag_cooc += t["cooccurrence_strength"]
                    tag_inhib += t["inhibition_strength"]
                    tag_glyph += t["glyph_strength"]
                    pair_count += 1

        if pair_count > 0:
            tag_sem /= pair_count
            tag_sync /= pair_count
            tag_cooc /= pair_count
            tag_inhib /= pair_count
            tag_glyph /= pair_count

        tag_projected = tag_sem + tag_sync + tag_cooc + tag_glyph - tag_inhib

        # Temporal proximity (exponential decay)
        time_a = mem_times.get(a, "")
        time_b = mem_times.get(b, "")
        temporal = self._temporal_proximity(time_a, time_b)

        # Source proximity (same source_model = 1.0)
        src_a = mem_sources.get(a, "")
        src_b = mem_sources.get(b, "")
        source_prox = 1.0 if src_a and src_b and src_a == src_b else 0.0

        # Co-retrieval strength
        coret = coret_counts.get((a, b), 0)
        co_retrieval = min(1.0, coret / 5.0) if coret > 0 else 0.0

        # Bridge strength: connected but spanning different tag domains
        if tags_a and tags_b:
            jaccard = len(tags_a & tags_b) / len(tags_a | tags_b)
        else:
            jaccard = 0.0
        tag_distance = 1.0 - jaccard
        connection = max(semantic, co_retrieval)
        bridge = connection * tag_distance

        composite = semantic + tag_projected + co_retrieval + bridge
        if composite < self._cfg.episodic_min_edge_strength:
            return None

        return (
            a, b,
            semantic, tag_projected, co_retrieval, temporal, source_prox,
            tag_sem, tag_sync, tag_cooc, tag_inhib, tag_glyph,
            # lazy fields
            0.0, 0.0, 0.0, 0, 0.0, bridge,
            0, "candidate",
        )

    def _temporal_proximity(self, time_a: str, time_b: str) -> float:
        if not time_a or not time_b:
            return 0.0
        from datetime import datetime
        try:
            ta = datetime.fromisoformat(time_a.replace("Z", "+00:00"))
            tb = datetime.fromisoformat(time_b.replace("Z", "+00:00"))
            hours_apart = abs((ta - tb).total_seconds()) / 3600.0
            halflife = self._cfg.episodic_temporal_halflife_hours
            return math.exp(-0.693 * hours_apart / halflife)
        except (ValueError, TypeError):
            return 0.0

    # ------------------------------------------------------------------
    # Batch insert
    # ------------------------------------------------------------------

    def _batch_insert(self, rows: list[tuple]) -> None:
        self._db.connection.executemany(
            """INSERT INTO memory_relational_tensor
               (memory_a_id, memory_b_id,
                semantic_strength, tag_projected_strength, co_retrieval_strength,
                temporal_proximity, source_proximity,
                tag_semantic_projected, tag_sync_projected,
                tag_cooccurrence_projected, tag_inhibition_projected,
                tag_glyph_projected,
                retrieval_reinforcement, narrative_continuity,
                supersession_strength, supersession_direction,
                contradiction_strength, bridge_strength,
                activation_count, edge_status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(memory_a_id, memory_b_id) DO UPDATE SET
                   semantic_strength = MAX(semantic_strength, excluded.semantic_strength),
                   tag_projected_strength = MAX(tag_projected_strength, excluded.tag_projected_strength),
                   co_retrieval_strength = MAX(co_retrieval_strength, excluded.co_retrieval_strength),
                   bridge_strength = MAX(bridge_strength, excluded.bridge_strength),
                   activation_count = activation_count + 1""",
            rows,
        )
        self._db.connection.commit()

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_all_embeddings(self) -> tuple[list[str], np.ndarray]:
        rows = self._db.execute(
            """SELECT memory_id, embedding, dimension FROM embeddings
               WHERE chunk_index = 0
               ORDER BY memory_id"""
        ).fetchall()

        if not rows:
            return [], np.array([])

        dim = rows[0]["dimension"]
        mem_ids = [r["memory_id"] for r in rows]
        all_bytes = b"".join(r["embedding"] for r in rows)
        vectors = np.frombuffer(all_bytes, dtype=np.float32).reshape(-1, dim).copy()

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors /= norms

        return mem_ids, vectors

    def _load_all_memory_tags(self) -> dict[str, list[int]]:
        rows = self._db.execute(
            "SELECT memory_id, tag_id FROM memory_tags ORDER BY memory_id"
        ).fetchall()
        result: dict[str, list[int]] = {}
        for r in rows:
            result.setdefault(r["memory_id"], []).append(r["tag_id"])
        return result

    def _load_tag_tensor_cache(
        self, mem_tags: dict[str, list[int]],
    ) -> dict[tuple[int, int], dict]:
        all_tags: set[int] = set()
        for tags in mem_tags.values():
            all_tags.update(tags)

        if not all_tags:
            return {}

        placeholders = ",".join("?" * len(all_tags))
        sorted_tags = sorted(all_tags)
        rows = self._db.execute(
            f"""SELECT tag_a_id, tag_b_id,
                       cooccurrence_strength, synchronicity_strength,
                       semantic_strength, inhibition_strength, glyph_strength
                FROM tag_relational_tensor
                WHERE tag_a_id IN ({placeholders}) AND tag_b_id IN ({placeholders})""",
            sorted_tags + sorted_tags,
        ).fetchall()

        cache: dict[tuple[int, int], dict] = {}
        for r in rows:
            cache[(r["tag_a_id"], r["tag_b_id"])] = dict(r)
        return cache

    def _load_memory_timestamps(self) -> dict[str, str]:
        rows = self._db.execute(
            "SELECT id, created_at FROM memories"
        ).fetchall()
        return {r["id"]: r["created_at"] for r in rows}

    def _load_memory_sources(self) -> dict[str, str]:
        rows = self._db.execute(
            "SELECT id, source_model FROM memories"
        ).fetchall()
        return {r["id"]: r["source_model"] for r in rows}

    def _load_co_retrieval_counts(self) -> dict[tuple[str, str], int]:
        rows = self._db.execute(
            """SELECT r1.memory_id as a, r2.memory_id as b, COUNT(*) as cnt
               FROM retrieval_results r1
               JOIN retrieval_results r2
                 ON r1.retrieval_id = r2.retrieval_id
                AND r1.memory_id < r2.memory_id
               GROUP BY r1.memory_id, r2.memory_id"""
        ).fetchall()
        return {(r["a"], r["b"]): r["cnt"] for r in rows}

    # ------------------------------------------------------------------
    # Incremental update (for orchestrator integration)
    # ------------------------------------------------------------------

    def update_on_store(self, memory_id: str) -> int:
        """Create candidate edges for a newly stored memory."""
        row = self._db.execute(
            """SELECT embedding, dimension FROM embeddings
               WHERE memory_id = ? AND chunk_index = 0""",
            (memory_id,),
        ).fetchone()
        if not row:
            return 0

        vec = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        mem_ids, vectors = self._load_all_embeddings()
        if len(mem_ids) == 0:
            return 0

        scores = vectors @ vec
        threshold = self._cfg.similarity_threshold
        above = np.where(scores >= threshold)[0]

        id_to_idx = {mid: i for i, mid in enumerate(mem_ids)}
        mem_tags = self._load_all_memory_tags()
        tag_tensor = self._load_tag_tensor_cache(mem_tags)
        mem_times = self._load_memory_timestamps()
        mem_sources = self._load_memory_sources()
        coret_counts = self._load_co_retrieval_counts()

        rows: list[tuple] = []
        for idx in above:
            other = mem_ids[int(idx)]
            if other == memory_id:
                continue
            a, b = min(memory_id, other), max(memory_id, other)
            edge = self._compute_eager_fields(
                a, b, id_to_idx, vectors, mem_tags, tag_tensor,
                mem_times, mem_sources, coret_counts,
            )
            if edge is not None:
                rows.append(edge)

        if rows:
            self._batch_insert(rows)
        return len(rows)

    def activate_edge(self, memory_a: str, memory_b: str) -> None:
        """Record an activation and promote lifecycle based on composite strength.

        Lifecycle is driven by the edge's composite signal (semantic +
        tag_projected + co_retrieval + bridge) rather than raw activation
        count.  ``episodic_tag_affinity_threshold`` gates candidate→warm;
        the mean of the composite and 1.0 gates warm→mature.
        """
        a, b = min(memory_a, memory_b), max(memory_a, memory_b)
        threshold = self._cfg.episodic_tag_affinity_threshold

        self._db.execute(
            """UPDATE memory_relational_tensor SET
                   activation_count = activation_count + 1,
                   last_activated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now'),
                   edge_status = CASE
                       WHEN (semantic_strength + tag_projected_strength +
                             co_retrieval_strength + bridge_strength) >=
                            (1.0 + ?) / 2.0
                       THEN 'mature'
                       WHEN (semantic_strength + tag_projected_strength +
                             co_retrieval_strength + bridge_strength) >= ?
                       THEN 'warm'
                       ELSE edge_status
                   END
               WHERE memory_a_id = ? AND memory_b_id = ?""",
            (threshold, threshold, a, b),
        )

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_neighbors(
        self, memory_id: str, top_k: int = 20, min_status: str = "candidate",
    ) -> list[dict]:
        """Get the strongest episodic neighbors for a memory."""
        status_order = {"candidate": 0, "warm": 1, "mature": 2, "decaying": 3, "archived": 4}
        min_rank = status_order.get(min_status, 0)
        valid = [s for s, r in status_order.items() if r >= min_rank and s != "archived"]
        placeholders = ",".join("?" * len(valid))

        rows = self._db.execute(
            f"""SELECT * FROM memory_relational_tensor
                WHERE (memory_a_id = ? OR memory_b_id = ?)
                  AND edge_status IN ({placeholders})
                ORDER BY tag_projected_strength + semantic_strength + bridge_strength DESC
                LIMIT ?""",
            [memory_id, memory_id] + valid + [top_k],
        ).fetchall()
        return [dict(r) for r in rows]

    def get_edge(self, memory_a: str, memory_b: str) -> dict | None:
        a, b = min(memory_a, memory_b), max(memory_a, memory_b)
        row = self._db.execute(
            "SELECT * FROM memory_relational_tensor WHERE memory_a_id=? AND memory_b_id=?",
            (a, b),
        ).fetchone()
        return dict(row) if row else None

    def get_cluster_edges(self, memory_ids: list[str]) -> list[dict]:
        """Get all edges within a set of memories."""
        if len(memory_ids) < 2:
            return []
        placeholders = ",".join("?" * len(memory_ids))
        rows = self._db.execute(
            f"""SELECT * FROM memory_relational_tensor
                WHERE memory_a_id IN ({placeholders})
                  AND memory_b_id IN ({placeholders})
                ORDER BY tag_projected_strength + semantic_strength DESC""",
            memory_ids + memory_ids,
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Backfill bridge_strength for existing edges
    # ------------------------------------------------------------------

    def backfill_bridge_strength(self, batch_size: int = 5000) -> int:
        """Compute bridge_strength for all edges where it is currently 0.

        bridge = max(semantic, co_retrieval) * (1 - tag_jaccard)
        """
        mem_tags = self._load_all_memory_tags()

        total_updated = 0
        offset = 0
        while True:
            edges = self._db.execute(
                """SELECT rowid, memory_a_id, memory_b_id,
                          semantic_strength, co_retrieval_strength
                   FROM memory_relational_tensor
                   WHERE bridge_strength = 0
                   LIMIT ? OFFSET ?""",
                (batch_size, offset),
            ).fetchall()

            if not edges:
                break

            updates: list[tuple] = []
            for e in edges:
                tags_a = set(mem_tags.get(e["memory_a_id"], []))
                tags_b = set(mem_tags.get(e["memory_b_id"], []))
                if tags_a and tags_b:
                    jaccard = len(tags_a & tags_b) / len(tags_a | tags_b)
                else:
                    jaccard = 0.0
                tag_distance = 1.0 - jaccard
                connection = max(e["semantic_strength"], e["co_retrieval_strength"])
                bridge = connection * tag_distance
                if bridge > 0:
                    updates.append((bridge, e["rowid"]))

            if updates:
                self._db.connection.executemany(
                    "UPDATE memory_relational_tensor SET bridge_strength = ? WHERE rowid = ?",
                    updates,
                )
                self._db.connection.commit()
                total_updated += len(updates)

            if len(edges) < batch_size:
                break
            offset += batch_size

        log.info("Backfilled bridge_strength for %d edges", total_updated)
        return total_updated

    # ------------------------------------------------------------------
    # Pruning & lifecycle enforcement
    # ------------------------------------------------------------------

    def prune(self) -> dict[str, int]:
        """Run all pruning passes: weak edges, inactivity decay, archival.

        Returns counts for each pass.
        """
        stats: dict[str, int] = {}
        stats["weak_pruned"] = self._prune_weak_edges()
        stats["cap_pruned"] = 0
        stats["decayed"] = self._decay_inactive()
        stats["archived"] = self._archive_long_decayed()
        return stats

    def _prune_weak_edges(self) -> int:
        """Remove edges below the minimum composite strength threshold."""
        threshold = self._cfg.episodic_min_edge_strength
        cursor = self._db.execute(
            """DELETE FROM memory_relational_tensor
               WHERE (semantic_strength + tag_projected_strength +
                      co_retrieval_strength + bridge_strength) < ?
                 AND edge_status != 'mature'""",
            (threshold,),
        )
        pruned = cursor.rowcount
        if pruned:
            self._db.connection.commit()
            log.info("Episodic prune: %d weak edges removed (threshold=%.3f)", pruned, threshold)
        return pruned

    def _decay_inactive(self) -> int:
        """Transition edges to 'decaying' if not activated within the decay window."""
        hours = self._cfg.episodic_decay_inactive_hours
        cursor = self._db.execute(
            """UPDATE memory_relational_tensor SET edge_status = 'decaying'
               WHERE edge_status IN ('candidate', 'warm')
                 AND (last_activated_at IS NULL
                      OR last_activated_at < strftime('%%Y-%%m-%%dT%%H:%%M:%%f',
                          'now', ? || ' hours'))""",
            (f"-{hours}",),
        )
        decayed = cursor.rowcount
        if decayed:
            self._db.connection.commit()
            log.info("Episodic decay: %d edges → decaying (inactive > %.0f hours)", decayed, hours)
        return decayed

    def _archive_long_decayed(self) -> int:
        """Archive edges that have been decaying for twice the decay window."""
        hours = self._cfg.episodic_decay_inactive_hours * 2
        cursor = self._db.execute(
            """UPDATE memory_relational_tensor SET edge_status = 'archived'
               WHERE edge_status = 'decaying'
                 AND (last_activated_at IS NULL
                      OR last_activated_at < strftime('%%Y-%%m-%%dT%%H:%%M:%%f',
                          'now', ? || ' hours'))""",
            (f"-{hours}",),
        )
        archived = cursor.rowcount
        if archived:
            self._db.connection.commit()
            log.info("Episodic archive: %d edges → archived (inactive > %.0f hours)", archived, hours)
        return archived
