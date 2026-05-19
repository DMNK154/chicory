"""Episodic relational tensor: episode-scoped, tag-variance-driven memory edges.

The tag tensor is global because tags are few and abstract.
The episodic tensor is local and episodic because memories are many and concrete.

Tag tensor:  "How do concepts relate globally?"
Episodic tensor: "Why did these two specific memories become meaningfully adjacent?"

Candidate discovery is scoped by temporal episodes: within each episode,
pairwise semantic similarity (matrix multiply) identifies the top-K
neighbors per memory.  Tag tensor projection scores the edges across
all channels (semantic, synchronicity, co-occurrence, glyph, inhibition).
Co-retrieval strength accumulates incrementally via activate_edge.
"""

from __future__ import annotations

import logging
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

    _NEIGHBORS_K = 48

    def bootstrap(self, batch_size: int = 500) -> int:
        """Build the episodic tensor from existing memories.

        Scopes candidate discovery by temporal episode: within each
        episode, computes pairwise semantic similarity via matrix
        multiply and keeps the top-K neighbors per memory.  Tag tensor
        projection scores the edges.  Co-retrieval strength accumulates
        incrementally via activate_edge on retrieval.

        Returns total edges created.
        """
        episodes = self._load_episode_members()

        if episodes:
            assigned_ids = set()
            for mids in episodes.values():
                assigned_ids.update(mids)
            log.info("Episodic tensor: loading embeddings for %d assigned memories...", len(assigned_ids))
            mem_ids, vectors = self._load_embeddings_for(assigned_ids)
        else:
            log.info("No episode assignments — loading all embeddings for global fallback")
            mem_ids, vectors = self._load_all_embeddings()
            episodes = {0: mem_ids}

        if len(mem_ids) == 0:
            log.warning("No embeddings found — skipping episodic tensor bootstrap")
            return 0

        log.info("Episodic tensor: %d memories with embeddings", len(mem_ids))

        id_to_idx = {mid: i for i, mid in enumerate(mem_ids)}
        mem_tags = self._load_all_memory_tags()
        tag_tensor = self._load_tag_tensor_cache(mem_tags)
        mem_times = self._load_memory_timestamps()
        mem_sources = self._load_memory_sources()

        total_edges = 0
        for ep_idx, (ep_id, member_ids) in enumerate(episodes.items()):
            ep_members = [m for m in member_ids if m in id_to_idx]
            if len(ep_members) < 2:
                continue

            edges = self._process_episode(
                ep_members, id_to_idx, vectors, mem_tags, tag_tensor,
                mem_times, mem_sources, batch_size,
            )
            total_edges += edges

            if (ep_idx + 1) % 20 == 0:
                log.info("  processed %d / %d episodes (%d edges)",
                         ep_idx + 1, len(episodes), total_edges)

        self.prune()

        final = self._db.execute(
            "SELECT COUNT(*) as c FROM memory_relational_tensor "
            "WHERE edge_status != 'archived'"
        ).fetchone()["c"]
        log.info("Episodic tensor bootstrap: %d active edges", final)
        return final

    def _load_episode_members(self) -> dict[int, list[str]]:
        rows = self._db.execute(
            "SELECT memory_id, episode_id FROM memory_episode_assignments"
        ).fetchall()
        episodes: dict[int, list[str]] = {}
        for r in rows:
            episodes.setdefault(r["episode_id"], []).append(r["memory_id"])
        return episodes

    def _process_episode(
        self,
        ep_members: list[str],
        id_to_idx: dict[str, int],
        vectors: np.ndarray,
        mem_tags: dict[str, list[int]],
        tag_tensor: dict[tuple[int, int], dict],
        mem_times: dict[str, str],
        mem_sources: dict[str, str],
        batch_size: int,
    ) -> int:
        """Discover and score candidate edges within one episode."""
        indices = np.array([id_to_idx[m] for m in ep_members])
        ep_vectors = vectors[indices]
        n = len(ep_members)
        k = min(self._NEIGHBORS_K, n - 1)
        threshold = self._cfg.similarity_threshold

        sim_matrix = ep_vectors @ ep_vectors.T
        np.fill_diagonal(sim_matrix, -1.0)

        candidates: set[tuple[str, str]] = set()
        for i in range(n):
            row = sim_matrix[i]
            top_k = np.argpartition(row, -k)[-k:]
            for j in top_k:
                if row[int(j)] < threshold:
                    continue
                a = min(ep_members[i], ep_members[int(j)])
                b = max(ep_members[i], ep_members[int(j)])
                candidates.add((a, b))

        rows: list[tuple] = []
        edge_count = 0
        for a, b in candidates:
            edge = self._compute_eager_fields(
                a, b, id_to_idx, vectors, mem_tags, tag_tensor,
                mem_times, mem_sources,
            )
            if edge is not None:
                rows.append(edge)

            if len(rows) >= batch_size:
                self._batch_insert(rows)
                edge_count += len(rows)
                rows.clear()

        if rows:
            self._batch_insert(rows)
            edge_count += len(rows)

        return edge_count


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
    ) -> tuple | None:
        idx_a = id_to_idx.get(a)
        idx_b = id_to_idx.get(b)
        if idx_a is not None and idx_b is not None:
            semantic = float(np.dot(vectors[idx_a], vectors[idx_b]))
        else:
            semantic = 0.0

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

        time_a = mem_times.get(a, "")
        time_b = mem_times.get(b, "")
        temporal = self._temporal_proximity(time_a, time_b)

        src_a = mem_sources.get(a, "")
        src_b = mem_sources.get(b, "")
        source_prox = 1.0 if src_a and src_b and src_a == src_b else 0.0

        if tags_a and tags_b:
            jaccard = len(tags_a & tags_b) / len(tags_a | tags_b)
        else:
            jaccard = 0.0
        tag_distance = 1.0 - jaccard
        bridge = semantic * tag_distance

        composite = semantic + tag_projected + bridge
        if composite < self._cfg.episodic_min_edge_strength:
            return None

        return (
            a, b,
            semantic, tag_projected, 0.0, temporal, source_prox,
            tag_sem, tag_sync, tag_cooc, tag_inhib, tag_glyph,
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
            return float(np.exp(-0.693 * hours_apart / halflife))
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

    def _load_embeddings_for(self, memory_ids: set[str]) -> tuple[list[str], np.ndarray]:
        if not memory_ids:
            return [], np.array([])

        BATCH = 20_000
        id_list = list(memory_ids)
        all_rows = []
        for start in range(0, len(id_list), BATCH):
            chunk = id_list[start:start + BATCH]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._db.execute(
                f"""SELECT memory_id, embedding, dimension FROM embeddings
                   WHERE chunk_index = 0 AND memory_id IN ({placeholders})
                   ORDER BY memory_id""",
                tuple(chunk),
            ).fetchall()
            all_rows.extend(rows)

        if not all_rows:
            return [], np.array([])

        dim = all_rows[0]["dimension"]
        mem_ids = [r["memory_id"] for r in all_rows]
        all_bytes = b"".join(r["embedding"] for r in all_rows)
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


    # ------------------------------------------------------------------
    # Incremental update (for orchestrator integration)
    # ------------------------------------------------------------------

    def update_on_store(self, memory_id: str) -> int:
        """Create candidate edges for a newly stored memory.

        Scoped to the memory's temporal episode — only compares against
        episode-mates rather than the entire database.
        """
        row = self._db.execute(
            "SELECT embedding, dimension FROM embeddings "
            "WHERE memory_id = ? AND chunk_index = 0",
            (memory_id,),
        ).fetchone()
        if not row:
            return 0

        vec = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        dim = row["dimension"]

        ep_row = self._db.execute(
            "SELECT episode_id FROM memory_episode_assignments "
            "WHERE memory_id = ? ORDER BY assigned_at DESC LIMIT 1",
            (memory_id,),
        ).fetchone()
        if not ep_row:
            return 0

        mates = self._db.execute(
            "SELECT memory_id FROM memory_episode_assignments "
            "WHERE episode_id = ? AND memory_id != ?",
            (ep_row["episode_id"], memory_id),
        ).fetchall()
        mate_ids = [r["memory_id"] for r in mates]
        if not mate_ids:
            return 0

        ph = ",".join("?" * len(mate_ids))
        emb_rows = self._db.execute(
            f"SELECT memory_id, embedding FROM embeddings "
            f"WHERE memory_id IN ({ph}) AND chunk_index = 0",
            mate_ids,
        ).fetchall()
        if not emb_rows:
            return 0

        mate_mem_ids = [r["memory_id"] for r in emb_rows]
        mate_vectors = np.vstack([
            np.frombuffer(r["embedding"], dtype=np.float32).reshape(1, -1)
            for r in emb_rows
        ])
        norms = np.linalg.norm(mate_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mate_vectors /= norms

        scores = (mate_vectors @ vec).ravel()
        threshold = self._cfg.similarity_threshold
        k = min(self._NEIGHBORS_K, len(scores))
        top_k = np.argpartition(scores, -k)[-k:]
        above = [int(i) for i in top_k if scores[int(i)] >= threshold]

        if not above:
            return 0

        all_ids = [memory_id] + [mate_mem_ids[i] for i in above]
        all_vecs = np.vstack([vec.reshape(1, -1)] + [
            mate_vectors[i].reshape(1, -1) for i in above
        ])
        id_to_idx = {mid: i for i, mid in enumerate(all_ids)}

        scope_ph = ",".join("?" * len(all_ids))
        tag_rows = self._db.execute(
            f"SELECT memory_id, tag_id FROM memory_tags "
            f"WHERE memory_id IN ({scope_ph})",
            all_ids,
        ).fetchall()
        mem_tags: dict[str, list[int]] = {}
        for r in tag_rows:
            mem_tags.setdefault(r["memory_id"], []).append(r["tag_id"])

        tag_tensor = self._load_tag_tensor_cache(mem_tags)

        time_rows = self._db.execute(
            f"SELECT id, created_at FROM memories WHERE id IN ({scope_ph})",
            all_ids,
        ).fetchall()
        mem_times = {r["id"]: r["created_at"] for r in time_rows}

        src_rows = self._db.execute(
            f"SELECT id, source_model FROM memories WHERE id IN ({scope_ph})",
            all_ids,
        ).fetchall()
        mem_sources = {r["id"]: r["source_model"] for r in src_rows}

        rows: list[tuple] = []
        for idx in above:
            other = mate_mem_ids[idx]
            a, b = min(memory_id, other), max(memory_id, other)
            edge = self._compute_eager_fields(
                a, b, id_to_idx, all_vecs, mem_tags, tag_tensor,
                mem_times, mem_sources,
            )
            if edge is not None:
                rows.append(edge)

        if rows:
            self._batch_insert(rows)
        return len(rows)

    def activate_edge(
        self, memory_a: str, memory_b: str, outflow_strength: float = 0.0,
    ) -> None:
        """Record an activation, EMA-blend co_retrieval_strength, and promote lifecycle.

        co_retrieval_strength is EMA'd toward (outflow_strength × status_weight),
        where status_weight reflects lifecycle maturity (candidate 0.25 → mature 1.0).

        Lifecycle promotion uses the composite signal (semantic +
        tag_projected + co_retrieval + bridge).
        """
        a, b = min(memory_a, memory_b), max(memory_a, memory_b)
        threshold = self._cfg.episodic_tag_affinity_threshold
        alpha = self._cfg.episodic_co_retrieval_ema_alpha

        self._db.execute(
            """UPDATE memory_relational_tensor SET
                   activation_count = activation_count + 1,
                   last_activated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now'),
                   co_retrieval_strength =
                       ? * (? * CASE edge_status
                                  WHEN 'mature'    THEN 1.0
                                  WHEN 'warm'      THEN 0.5
                                  WHEN 'candidate' THEN 0.25
                                  WHEN 'decaying'  THEN 0.1
                                  ELSE 0.0
                                END)
                       + (1.0 - ?) * co_retrieval_strength,
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
            (alpha, outflow_strength, alpha, threshold, threshold, a, b),
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
