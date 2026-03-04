"""Retrieval-driven centroid sub-graph reweighting.

Maintains tag embedding centroids incrementally and tracks co-retrieval
edges between tags.  On each retrieval:

  1. Compute incoming association strengths (centroid similarity × intensity)
  2. ADD incoming strengths to existing resonance/tensor scores
  3. Rank incoming strengths, invert the values, SUBTRACT inverted amounts
     from existing scores where they overlap

Strongest retrieval associations grow; weakest ones shrink.  Old connections
that stop being retrieved lose strength as new retrievals erode them.
"""

from __future__ import annotations

import itertools
import json
import logging
from typing import Optional

import numpy as np

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine

logger = logging.getLogger(__name__)


# ── Blob helpers (same format as embedding_engine) ──────────────────────


def _array_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _blob_to_array(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


class CentroidSubgraph:
    """Tag centroid maintenance and retrieval-driven resonance reweighting."""

    def __init__(
        self,
        config: ChicoryConfig,
        db: DatabaseEngine,
        embedding_engine: object,  # EmbeddingEngine — avoid circular import
    ) -> None:
        self._config = config
        self._db = db
        self._embedding_engine = embedding_engine

    # ── Centroid Maintenance ────────────────────────────────────────────

    def update_centroid_on_store(
        self, tag_id: int, embedding: np.ndarray,
    ) -> None:
        """Incrementally update a tag's centroid via EMA when a memory is stored."""
        row = self._db.execute(
            "SELECT centroid, memory_count FROM tag_centroids WHERE tag_id = ?",
            (tag_id,),
        ).fetchone()

        alpha = self._config.centroid_ema_alpha
        if row is None:
            # First memory for this tag — initialize
            centroid = embedding.astype(np.float32)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            self._db.execute(
                "INSERT INTO tag_centroids (tag_id, centroid, memory_count) VALUES (?, ?, ?)",
                (tag_id, _array_to_blob(centroid), 1),
            )
        else:
            old = _blob_to_array(row["centroid"])
            count = row["memory_count"]
            new_centroid = alpha * embedding + (1 - alpha) * old
            norm = np.linalg.norm(new_centroid)
            if norm > 0:
                new_centroid = new_centroid / norm
            self._db.execute(
                "UPDATE tag_centroids SET centroid = ?, memory_count = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') WHERE tag_id = ?",
                (_array_to_blob(new_centroid), count + 1, tag_id),
            )

    def update_centroids_batch(
        self, pairs: list[tuple[int, np.ndarray]],
    ) -> None:
        """Batch update multiple tag centroids (e.g. after ingest)."""
        for tag_id, embedding in pairs:
            self.update_centroid_on_store(tag_id, embedding)

    def get_centroids_batch(
        self, tag_ids: list[int],
    ) -> dict[int, np.ndarray]:
        """Load centroids for multiple tags. Returns {tag_id: unit_vector}."""
        if not tag_ids:
            return {}
        placeholders = ",".join("?" for _ in tag_ids)
        rows = self._db.execute(
            f"SELECT tag_id, centroid FROM tag_centroids WHERE tag_id IN ({placeholders})",
            tuple(tag_ids),
        ).fetchall()
        return {r["tag_id"]: _blob_to_array(r["centroid"]) for r in rows}

    def rebuild_centroids(self) -> int:
        """Full recomputation of all tag centroids from embeddings.

        Used on first boot after schema migration.  Returns count of
        centroids computed.
        """
        # Get all (tag_id, memory_id) pairs
        rows = self._db.execute(
            "SELECT mt.tag_id, mt.memory_id "
            "FROM memory_tags mt "
            "JOIN memories m ON m.id = mt.memory_id "
            "WHERE m.is_archived = 0"
        ).fetchall()

        if not rows:
            return 0

        # Group memory_ids by tag
        tag_memories: dict[int, list[str]] = {}
        for r in rows:
            tag_memories.setdefault(r["tag_id"], []).append(r["memory_id"])

        # Load all first-chunk embeddings
        all_cached = self._embedding_engine.get_all_cached()

        count = 0
        upserts: list[tuple] = []
        for tag_id, memory_ids in tag_memories.items():
            vecs = [all_cached[mid] for mid in memory_ids if mid in all_cached]
            if not vecs:
                continue
            centroid = np.mean(vecs, axis=0).astype(np.float32)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            upserts.append((_array_to_blob(centroid), len(vecs), tag_id))
            count += 1

        if upserts:
            self._db.executemany(
                "INSERT INTO tag_centroids (centroid, memory_count, tag_id) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(tag_id) DO UPDATE SET "
                "centroid = excluded.centroid, memory_count = excluded.memory_count, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')",
                # executemany expects (centroid, count, tag_id)
                upserts,
            )
            self._db.connection.commit()

        logger.info("Rebuilt %d tag centroids from embeddings", count)
        return count

    # ── Co-Retrieval Edge Tracking ─────────────────────────────────────

    def record_co_retrieval(self, tag_ids: list[int]) -> None:
        """Record that these tags were co-retrieved.

        For every pair, applies EMA: strength = alpha * 1.0 + (1 - alpha) * old.
        Uses UPSERT with SQL expression so no pre-load is needed.
        """
        if len(tag_ids) < 2:
            return

        alpha = self._config.centroid_edge_ema_alpha
        pairs = list(itertools.combinations(sorted(set(tag_ids)), 2))

        if pairs:
            self._db.executemany(
                "INSERT INTO centroid_edges (tag_a_id, tag_b_id, edge_strength, co_retrieval_count) "
                "VALUES (?, ?, ?, 1) "
                "ON CONFLICT(tag_a_id, tag_b_id) DO UPDATE SET "
                "edge_strength = ? * 1.0 + (1.0 - ?) * edge_strength, "
                "co_retrieval_count = co_retrieval_count + 1, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')",
                [(a, b, alpha, alpha, alpha) for a, b in pairs],
            )

    def rebuild_edges_from_history(self) -> int:
        """Populate co-retrieval edges from retrieval_tag_hits history.

        Groups tag hits by retrieval_id, computes co-occurrence pairs,
        and builds edge strengths with decayed EMA.  Returns edge count.
        """
        rows = self._db.execute(
            "SELECT rth.retrieval_id, rth.tag_id "
            "FROM retrieval_tag_hits rth "
            "ORDER BY rth.retrieval_id"
        ).fetchall()

        if not rows:
            return 0

        # Group tags by retrieval
        retrieval_tags: dict[int, list[int]] = {}
        for r in rows:
            retrieval_tags.setdefault(r["retrieval_id"], []).append(r["tag_id"])

        # Accumulate edge strengths via EMA (ordered by retrieval_id = chronological)
        alpha = self._config.centroid_edge_ema_alpha
        edges: dict[tuple[int, int], tuple[float, int]] = {}

        for _rid, tag_ids in sorted(retrieval_tags.items()):
            for a, b in itertools.combinations(sorted(set(tag_ids)), 2):
                old_strength, old_count = edges.get((a, b), (0.0, 0))
                edges[(a, b)] = (
                    alpha * 1.0 + (1 - alpha) * old_strength,
                    old_count + 1,
                )

        # Batch write
        upserts = [
            (a, b, strength, count)
            for (a, b), (strength, count) in edges.items()
        ]
        if upserts:
            self._db.executemany(
                "INSERT INTO centroid_edges (tag_a_id, tag_b_id, edge_strength, co_retrieval_count) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(tag_a_id, tag_b_id) DO UPDATE SET "
                "edge_strength = excluded.edge_strength, "
                "co_retrieval_count = excluded.co_retrieval_count, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')",
                upserts,
            )
            self._db.connection.commit()

        logger.info("Rebuilt %d co-retrieval edges from history", len(edges))
        return len(edges)

    # ── Retrieval Reweighting: Add + Inverted Subtract ────────────────

    def update_on_retrieval(
        self,
        activated_tag_ids: list[int],
        mean_relevance: float,
    ) -> dict[tuple[int, int], float]:
        """Core algorithm.  Called after each retrieval.

        1. Compute incoming association strength per tag pair
           (centroid cosine similarity × mean retrieval relevance)
        2. ADD incoming strengths to existing resonance/tensor scores
        3. Rank incoming strengths highest→lowest, invert the values
           (highest incoming → smallest subtraction value, lowest → largest)
        4. Scale each subtraction by **parallelness**: how geometrically
           aligned the pair's direction is with stronger pairs.
           Orthogonal (independent) pairs get near-zero subtraction;
           parallel (redundant) pairs get full subtraction.
        5. SUBTRACT scaled values from existing scores where they overlap

        Returns the net delta map {(tag_a, tag_b): net_change}.
        """
        if len(activated_tag_ids) < 2:
            return {}

        tag_ids = sorted(set(activated_tag_ids))
        centroids = self.get_centroids_batch(tag_ids)
        tag_ids = [t for t in tag_ids if t in centroids]
        if len(tag_ids) < 2:
            return {}

        # ── Compute incoming association strengths ──────────────────
        tag_to_idx = {t: i for i, t in enumerate(tag_ids)}
        n = len(tag_ids)
        C = np.stack([centroids[t] for t in tag_ids])  # (n, d)
        sim_matrix = C @ C.T  # cosine sim (unit-normalized centroids)

        scale = self._config.centroid_inhibition_scale
        incoming: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                strength = float(sim_matrix[i, j]) * mean_relevance * scale
                if strength > 0:
                    incoming[(tag_ids[i], tag_ids[j])] = strength

        if not incoming:
            return {}

        # ── Rank incoming and build inverted subtraction values ─────
        ranked = sorted(incoming.items(), key=lambda x: x[1], reverse=True)
        values = [v for _, v in ranked]
        inverted_values = list(reversed(values))

        # ── Compute direction vectors for each ranked pair ──────────
        # Direction = normalize(centroid_a - centroid_b) — the "axis"
        # of the association in embedding space.
        k = len(ranked)
        d = C.shape[1]
        D = np.zeros((k, d), dtype=np.float32)
        for idx, ((a, b), _) in enumerate(ranked):
            diff = C[tag_to_idx[a]] - C[tag_to_idx[b]]
            norm = np.linalg.norm(diff)
            if norm > 0:
                D[idx] = diff / norm

        # Parallelness matrix: |D @ D.T| — absolute cosine between
        # direction vectors.  Anti-parallel still means competing.
        P = np.abs(D @ D.T)  # (k, k)

        # For each pair, parallelness = max |cos| with any stronger pair.
        # Top-ranked pair has no reference → parallelness = 0 → zero subtraction.
        # Orthogonal pairs get near-zero subtraction regardless of rank.
        parallelness = np.zeros(k, dtype=np.float32)
        for i in range(1, k):
            parallelness[i] = float(np.max(P[i, :i]))

        # Map: pair → (add_amount, scaled_subtract_amount)
        deltas: dict[tuple[int, int], tuple[float, float]] = {}
        for idx, (pair, add_val) in enumerate(ranked):
            scaled_sub = inverted_values[idx] * float(parallelness[idx])
            deltas[pair] = (add_val, scaled_sub)

        # ── Apply to tensor (direct tag-pair mapping) ──────────────
        net_deltas = self._apply_deltas_to_tensor(deltas)

        # ── Apply to resonances (event-pair mapping) ───────────────
        self._apply_deltas_to_resonances(deltas)

        return net_deltas

    def _apply_deltas_to_tensor(
        self,
        deltas: dict[tuple[int, int], tuple[float, float]],
    ) -> dict[tuple[int, int], float]:
        """Add incoming and subtract inverted from tensor synchronicity_strength.

        Returns net delta per pair.
        """
        pairs = list(deltas.keys())
        if not pairs:
            return {}

        # Load existing tensor values in chunks to stay within SQLite limits
        existing: dict[tuple[int, int], float] = {}
        chunk_size = 400  # 2 params per pair → 800 params per chunk
        for i in range(0, len(pairs), chunk_size):
            chunk = pairs[i : i + chunk_size]
            conds = " OR ".join(
                "(tag_a_id = ? AND tag_b_id = ?)" for _ in chunk
            )
            params: list[int] = []
            for a, b in chunk:
                params.extend([a, b])
            rows = self._db.execute(
                f"SELECT tag_a_id, tag_b_id, synchronicity_strength "
                f"FROM tag_relational_tensor WHERE {conds}",
                tuple(params),
            ).fetchall()
            for r in rows:
                existing[(r["tag_a_id"], r["tag_b_id"])] = r["synchronicity_strength"]

        updates: list[tuple] = []
        net_deltas: dict[tuple[int, int], float] = {}
        for pair, (add_val, sub_val) in deltas.items():
            old = existing.get(pair)
            if old is None:
                continue
            net = add_val - sub_val
            new_val = max(0.0, old + net)
            updates.append((new_val, pair[0], pair[1]))
            net_deltas[pair] = net

        if updates:
            self._db.executemany(
                "UPDATE tag_relational_tensor SET synchronicity_strength = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') "
                "WHERE tag_a_id = ? AND tag_b_id = ?",
                updates,
            )

        return net_deltas

    def _apply_deltas_to_resonances(
        self,
        deltas: dict[tuple[int, int], tuple[float, float]],
    ) -> None:
        """Add incoming and subtract inverted from resonance_strength.

        For each resonance, sum the net deltas across all cross-event
        tag pairs that overlap with the delta map.
        """
        if not deltas:
            return

        all_tags: set[int] = set()
        for a, b in deltas:
            all_tags.add(a)
            all_tags.add(b)
        tag_list = sorted(all_tags)

        # Query resonances in chunks to stay within SQLite expression limits.
        # Each tag generates 2 LIKE clauses (sea + seb), so limit chunk size.
        matched: dict[int, dict] = {}  # resonance id → row
        chunk_size = 200
        for i in range(0, len(tag_list), chunk_size):
            chunk = tag_list[i : i + chunk_size]
            like_clauses = " OR ".join(
                "sea.involved_tags LIKE '%' || ? || '%'" for _ in chunk
            )
            sql = (
                f"SELECT r.id, r.resonance_strength, "
                f"  sea.involved_tags AS tags_a, seb.involved_tags AS tags_b "
                f"FROM resonances r "
                f"JOIN synchronicity_events sea ON sea.id = r.event_a_id "
                f"JOIN synchronicity_events seb ON seb.id = r.event_b_id "
                f"WHERE ({like_clauses}) "
                f"   OR ({like_clauses.replace('sea.', 'seb.')}) "
                f"ORDER BY r.resonance_strength DESC LIMIT 1000"
            )
            params = tuple(str(t) for t in chunk) * 2
            for row in self._db.execute(sql, params).fetchall():
                matched[row["id"]] = row

        if not matched:
            return

        updates: list[tuple] = []
        for row in matched.values():
            try:
                tags_a = set(json.loads(row["tags_a"]))
                tags_b = set(json.loads(row["tags_b"]))
            except (json.JSONDecodeError, TypeError):
                continue

            total_net = 0.0
            for ta in tags_a:
                for tb in tags_b:
                    key = (min(ta, tb), max(ta, tb))
                    if key in deltas:
                        add_val, sub_val = deltas[key]
                        total_net += add_val - sub_val

            if total_net == 0.0:
                continue

            new_strength = max(0.0, row["resonance_strength"] + total_net)
            updates.append((new_strength, row["id"]))

        if updates:
            self._db.executemany(
                "UPDATE resonances SET resonance_strength = ? WHERE id = ?",
                updates,
            )
