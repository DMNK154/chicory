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

        import time as _time
        _rt = {}
        _rt0 = _time.perf_counter()

        tag_ids = sorted(set(activated_tag_ids))
        _rt["n_input_tags"] = len(tag_ids)

        _s = _time.perf_counter()
        centroids = self.get_centroids_batch(tag_ids)
        tag_ids = [t for t in tag_ids if t in centroids]
        _rt["get_centroids"] = _time.perf_counter() - _s
        if len(tag_ids) < 2:
            return {}

        _rt["n_tags_with_centroids"] = len(tag_ids)

        _s = _time.perf_counter()
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
        _rt["compute_incoming"] = _time.perf_counter() - _s
        _rt["n_pairs"] = len(incoming)

        if not incoming:
            return {}

        _s = _time.perf_counter()
        ranked = sorted(incoming.items(), key=lambda x: x[1], reverse=True)
        values = [v for _, v in ranked]
        inverted_values = list(reversed(values))

        k = len(ranked)
        d = C.shape[1]
        D = np.zeros((k, d), dtype=np.float32)
        for idx, ((a, b), _) in enumerate(ranked):
            diff = C[tag_to_idx[a]] - C[tag_to_idx[b]]
            norm = np.linalg.norm(diff)
            if norm > 0:
                D[idx] = diff / norm

        P = np.abs(D @ D.T)  # (k, k)

        parallelness = np.zeros(k, dtype=np.float32)
        for i in range(1, k):
            parallelness[i] = float(np.max(P[i, :i]))

        deltas: dict[tuple[int, int], tuple[float, float]] = {}
        for idx, (pair, add_val) in enumerate(ranked):
            scaled_sub = inverted_values[idx] * float(parallelness[idx])
            deltas[pair] = (add_val, scaled_sub)
        _rt["rank_and_parallel"] = _time.perf_counter() - _s

        _s = _time.perf_counter()
        net_deltas = self._apply_deltas_to_tensor(deltas)
        _rt["apply_tensor"] = _time.perf_counter() - _s

        _s = _time.perf_counter()
        self._apply_deltas_to_resonances(deltas)
        _rt["apply_resonances"] = _time.perf_counter() - _s

        _rt["total"] = _time.perf_counter() - _rt0
        import logging
        logging.getLogger("chicory.centroid").info("reweight_timing: %s", _rt)

        return net_deltas

    def _apply_deltas_to_tensor(
        self,
        deltas: dict[tuple[int, int], tuple[float, float]],
    ) -> dict[tuple[int, int], float]:
        """EMA-update tensor synchronicity_strength toward incoming signal.

        For each pair: new = alpha * signal + (1 - alpha) * old
        where signal = add_val - sub_val (net incoming strength).
        Returns net delta per pair.
        """
        pairs = list(deltas.keys())
        if not pairs:
            return {}

        alpha = self._config.centroid_edge_ema_alpha

        existing: dict[tuple[int, int], float] = {}
        chunk_size = 400
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
            signal = max(0.0, add_val - sub_val)
            new_val = alpha * signal + (1.0 - alpha) * old
            updates.append((new_val, pair[0], pair[1]))
            net_deltas[pair] = new_val - old

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
        """EMA-update resonance_strength toward incoming signal.

        For each resonance, compute the net signal from overlapping
        tag pairs, then apply: new = alpha * signal + (1 - alpha) * old.
        Uses sync_event_tags junction table for indexed lookup.
        """
        if not deltas:
            return

        all_tags: set[int] = set()
        for a, b in deltas:
            all_tags.add(a)
            all_tags.add(b)
        tag_list = sorted(all_tags)

        # Find resonances involving any of the activated tags via indexed junction
        min_res = self._config.resonance_min_edge_strength
        matched: dict[int, dict] = {}
        for i in range(0, len(tag_list), 500):
            chunk = tag_list[i:i + 500]
            ph = ",".join("?" * len(chunk))
            rows = self._db.execute(
                f"""SELECT r.id, r.resonance_strength,
                       r.event_a_id, r.event_b_id
                    FROM resonances r
                    JOIN sync_event_tags st ON st.event_id = r.event_a_id
                    WHERE st.tag_id IN ({ph}) AND r.resonance_strength >= ?
                    UNION ALL
                    SELECT r.id, r.resonance_strength,
                       r.event_a_id, r.event_b_id
                    FROM resonances r
                    JOIN sync_event_tags st ON st.event_id = r.event_b_id
                    WHERE st.tag_id IN ({ph}) AND r.resonance_strength >= ?""",
                (*chunk, min_res, *chunk, min_res),
            ).fetchall()
            for row in rows:
                matched[row["id"]] = row

        if not matched:
            return

        # Batch-load tags for matched events via junction table
        event_ids: set[int] = set()
        for row in matched.values():
            event_ids.add(row["event_a_id"])
            event_ids.add(row["event_b_id"])

        event_tags: dict[int, set[int]] = {eid: set() for eid in event_ids}
        eid_list = sorted(event_ids)
        for i in range(0, len(eid_list), 500):
            chunk = eid_list[i:i + 500]
            ph = ",".join("?" * len(chunk))
            rows = self._db.execute(
                f"SELECT event_id, tag_id FROM sync_event_tags WHERE event_id IN ({ph})",
                tuple(chunk),
            ).fetchall()
            for r in rows:
                event_tags[r["event_id"]].add(r["tag_id"])

        alpha = self._config.centroid_edge_ema_alpha
        updates: list[tuple] = []
        for row in matched.values():
            tags_a = event_tags.get(row["event_a_id"], set())
            tags_b = event_tags.get(row["event_b_id"], set())

            signal = 0.0
            for ta in tags_a:
                for tb in tags_b:
                    key = (min(ta, tb), max(ta, tb))
                    if key in deltas:
                        add_val, sub_val = deltas[key]
                        signal += add_val - sub_val

            signal = max(0.0, signal)
            old = row["resonance_strength"]
            new_strength = alpha * signal + (1.0 - alpha) * old

            if abs(new_strength - old) < 1e-6:
                continue

            updates.append((new_strength, row["id"]))

        if updates:
            self._db.executemany(
                "UPDATE resonances SET resonance_strength = ? WHERE id = ?",
                updates,
            )
