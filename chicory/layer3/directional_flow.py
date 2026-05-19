"""Directional flow detectors: inflow (convergence attractor) and outflow (distributor).

Direction is defined by retrieval causality: query tags are SOURCE,
result memories are DESTINATION.  The inflow detector lives in memory-space
and tracks clusters that get pulled in by diverse queries.  The outflow
detector lives in tag-space and tracks query neighborhoods that reach
diverse, distant result clusters.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from typing import TYPE_CHECKING

from chicory.models.canopy import InflowScore, OutflowScore

if TYPE_CHECKING:
    from chicory.config import ChicoryConfig
    from chicory.db.engine import DatabaseEngine

logger = logging.getLogger(__name__)


def _hash_sorted_ids(ids: list) -> str:
    raw = json.dumps(sorted(str(i) for i in ids))
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _mean_pairwise_jaccard(sets: list[set]) -> float:
    """Average Jaccard similarity across all C(n,2) pairs."""
    n = len(sets)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            union = sets[i] | sets[j]
            if union:
                total += len(sets[i] & sets[j]) / len(union)
            pairs += 1
    return total / pairs if pairs else 0.0


def _sample_correction(n: int) -> float:
    return 1.0 - 1.0 / math.sqrt(n + 1)


class InflowObserver:
    """Tracks result memory clusters that are convergence attractors."""

    def __init__(self, db: DatabaseEngine, config: ChicoryConfig) -> None:
        self._db = db
        self._cfg = config

    def observe(
        self,
        retrieval_id: int,
        query_tag_ids: list[int],
        query_tag_hash: str,
        result_memory_ids: list[str],
        result_tag_ids: list[int],
    ) -> InflowScore:
        if len(result_memory_ids) < 2:
            return InflowScore()

        block_key = _hash_sorted_ids(result_memory_ids)

        historical_hashes = self._db.execute(
            "SELECT DISTINCT query_tag_hash FROM inflow_canopy_observations "
            "WHERE block_key = ?",
            (block_key,),
        ).fetchall()
        past_hashes = {r["query_tag_hash"] for r in historical_hashes}
        past_hashes.add(query_tag_hash)
        unique_contexts = len(past_hashes)

        query_tag_sets = self._load_historical_query_tag_sets(block_key)
        query_tag_sets.append(set(query_tag_ids))

        diversity = 1.0 - _mean_pairwise_jaccard(query_tag_sets)
        correction = _sample_correction(unique_contexts)

        total_row = self._db.execute(
            "SELECT total_activations FROM inflow_canopy_blocks "
            "WHERE block_key = ?",
            (block_key,),
        ).fetchone()
        prev_activations = total_row["total_activations"] if total_row else 0
        total_activations = prev_activations + 1

        max_row = self._db.execute(
            "SELECT MAX(total_activations) as mx FROM inflow_canopy_blocks"
        ).fetchone()
        max_activations = max((max_row["mx"] or 0) if max_row else 0, total_activations)

        freq_factor = (
            math.log(1.0 + total_activations) / math.log(1.0 + max_activations)
            if max_activations > 0
            else 0.0
        )

        w_div = self._cfg.canopy_directional_inflow_diversity_weight
        w_freq = self._cfg.canopy_directional_inflow_frequency_weight
        raw_strength = (w_div * diversity + w_freq * freq_factor) * correction

        prev_row = self._db.execute(
            "SELECT inflow_strength FROM inflow_canopy_blocks WHERE block_key = ?",
            (block_key,),
        ).fetchone()
        alpha = self._cfg.canopy_directional_ema_alpha
        if prev_row is not None:
            strength = alpha * raw_strength + (1.0 - alpha) * prev_row["inflow_strength"]
        else:
            strength = raw_strength

        score = InflowScore(
            block_key=block_key,
            inflow_diversity=diversity,
            unique_query_contexts=unique_contexts,
            total_activations=total_activations,
            inflow_strength=strength,
        )

        self._db.execute(
            """INSERT INTO inflow_canopy_blocks
               (block_key, memory_ids, representative_tags,
                inflow_diversity, inflow_strength,
                unique_query_contexts, total_activations, evidence_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, 1)
               ON CONFLICT(block_key) DO UPDATE SET
                   inflow_diversity = ?,
                   inflow_strength = ?,
                   unique_query_contexts = ?,
                   total_activations = ?,
                   evidence_count = evidence_count + 1,
                   last_observed_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')""",
            (
                block_key,
                json.dumps(sorted(result_memory_ids)),
                json.dumps(sorted(result_tag_ids)),
                diversity, strength, unique_contexts, total_activations,
                diversity, strength, unique_contexts, total_activations,
            ),
        )

        self._db.execute(
            """INSERT INTO inflow_canopy_observations
               (block_key, retrieval_id, query_tag_hash, query_tag_ids,
                result_memory_ids, inflow_diversity, inflow_strength)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                block_key, retrieval_id, query_tag_hash,
                json.dumps(sorted(query_tag_ids)),
                json.dumps(sorted(result_memory_ids)),
                diversity, strength,
            ),
        )

        return score

    def _load_historical_query_tag_sets(self, block_key: str) -> list[set]:
        rows = self._db.execute(
            "SELECT query_tag_ids FROM inflow_canopy_observations "
            "WHERE block_key = ? ORDER BY observed_at DESC LIMIT 50",
            (block_key,),
        ).fetchall()
        return [set(json.loads(r["query_tag_ids"])) for r in rows]

    def compute_pair_diversity(
        self,
        block_key: str,
        tag_pairs: list[tuple[int, int]],
    ) -> dict[tuple[int, int], float]:
        """Per-pair diversity: fraction of historical query contexts containing both tags."""
        if not tag_pairs or not block_key:
            return {pair: 0.0 for pair in tag_pairs}

        query_tag_sets = self._load_historical_query_tag_sets(block_key)
        n = len(query_tag_sets)
        if n == 0:
            return {pair: 0.0 for pair in tag_pairs}

        result: dict[tuple[int, int], float] = {}
        for a, b in tag_pairs:
            count = sum(1 for s in query_tag_sets if a in s and b in s)
            result[(a, b)] = count / n
        return result


class OutflowObserver:
    """Tracks query tag neighborhoods that are distributors."""

    def __init__(self, db: DatabaseEngine, config: ChicoryConfig) -> None:
        self._db = db
        self._cfg = config

    def observe(
        self,
        retrieval_id: int,
        query_tag_ids: list[int],
        query_tag_hash: str,
        result_memory_ids: list[str],
    ) -> OutflowScore:
        if not query_tag_ids:
            return OutflowScore()

        block_key = _hash_sorted_ids(query_tag_ids)
        result_cluster_key = _hash_sorted_ids(result_memory_ids)

        result_sets = self._load_historical_result_sets(block_key)
        result_sets.append(set(result_memory_ids))

        diversity = 1.0 - _mean_pairwise_jaccard(result_sets)

        reach = 0.0
        n = len(result_sets)
        if n >= 2:
            total_dist = 0.0
            pairs = 0
            for i in range(n):
                for j in range(i + 1, n):
                    union = result_sets[i] | result_sets[j]
                    shared = result_sets[i] & result_sets[j]
                    total_dist += 1.0 - (len(shared) / len(union)) if union else 0.0
                    pairs += 1
            reach = total_dist / pairs if pairs else 0.0

        distinct_clusters = self._count_distinct_result_clusters(block_key)
        if result_cluster_key not in self._get_known_cluster_keys(block_key):
            distinct_clusters += 1

        total_row = self._db.execute(
            "SELECT total_activations FROM outflow_canopy_blocks "
            "WHERE block_key = ?",
            (block_key,),
        ).fetchone()
        total_activations = (total_row["total_activations"] if total_row else 0) + 1

        correction = _sample_correction(distinct_clusters)
        w_div = self._cfg.canopy_directional_outflow_diversity_weight
        w_reach = self._cfg.canopy_directional_outflow_reach_weight
        raw_strength = (w_div * diversity + w_reach * reach) * correction

        prev_row = self._db.execute(
            "SELECT outflow_strength FROM outflow_canopy_blocks WHERE block_key = ?",
            (block_key,),
        ).fetchone()
        alpha = self._cfg.canopy_directional_ema_alpha
        if prev_row is not None:
            strength = alpha * raw_strength + (1.0 - alpha) * prev_row["outflow_strength"]
        else:
            strength = raw_strength

        score = OutflowScore(
            outflow_diversity=diversity,
            outflow_reach=reach,
            unique_result_clusters=distinct_clusters,
            total_activations=total_activations,
            outflow_strength=strength,
        )

        self._db.execute(
            """INSERT INTO outflow_canopy_blocks
               (block_key, query_tag_ids, query_tag_hash,
                outflow_diversity, outflow_reach, outflow_strength,
                unique_result_clusters, total_activations, evidence_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
               ON CONFLICT(block_key) DO UPDATE SET
                   outflow_diversity = ?,
                   outflow_reach = ?,
                   outflow_strength = ?,
                   unique_result_clusters = ?,
                   total_activations = ?,
                   evidence_count = evidence_count + 1,
                   last_observed_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')""",
            (
                block_key,
                json.dumps(sorted(query_tag_ids)),
                query_tag_hash,
                diversity, reach, strength, distinct_clusters, total_activations,
                diversity, reach, strength, distinct_clusters, total_activations,
            ),
        )

        self._db.execute(
            """INSERT INTO outflow_canopy_observations
               (block_key, retrieval_id, result_memory_ids,
                result_cluster_key, outflow_diversity, outflow_reach,
                outflow_strength)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                block_key, retrieval_id,
                json.dumps(sorted(result_memory_ids)),
                result_cluster_key,
                diversity, reach, strength,
            ),
        )

        return score

    def _load_historical_result_sets(self, block_key: str) -> list[set]:
        rows = self._db.execute(
            "SELECT result_memory_ids FROM outflow_canopy_observations "
            "WHERE block_key = ? ORDER BY observed_at DESC LIMIT 50",
            (block_key,),
        ).fetchall()
        return [set(json.loads(r["result_memory_ids"])) for r in rows]

    def _count_distinct_result_clusters(self, block_key: str) -> int:
        row = self._db.execute(
            "SELECT COUNT(DISTINCT result_cluster_key) as c "
            "FROM outflow_canopy_observations WHERE block_key = ?",
            (block_key,),
        ).fetchone()
        return row["c"] if row else 0

    def _get_known_cluster_keys(self, block_key: str) -> set[str]:
        rows = self._db.execute(
            "SELECT DISTINCT result_cluster_key FROM outflow_canopy_observations "
            "WHERE block_key = ?",
            (block_key,),
        ).fetchall()
        return {r["result_cluster_key"] for r in rows}
