"""Auto-classify documents as critical or reference based on tag network overlap.

Scores each incoming document by:
1. Tag coverage — fraction of its tags already present in the memory network
2. Tag weight — log-frequency of matching tags (popular tags = stronger signal)
3. Cluster density — Jaccard overlap with other incoming documents

Documents scoring above the threshold become 'critical' (full content + deep
network integration). Below → 'reference' (summary only + tags + lightweight
embedding).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chicory.db.engine import DatabaseEngine


def classify_documents(
    file_tags: dict[str, list[str]],
    db: DatabaseEngine,
    threshold: float = 0.3,
) -> dict[str, str]:
    """Classify files as 'critical' or 'reference'.

    Args:
        file_tags: mapping of filepath → derived tag list
        db: database with existing memory network
        threshold: score cutoff (0-1); above = critical

    Returns:
        mapping of filepath → 'critical' or 'reference'
    """
    if not file_tags:
        return {}

    existing_tag_counts = _get_existing_tag_counts(db)
    existing_tag_names = set(existing_tag_counts.keys())

    scores: dict[str, float] = {}

    for filepath, tags in file_tags.items():
        scores[filepath] = _network_relevance(tags, existing_tag_names, existing_tag_counts)

    _add_cluster_density(scores, file_tags)

    return {
        fp: "critical" if score >= threshold else "reference"
        for fp, score in scores.items()
    }


def _get_existing_tag_counts(db: DatabaseEngine) -> dict[str, int]:
    rows = db.execute(
        "SELECT t.name, COUNT(mt.memory_id) AS mem_count "
        "FROM tags t "
        "JOIN memory_tags mt ON mt.tag_id = t.id "
        "GROUP BY t.name"
    ).fetchall()
    return {r["name"]: r["mem_count"] for r in rows}


def _network_relevance(
    tags: list[str],
    existing_names: set[str],
    existing_counts: dict[str, int],
) -> float:
    """Score how connected a document's tags are to the existing network."""
    if not tags:
        return 0.0

    known_tags = [t for t in tags if t in existing_names]
    coverage = len(known_tags) / len(tags)

    if not known_tags:
        return 0.0

    log_weights = [math.log1p(existing_counts.get(t, 0)) for t in known_tags]
    max_weight = max(log_weights) if log_weights else 1.0
    avg_weight = sum(log_weights) / len(log_weights)
    normalized_weight = avg_weight / max_weight if max_weight > 0 else 0.0

    return coverage * (0.6 + 0.4 * normalized_weight)


def _add_cluster_density(
    scores: dict[str, float],
    file_tags: dict[str, list[str]],
) -> None:
    """Boost scores for documents that cluster with other incoming docs."""
    paths = list(file_tags.keys())
    if len(paths) < 2:
        return

    tag_sets = {fp: set(file_tags[fp]) for fp in paths}

    for fp_i in paths:
        tags_i = tag_sets[fp_i]
        if not tags_i:
            continue

        jaccard_sum = 0.0
        for fp_j in paths:
            if fp_i == fp_j:
                continue
            tags_j = tag_sets[fp_j]
            if not tags_j:
                continue
            intersection = len(tags_i & tags_j)
            union = len(tags_i | tags_j)
            if union > 0:
                jaccard_sum += intersection / union

        cluster_density = jaccard_sum / (len(paths) - 1)
        scores[fp_i] = 0.6 * scores[fp_i] + 0.4 * cluster_density
