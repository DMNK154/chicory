"""FAISS-backed vector index for approximate nearest-neighbor search."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np


def _import_faiss():
    """Lazy import of faiss — only needed when actually building/searching."""
    import faiss
    return faiss


class VectorIndex:
    """Wraps a FAISS IndexIVFFlat for fast similarity search.

    Partitions vectors into Voronoi cells via k-means, then searches
    only the nearest ``nprobe`` cells at query time.  Falls back to
    IndexFlatIP (exact SIMD-accelerated search) when the corpus is
    too small for IVF training.
    """

    def __init__(
        self,
        dimension: int,
        nprobe: int = 8,
        rebuild_threshold: int = 500,
    ) -> None:
        self._dimension = dimension
        self._nprobe = nprobe
        self._rebuild_threshold = rebuild_threshold

        self._index: Any = None  # faiss.Index, lazily typed
        self._id_to_memory: list[str] = []  # FAISS int ID -> memory_id
        self._is_ivf = False
        self._adds_since_build = 0

    def build(self, chunks: list[tuple[str, np.ndarray]]) -> None:
        """Train and populate the index from all chunk embeddings.

        Args:
            chunks: List of (memory_id, vector) pairs.
        """
        if not chunks:
            self._index = None
            self._id_to_memory = []
            self._adds_since_build = 0
            return

        faiss = _import_faiss()

        memory_ids = [mid for mid, _ in chunks]
        matrix = np.stack([vec for _, vec in chunks]).astype(np.float32)
        n = len(matrix)

        # Choose index type based on corpus size.
        # FAISS needs ~39 training points per centroid for quality clustering.
        # Cap nlist so we always have enough, and fall back to flat when
        # the corpus is too small for IVF to help.
        min_for_ivf = 39 * 2  # Need at least 2 centroids worth of data
        ideal_nlist = max(1, int(4 * math.sqrt(n)))
        nlist = min(ideal_nlist, n // 39) if n >= min_for_ivf else 0

        if nlist < 2:
            # Too few vectors for meaningful IVF training — use flat index
            index = faiss.IndexFlatIP(self._dimension)
            index.add(matrix)
            self._is_ivf = False
        else:
            quantizer = faiss.IndexFlatIP(self._dimension)
            index = faiss.IndexIVFFlat(
                quantizer, self._dimension, nlist, faiss.METRIC_INNER_PRODUCT,
            )
            index.train(matrix)
            index.add(matrix)
            index.nprobe = self._nprobe
            self._is_ivf = True

        self._index = index
        self._id_to_memory = memory_ids
        self._adds_since_build = 0

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Search for the most similar chunks, deduplicated by memory.

        Returns:
            List of (memory_id, score) pairs sorted by descending score,
            with at most ``top_k`` unique memories.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        # Over-fetch to account for chunk deduplication
        fetch_k = min(top_k * 3, self._index.ntotal)
        q = query.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(q, fetch_k)

        # Deduplicate: keep best chunk score per memory
        best: dict[str, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                break  # FAISS uses -1 for unfilled slots
            score_f = float(score)
            if score_f < threshold:
                continue
            mid = self._id_to_memory[idx]
            if mid not in best or score_f > best[mid]:
                best[mid] = score_f

        sorted_results = sorted(best.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def add(self, memory_id: str, vectors: list[np.ndarray]) -> None:
        """Incrementally add vectors for a new memory."""
        if self._index is None:
            return
        for vec in vectors:
            v = vec.reshape(1, -1).astype(np.float32)
            self._index.add(v)
            self._id_to_memory.append(memory_id)
            self._adds_since_build += 1

    @property
    def needs_rebuild(self) -> bool:
        """True if enough incremental adds have happened to warrant retraining."""
        return self._adds_since_build >= self._rebuild_threshold

    @property
    def is_built(self) -> bool:
        return self._index is not None
