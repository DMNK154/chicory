"""FAISS-backed vector index for approximate nearest-neighbor search."""

from __future__ import annotations

import json
import logging
import math
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np

_log = logging.getLogger(__name__)


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

        memory_ids = [mid for mid, _ in chunks]
        matrix = np.stack([vec for _, vec in chunks]).astype(np.float32)
        self.build_from_matrix(memory_ids, matrix)

    def build_from_matrix(self, memory_ids: list[str], matrix: np.ndarray) -> None:
        """Train and populate the index from a pre-built matrix.

        Args:
            memory_ids: Parallel list of memory IDs (one per row).
            matrix: (N, dim) float32 contiguous array.
        """
        if len(memory_ids) == 0:
            self._index = None
            self._id_to_memory = []
            self._adds_since_build = 0
            return

        faiss = _import_faiss()

        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
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
        """Incrementally add vectors for a new memory.

        One faiss call for the whole batch: per-vector `add()` measured
        10.4ms each on a 351k IVF index (500 vectors = 5.2s), against
        0.006s for the same 500 added together.

        Note this appends and never removes, so the positional id map stays
        valid.  If removal is ever added, this must switch to
        `add_with_ids` with a persisted counter — after a `remove_ids`,
        plain `add()` reuses ids and silently collides with live vectors.
        """
        if self._index is None or not vectors:
            return
        batch = np.ascontiguousarray(np.stack(vectors), dtype=np.float32)
        self._index.add(batch)
        self._id_to_memory.extend([memory_id] * len(batch))
        self._adds_since_build += len(batch)

    # ── Persistence ─────────────────────────────────────────────────
    #
    # Rebuilding from the embeddings table costs 39-57s on a 351k-chunk
    # corpus (SQL scan + IVF k-means training), and it ran on every server
    # start.  A written index reads back in well under a second and searches
    # identically.  `meta` carries whatever the caller needs to decide
    # staleness; this class stores it verbatim and does not interpret it.

    def save(self, path: Path, meta: dict) -> None:
        """Atomically publish the index, its id map, and caller metadata.

        The set is written into a directory named for a fresh nonce, and
        only then is a small pointer file swapped to name it. That pointer
        rename is the single commit point.

        Three independent renames cannot be made safe by cross-checking
        them. The dangerous interleaving is a NEW index beside an OLD id
        map and an OLD meta: those two agree with each other, so any
        nonce shared between them still validates, and every query returns
        real vectors under the wrong memory ids -- confident, plausible,
        wrong. Only one commit point removes that state.
        """
        if self._index is None:
            return
        faiss = _import_faiss()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nonce = uuid.uuid4().hex
        set_dir = path.parent / f"{path.name}.{nonce}"

        try:
            set_dir.mkdir(parents=True, exist_ok=False)
            faiss.write_index(self._index, str(set_dir / "index.faiss"))

            # Store the id map compactly: 351k x 36-char ids are 50MB raw
            # but ~19MB as unique-plus-inverse, and ids repeat per chunk.
            uniq, inv = np.unique(
                np.asarray(self._id_to_memory), return_inverse=True,
            )
            # Hand savez an open handle: given a *path* it appends ".npz"
            # to anything not already ending in it, writing a different
            # file than the one you named.
            with open(set_dir / "ids.npz", "wb") as fh:
                np.savez(fh, uniq=uniq, inv=inv.astype(np.int32))

            payload = dict(meta)
            payload.update({
                "dimension": self._dimension,
                "is_ivf": self._is_ivf,
                "ntotal": int(self._index.ntotal),
                "id_map_len": len(self._id_to_memory),
                "nonce": nonce,
            })
            (set_dir / "meta.json").write_text(
                json.dumps(payload), encoding="utf-8",
            )

            # Commit. Everything above is invisible until this line lands,
            # and a crash before it leaves the previous set intact.
            ptr_tmp = path.parent / f"{path.name}.current.tmp"
            ptr_tmp.write_text(nonce, encoding="utf-8")
            ptr_tmp.replace(self._pointer_path(path))
        except Exception:
            _log.exception("Index cache save failed; discarding partial set")
            shutil.rmtree(set_dir, ignore_errors=True)
            raise

        self._prune_old_sets(path, keep=nonce)

    @staticmethod
    def _pointer_path(path: Path) -> Path:
        return path.parent / f"{path.name}.current"

    @staticmethod
    def _prune_old_sets(path: Path, keep: str | None = None) -> None:
        """Remove superseded set directories and any legacy flat sidecars."""
        prefix = f"{path.name}."
        try:
            entries = list(path.parent.iterdir())
        except OSError:
            return
        for entry in entries:
            name = entry.name
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix):]
            if entry.is_dir():
                if keep is None or suffix != keep:
                    shutil.rmtree(entry, ignore_errors=True)
            elif suffix in ("ids.npz", "meta.json", "tmp",
                            "ids.npz.tmp", "meta.json.tmp"):
                # Sidecars from the pre-pointer layout.
                try:
                    entry.unlink(missing_ok=True)
                except OSError:
                    pass
        if keep is None:
            for p in (VectorIndex._pointer_path(path), path):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass

    def discard_cache(self, path: Path) -> None:
        """Remove the pointer and every set, leaving nothing loadable."""
        path = Path(path)
        try:
            VectorIndex._pointer_path(path).unlink(missing_ok=True)
        except OSError:
            pass
        self._prune_old_sets(path, keep=None)

    def load(self, path: Path) -> Optional[dict]:
        """Load the committed set. Returns its metadata, or None if unusable.

        Any inconsistency returns None so the caller rebuilds -- the worst
        case is exactly the behaviour that existed before this cache.
        """
        path = Path(path)
        ptr = self._pointer_path(path)
        if not ptr.exists():
            return None

        try:
            nonce = ptr.read_text(encoding="utf-8").strip()
            set_dir = path.parent / f"{path.name}.{nonce}"
            index_path = set_dir / "index.faiss"
            ids_path = set_dir / "ids.npz"
            meta_path = set_dir / "meta.json"
            if not (index_path.exists() and ids_path.exists()
                    and meta_path.exists()):
                _log.info("Cached index rejected: set %s incomplete", nonce)
                return None

            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("nonce") != nonce:
                _log.warning("Cached index rejected: pointer/meta nonce differ")
                return None
            if int(meta.get("dimension", -1)) != self._dimension:
                _log.info("Cached index rejected: dimension mismatch")
                return None

            # The caller's own row count must equal the vectors actually in
            # the index. A build that snapshots the corpus at T0 and stamps
            # a watermark read at T1 yields count=N+1 beside ntotal=N, and
            # that cache claims to cover a memory it never indexed.
            m_count, m_ntotal = meta.get("count"), meta.get("ntotal")
            if (m_count is not None and m_ntotal is not None
                    and m_count != m_ntotal):
                _log.warning(
                    "Cached index rejected: meta count %s != ntotal %s "
                    "(corpus changed mid-build)", m_count, m_ntotal,
                )
                return None

            faiss = _import_faiss()
            index = faiss.read_index(str(index_path))
            with np.load(ids_path, allow_pickle=False) as z:
                id_map = z["uniq"][z["inv"]].tolist()

            if index.ntotal != len(id_map):
                _log.warning(
                    "Cached index rejected: ntotal %d != id map %d",
                    index.ntotal, len(id_map),
                )
                return None

            self._index = index
            self._id_to_memory = id_map
            self._is_ivf = bool(meta.get("is_ivf", False))
            self._adds_since_build = 0
            if self._is_ivf:
                self._index.nprobe = self._nprobe
            return meta
        except Exception:
            _log.exception("Cached index load failed; will rebuild")
            return None

    @property
    def needs_rebuild(self) -> bool:
        """True if enough incremental adds have happened to warrant retraining.

        Proportional, not absolute: a flat threshold of 500 on a 351k-chunk
        corpus retrains the whole index for a 0.14% change.
        """
        if self._index is None:
            return False
        base = self._index.ntotal - self._adds_since_build
        if base <= 0:
            return self._adds_since_build >= self._rebuild_threshold
        return (
            self._adds_since_build >= self._rebuild_threshold
            and self._adds_since_build >= 0.25 * base
        )

    @property
    def ntotal(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)

    @property
    def is_built(self) -> bool:
        return self._index is not None
