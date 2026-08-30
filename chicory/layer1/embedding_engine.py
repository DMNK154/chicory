"""Embedding generation and caching using sentence-transformers."""

from __future__ import annotations

import logging
import struct
import threading
import time
from typing import Optional

import numpy as np

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.exceptions import EmbeddingError
from chicory.ingest.chunker import chunk_text_for_embedding

_log = logging.getLogger(__name__)


class EmbeddingEngine:
    """Generates and caches embeddings. Model loaded lazily on first use."""

    def __init__(self, config: ChicoryConfig, db: DatabaseEngine) -> None:
        self._config = config
        self._db = db
        self._model = None
        self._tokenizer = None
        # In-memory embedding caches — loaded lazily, invalidated on writes
        self._cache_all: dict[str, np.ndarray] | None = None
        self._cache_chunks: list[tuple[str, np.ndarray]] | None = None
        # FAISS vector index — built lazily on first search
        self._vector_index = None  # VectorIndex | None
        self._suppress_invalidation = False
        # Serialize the expensive one-time loads so a background warmup
        # thread and an early tool call can't double-load or race the build
        self._model_lock = threading.Lock()
        self._index_lock = threading.Lock()
        self._index_wait_seconds = float(
            getattr(config, "faiss_index_wait_seconds", 120.0)
        )
        # Bumped by _clear_memory_cache(). A builder captures it before
        # starting and publishes only if it has not moved: otherwise an
        # invalidation arriving mid-build (archive, quarantine, release) is
        # silently overwritten by the finishing builder, and the archived
        # memory keeps being returned — across restarts, once cached.
        self._index_epoch = 0

    def batch_write(self):
        """Context manager: suppress cache invalidation until the batch completes."""
        return _BatchWriteContext(self)

    def _load_model(self) -> None:
        """Lazily load the sentence-transformer model (thread-safe)."""
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._config.embedding_model)
            except Exception as e:
                raise EmbeddingError(f"Failed to load embedding model: {e}") from e

    def embed(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text."""
        self._load_model()
        try:
            vec = self._model.encode(text, normalize_embeddings=True)
            return np.asarray(vec, dtype=np.float32)
        except Exception as e:
            raise EmbeddingError(f"Embedding failed: {e}") from e

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts."""
        self._load_model()
        try:
            vecs = self._model.encode(texts, normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)
        except Exception as e:
            raise EmbeddingError(f"Batch embedding failed: {e}") from e

    def get_cached(self, memory_id: str) -> Optional[np.ndarray]:
        """Retrieve the first cached embedding for a memory, or None."""
        row = self._db.execute(
            "SELECT embedding, dimension FROM embeddings WHERE memory_id = ? ORDER BY chunk_index LIMIT 1",
            (memory_id,),
        ).fetchone()
        if not row:
            return None
        return _blob_to_array(row["embedding"], row["dimension"])

    def search_similar(
        self,
        query_vec: np.ndarray,
        top_k: int,
        threshold: float | None = None,
    ) -> list[tuple[str, float]]:
        """Search for similar chunks using the FAISS index.

        Builds the index lazily on first call.  Returns deduplicated
        (memory_id, score) pairs sorted by descending score.
        """
        thresh = threshold if threshold is not None else 0.0

        # Snapshot the reference once. _clear_memory_cache() sets this
        # attribute to None from other threads without any lock, and the
        # previous version dereferenced it AFTER releasing the lock — an
        # AttributeError waiting for the right interleaving.
        idx = self._vector_index
        if idx is not None and not idx.needs_rebuild:
            return idx.search(query_vec, top_k, threshold=thresh)

        # A usable index in hand always beats waiting. Blocking here is how
        # a background rebuild turned into a 57.4s retrieve_memories, and
        # returning [] instead would be indistinguishable from "no matches".
        # Take the lock only if it is free; otherwise serve what we have.
        blocking = idx is None
        acquired = self._index_lock.acquire(
            blocking=blocking,
            **({"timeout": self._index_wait_seconds} if blocking else {}),
        )
        if not acquired:
            if idx is not None:
                return idx.search(query_vec, top_k, threshold=thresh)
            _log.error(
                "No index and none could be built within %.0fs — returning "
                "no semantic hits", self._index_wait_seconds,
            )
            return []

        try:
            current = self._vector_index
            if current is not None and not current.needs_rebuild:
                return current.search(query_vec, top_k, threshold=thresh)

            epoch = self._index_epoch
            vi = self._load_or_build_index()
            if vi is None:
                return idx.search(query_vec, top_k, threshold=thresh) if idx else []

            if epoch != self._index_epoch:
                # Something was archived, quarantined or released while we
                # built. Publishing now would resurrect it and, once saved,
                # keep resurrecting it after every restart.
                _log.info("Index invalidated during build — discarding result")
                return []

            self._vector_index = vi
            return vi.search(query_vec, top_k, threshold=thresh)
        finally:
            self._index_lock.release()

    def _load_or_build_index(self):
        """Load the cached index, or build one. Caller must hold the lock."""
        from chicory.layer1.vector_index import VectorIndex

        cached = self._try_load_cached_index()
        if cached is not None:
            return cached

        # Read the watermark on both sides of the build and only cache the
        # result if the corpus did not move. Stamping a watermark read
        # AFTER the snapshot claims coverage of rows that were never
        # indexed, and that lie then survives every restart.
        wm_before = self._embedding_watermark()
        mem_ids, matrix = self._load_chunk_matrix()
        mem_gb = matrix.nbytes / (1024 ** 3) if matrix.size else 0
        _log.info(
            "Building FAISS index: %d chunks, %d dim, %.2f GB",
            len(mem_ids), self._config.embedding_dimension, mem_gb,
        )
        vi = VectorIndex(
            dimension=self._config.embedding_dimension,
            nprobe=self._config.faiss_nprobe,
            rebuild_threshold=self._config.faiss_rebuild_threshold,
        )
        try:
            vi.build_from_matrix(mem_ids, matrix)
        except MemoryError:
            _log.error(
                "FAISS index build failed: unable to allocate %.2f GB "
                "for %d chunks. Reduce corpus size or add RAM.",
                mem_gb, len(mem_ids),
            )
            raise

        if self._embedding_watermark() == wm_before:
            self._save_cached_index(vi, wm_before)
        else:
            _log.info("Corpus changed during build — index built but not cached")
        return vi

    # ── On-disk index cache ─────────────────────────────────────────

    def _index_cache_path(self):
        from pathlib import Path
        db = Path(self._config.db_path)
        return db.with_suffix(db.suffix + ".faiss")

    def _embedding_watermark(self) -> tuple[int, int]:
        """(max id, row count) over exactly the rows the index is built from.

        `embeddings.id` is INTEGER PRIMARY KEY AUTOINCREMENT, so it is
        monotonic and never reused; INSERT OR REPLACE gets a fresh higher
        id. Pairing it with a count also catches deletes, which lower the
        count without moving the maximum.

        The JOIN is load-bearing, not decoration. A tier change — releasing
        a memory from quarantine — touches no row in `embeddings`, so a key
        taken over that table alone would match while the cached index still
        excluded the released memory, making a transient bug permanent
        across restarts. Keying on the same query `_load_chunk_matrix` uses
        costs ~0.8s against a 39-57s rebuild.

        SUM is here because MAX and COUNT together are blind to offsetting
        changes: release one memory and quarantine another of equal chunk
        count, and both are unchanged while the membership is different.
        SUM(e.id) breaks that symmetry for the price of the same scan.
        """
        row = self._db.execute(
            """
            SELECT COALESCE(MAX(e.id), 0) AS mx, COUNT(*) AS n,
                   COALESCE(SUM(e.id), 0) AS s
            FROM embeddings e JOIN memories m ON m.id = e.memory_id
            WHERE m.is_archived = 0 AND m.ingestion_tier != 'quarantine'
            """
        ).fetchone()
        return int(row["mx"]), int(row["n"]), int(row["s"])

    def _try_load_cached_index(self):
        """Return a VectorIndex from disk if it is exactly current, else None.

        Deliberately all-or-nothing: any doubt falls through to a full
        rebuild, which is the behaviour that existed before this cache.
        """
        if not getattr(self._config, "faiss_index_cache_enabled", True):
            return None
        # No existence check here: the cache is published through a pointer
        # file beside this path, not at it, so testing the base path would
        # reject every valid cache. load() resolves the pointer.
        path = self._index_cache_path()

        from chicory.layer1.vector_index import VectorIndex

        vi = VectorIndex(
            dimension=self._config.embedding_dimension,
            nprobe=self._config.faiss_nprobe,
            rebuild_threshold=self._config.faiss_rebuild_threshold,
        )
        t = time.time()
        meta = vi.load(path)
        if meta is None:
            return None

        saved = meta.get("watermark")
        current = list(self._embedding_watermark())
        if saved is None or list(saved) != current:
            _log.info(
                "Cached index stale (saved %s, now %s) — rebuilding",
                saved, current,
            )
            return None
        if meta.get("model_name") != self._config.embedding_model:
            _log.info("Cached index rejected: built with a different model")
            return None

        _log.info(
            "FAISS index loaded from cache in %.2fs (%d vectors)",
            time.time() - t, vi.ntotal,
        )
        return vi

    def _save_cached_index(self, index, watermark) -> None:
        """Persist `index`, stamped with the watermark it was built from.

        The watermark is passed in rather than re-read: reading it here
        would describe the corpus as of the save, not as of the snapshot
        the index was actually built from.
        """
        if not getattr(self._config, "faiss_index_cache_enabled", True):
            return
        if index is None:
            return
        try:
            t = time.time()
            index.save(
                self._index_cache_path(),
                {
                    "watermark": list(watermark),
                    "count": int(watermark[1]),
                    "model_name": self._config.embedding_model,
                    "nprobe": self._config.faiss_nprobe,
                },
            )
            _log.info("FAISS index cached in %.2fs", time.time() - t)
        except Exception:
            # A cache that cannot be written must never break retrieval —
            # but it must also never leave a half-written set on disk, which
            # save() handles by removing the whole set on failure.
            _log.exception("Failed to write FAISS index cache")

    def store_cached(self, memory_id: str, embedding: np.ndarray, chunk_index: int = 0) -> None:
        """Store a single chunk embedding in the database."""
        blob = _array_to_blob(embedding)
        self._db.execute(
            """
            INSERT OR REPLACE INTO embeddings (memory_id, chunk_index, embedding, model_name, dimension)
            VALUES (?, ?, ?, ?, ?)
            """,
            (memory_id, chunk_index, blob, self._config.embedding_model, len(embedding)),
        )
        if not self._suppress_invalidation:
            self._db.connection.commit()
            if self._vector_index is not None and self._vector_index.is_built:
                self._vector_index.add(memory_id, [embedding])
            self._cache_all = None
            self._cache_chunks = None

    def store_chunks(self, memory_id: str, embeddings: list[np.ndarray]) -> None:
        """Store multiple chunk embeddings for a memory, replacing any existing ones."""
        self._db.execute(
            "DELETE FROM embeddings WHERE memory_id = ?", (memory_id,)
        )
        for i, vec in enumerate(embeddings):
            blob = _array_to_blob(vec)
            self._db.execute(
                """
                INSERT INTO embeddings (memory_id, chunk_index, embedding, model_name, dimension)
                VALUES (?, ?, ?, ?, ?)
                """,
                (memory_id, i, blob, self._config.embedding_model, len(vec)),
            )
        if not self._suppress_invalidation:
            self._db.connection.commit()
            self._clear_memory_cache()

    def invalidate(self, memory_id: str) -> None:
        """Remove all cached embeddings for a memory from the database."""
        self._db.execute(
            "DELETE FROM embeddings WHERE memory_id = ?", (memory_id,)
        )
        self._db.connection.commit()
        self._clear_memory_cache()

    def _clear_memory_cache(self) -> None:
        """Clear the in-memory caches so the next read reloads from SQLite.

        This does NOT delete any data from the database — it only resets the
        Python-side variables so fresh data is loaded on the next retrieval.

        The epoch bump is what makes this stick. Without it a builder that
        started before this call finishes afterwards and reinstates the
        index it had already snapshotted, so an archive or quarantine
        arriving mid-build is silently lost.
        """
        self._cache_all = None
        self._cache_chunks = None
        self._vector_index = None
        self._index_epoch += 1

    def get_all_cached(self) -> dict[str, np.ndarray]:
        """Load first-chunk embeddings per memory. Returns {memory_id: vector}.

        Used by synchronicity detector and other consumers that need one
        representative vector per memory.  Results are kept in memory and
        reused until a write operation invalidates the cache.
        """
        if self._cache_all is not None:
            return self._cache_all

        rows = self._db.execute(
            """
            SELECT e.memory_id, e.embedding, e.dimension FROM embeddings e
            JOIN memories m ON m.id = e.memory_id
            WHERE e.chunk_index = 0 AND m.ingestion_tier != 'quarantine'
            """
        ).fetchall()
        if not rows:
            self._cache_all = {}
            return self._cache_all

        dim = rows[0]["dimension"]
        # Single deserialization: concat all blobs, one frombuffer + reshape
        all_bytes = b"".join(r["embedding"] for r in rows)
        matrix = np.frombuffer(all_bytes, dtype=np.float32).reshape(-1, dim).copy()
        self._cache_all = {r["memory_id"]: matrix[i] for i, r in enumerate(rows)}
        return self._cache_all

    def _load_chunk_matrix(self) -> tuple[list[str], np.ndarray]:
        """Load all chunk embeddings as a contiguous matrix.

        Returns (memory_ids, matrix) where matrix is (N, dim) float32.
        One entry per chunk — memory_ids may contain duplicates.
        Archived and quarantined memories are excluded so they never
        enter the FAISS index.
        """
        rows = self._db.execute(
            """
            SELECT e.memory_id, e.embedding, e.dimension FROM embeddings e
            JOIN memories m ON m.id = e.memory_id
            WHERE m.is_archived = 0 AND m.ingestion_tier != 'quarantine'
            ORDER BY e.memory_id, e.chunk_index
            """
        ).fetchall()
        if not rows:
            return [], np.array([], dtype=np.float32)
        dim = rows[0]["dimension"]
        mem_ids = [r["memory_id"] for r in rows]
        all_bytes = b"".join(r["embedding"] for r in rows)
        matrix = np.frombuffer(all_bytes, dtype=np.float32).reshape(-1, dim).copy()
        return mem_ids, matrix

    def get_all_chunk_embeddings(self) -> list[tuple[str, np.ndarray]]:
        """Load all chunk embeddings. Returns [(memory_id, vector), ...].

        Multiple entries may share the same memory_id (one per chunk).
        Used by semantic retrieval for fine-grained matching.  Results are
        kept in memory and reused until a write operation invalidates the cache.
        """
        if self._cache_chunks is not None:
            return self._cache_chunks

        rows = self._db.execute(
            """
            SELECT e.memory_id, e.embedding, e.dimension FROM embeddings e
            JOIN memories m ON m.id = e.memory_id
            WHERE m.is_archived = 0 AND m.ingestion_tier != 'quarantine'
            ORDER BY e.memory_id, e.chunk_index
            """
        ).fetchall()
        if not rows:
            self._cache_chunks = []
            return self._cache_chunks

        dim = rows[0]["dimension"]
        # Single deserialization: concat all blobs, one frombuffer + reshape
        # Matrix rows are views into the contiguous array — no per-row copy
        all_bytes = b"".join(r["embedding"] for r in rows)
        matrix = np.frombuffer(all_bytes, dtype=np.float32).reshape(-1, dim).copy()
        self._cache_chunks = [(r["memory_id"], matrix[i]) for i, r in enumerate(rows)]
        return self._cache_chunks

    def reembed_all(self, new_model: str | None = None) -> int:
        """Re-embed all memories with the current (or new) model. Returns count."""
        if new_model:
            self._config.embedding_model = new_model
            self._model = None  # Force reload

        rows = self._db.execute(
            "SELECT id, content FROM memories WHERE is_archived = 0"
        ).fetchall()

        if not rows:
            return 0

        with self.batch_write():
            for row in rows:
                chunks = chunk_text_for_embedding(row["content"])
                if not chunks:
                    continue
                vecs = self.embed_batch(chunks) if len(chunks) > 1 else [self.embed(chunks[0])]
                self.store_chunks(row["id"], list(vecs))

        return len(rows)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors (assumed normalized)."""
        return float(np.dot(a, b))

    @staticmethod
    def bulk_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Cosine similarity of query against all candidates (assumed normalized)."""
        return candidates @ query


class _BatchWriteContext:
    """Suppresses per-write cache invalidation; flushes once on exit."""

    def __init__(self, engine: EmbeddingEngine) -> None:
        self._engine = engine

    def __enter__(self) -> EmbeddingEngine:
        self._engine._suppress_invalidation = True
        return self._engine

    def __exit__(self, *exc) -> None:
        self._engine._suppress_invalidation = False
        self._engine._db.connection.commit()
        self._engine._clear_memory_cache()


def _array_to_blob(arr: np.ndarray) -> bytes:
    """Serialize a float32 numpy array to bytes."""
    return arr.astype(np.float32).tobytes()


def _blob_to_array(blob: bytes, dimension: int) -> np.ndarray:
    """Deserialize bytes to a float32 numpy array."""
    return np.frombuffer(blob, dtype=np.float32).copy()
