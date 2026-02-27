"""Embedding generation and caching using sentence-transformers."""

from __future__ import annotations

import struct
from typing import Optional

import numpy as np

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.exceptions import EmbeddingError
from chicory.ingest.chunker import chunk_text_for_embedding


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

    def _load_model(self) -> None:
        """Lazily load the sentence-transformer model."""
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
        if self._vector_index is None or self._vector_index.needs_rebuild:
            from chicory.layer1.vector_index import VectorIndex

            chunks = self.get_all_chunk_embeddings()
            vi = VectorIndex(
                dimension=self._config.embedding_dimension,
                nprobe=self._config.faiss_nprobe,
                rebuild_threshold=self._config.faiss_rebuild_threshold,
            )
            vi.build(chunks)
            self._vector_index = vi

        thresh = threshold if threshold is not None else 0.0
        return self._vector_index.search(query_vec, top_k, threshold=thresh)

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
        self._db.connection.commit()
        # Incrementally add to FAISS index if it exists; otherwise clear cache
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
        self._db.connection.commit()
        # Replacement invalidates the index — IVF can't remove old entries
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
        """
        self._cache_all = None
        self._cache_chunks = None
        self._vector_index = None

    def get_all_cached(self) -> dict[str, np.ndarray]:
        """Load first-chunk embeddings per memory. Returns {memory_id: vector}.

        Used by synchronicity detector and other consumers that need one
        representative vector per memory.  Results are kept in memory and
        reused until a write operation invalidates the cache.
        """
        if self._cache_all is not None:
            return self._cache_all

        rows = self._db.execute(
            "SELECT memory_id, embedding, dimension FROM embeddings WHERE chunk_index = 0"
        ).fetchall()
        self._cache_all = {
            r["memory_id"]: _blob_to_array(r["embedding"], r["dimension"])
            for r in rows
        }
        return self._cache_all

    def get_all_chunk_embeddings(self) -> list[tuple[str, np.ndarray]]:
        """Load all chunk embeddings. Returns [(memory_id, vector), ...].

        Multiple entries may share the same memory_id (one per chunk).
        Used by semantic retrieval for fine-grained matching.  Results are
        kept in memory and reused until a write operation invalidates the cache.
        """
        if self._cache_chunks is not None:
            return self._cache_chunks

        rows = self._db.execute(
            "SELECT memory_id, embedding, dimension FROM embeddings ORDER BY memory_id, chunk_index"
        ).fetchall()
        self._cache_chunks = [
            (r["memory_id"], _blob_to_array(r["embedding"], r["dimension"]))
            for r in rows
        ]
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


def _array_to_blob(arr: np.ndarray) -> bytes:
    """Serialize a float32 numpy array to bytes."""
    return arr.astype(np.float32).tobytes()


def _blob_to_array(blob: bytes, dimension: int) -> np.ndarray:
    """Deserialize bytes to a float32 numpy array."""
    return np.frombuffer(blob, dtype=np.float32).copy()
