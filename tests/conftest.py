"""Shared test fixtures."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema


@pytest.fixture
def config():
    """Test configuration with in-memory database."""
    return ChicoryConfig(
        db_path=Path(":memory:"),
        anthropic_api_key="test-key",
        llm_model="test-model",
        embedding_model="test-model",
        embedding_dimension=32,  # Small for testing
    )


@pytest.fixture
def db(config):
    """In-memory database with schema applied."""
    engine = DatabaseEngine(config)
    engine.connect()
    apply_schema(engine)
    yield engine
    engine.close()


class MockEmbeddingEngine:
    """Deterministic mock embedding engine for testing.
    Produces reproducible vectors by hashing text content."""

    def __init__(self, dimension: int = 32):
        self.dimension = dimension
        self._db = None

    def set_db(self, db):
        self._db = db

    def embed(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.RandomState(int.from_bytes(h[:4], "big"))
        vec = rng.randn(self.dimension).astype(np.float32)
        return vec / np.linalg.norm(vec)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])

    def get_cached(self, memory_id: str):
        if self._db is None:
            return None
        row = self._db.execute(
            "SELECT embedding, dimension FROM embeddings WHERE memory_id = ? ORDER BY chunk_index LIMIT 1",
            (memory_id,),
        ).fetchone()
        if not row:
            return None
        return np.frombuffer(row["embedding"], dtype=np.float32).copy()

    def store_cached(self, memory_id: str, embedding: np.ndarray, chunk_index: int = 0) -> None:
        if self._db is None:
            return
        blob = embedding.astype(np.float32).tobytes()
        self._db.execute(
            """
            INSERT OR REPLACE INTO embeddings (memory_id, chunk_index, embedding, model_name, dimension)
            VALUES (?, ?, ?, 'mock', ?)
            """,
            (memory_id, chunk_index, blob, len(embedding)),
        )
        self._db.connection.commit()

    def store_chunks(self, memory_id: str, embeddings: list[np.ndarray]) -> None:
        if self._db is None:
            return
        self._db.execute(
            "DELETE FROM embeddings WHERE memory_id = ?", (memory_id,)
        )
        for i, vec in enumerate(embeddings):
            blob = vec.astype(np.float32).tobytes()
            self._db.execute(
                """
                INSERT INTO embeddings (memory_id, chunk_index, embedding, model_name, dimension)
                VALUES (?, ?, ?, 'mock', ?)
                """,
                (memory_id, i, blob, len(vec)),
            )
        self._db.connection.commit()

    def get_all_cached(self) -> dict[str, np.ndarray]:
        if self._db is None:
            return {}
        rows = self._db.execute(
            "SELECT memory_id, embedding, dimension FROM embeddings WHERE chunk_index = 0"
        ).fetchall()
        return {
            r["memory_id"]: np.frombuffer(r["embedding"], dtype=np.float32).copy()
            for r in rows
        }

    def get_all_chunk_embeddings(self) -> list[tuple[str, np.ndarray]]:
        if self._db is None:
            return []
        rows = self._db.execute(
            "SELECT memory_id, embedding, dimension FROM embeddings ORDER BY memory_id, chunk_index"
        ).fetchall()
        return [
            (r["memory_id"], np.frombuffer(r["embedding"], dtype=np.float32).copy())
            for r in rows
        ]

    def search_similar(
        self,
        query_vec: np.ndarray,
        top_k: int,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """Brute-force search matching the FAISS-backed interface."""
        all_chunks = self.get_all_chunk_embeddings()
        if not all_chunks:
            return []
        mids = [mid for mid, _ in all_chunks]
        matrix = np.stack([vec for _, vec in all_chunks])
        sims = matrix @ query_vec
        best: dict[str, float] = {}
        for i, score in enumerate(sims):
            s = float(score)
            if s < threshold:
                continue
            mid = mids[i]
            if mid not in best or s > best[mid]:
                best[mid] = s
        sorted_results = sorted(best.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]

    def invalidate(self, memory_id: str) -> None:
        if self._db:
            self._db.execute(
                "DELETE FROM embeddings WHERE memory_id = ?", (memory_id,)
            )
            self._db.connection.commit()

    @staticmethod
    def cosine_similarity(a, b):
        return float(np.dot(a, b))

    @staticmethod
    def bulk_similarity(query, candidates):
        return candidates @ query


@pytest.fixture
def mock_embeddings():
    """Mock embedding engine with deterministic vectors."""
    return MockEmbeddingEngine(dimension=32)
