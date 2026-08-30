"""Staleness and invalidation for the on-disk FAISS index cache.

Every bug these cover had the same shape: the cache remained *accepted*
while no longer describing the database. That is worse than a slow rebuild,
because a restart corrects a slow rebuild and entrenches a wrong cache.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer1.embedding_engine import EmbeddingEngine

pytest.importorskip("faiss", reason="faiss not installed")

DIM = 32


@pytest.fixture
def tmp_dir():
    d = Path(tempfile.mkdtemp(prefix="chicory-staleness-"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def engine(config, tmp_dir):
    """An EmbeddingEngine over a small on-disk corpus."""
    cfg = config.model_copy(update={"db_path": tmp_dir / "t.db"})
    db = DatabaseEngine(cfg)
    db.connect()
    apply_schema(db)

    rng = np.random.RandomState(7)
    for i in range(60):
        mid = f"mem-{i:03d}"
        db.execute(
            "INSERT INTO memories (id, content, source_model, is_archived, "
            "ingestion_tier) VALUES (?, ?, 'test', 0, 'critical')",
            (mid, f"content {i}"),
        )
        v = rng.randn(DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        db.execute(
            "INSERT INTO embeddings (memory_id, chunk_index, embedding, "
            "model_name, dimension) VALUES (?, 0, ?, ?, ?)",
            (mid, v.tobytes(), cfg.embedding_model, DIM),
        )
    db.connection.commit()

    eng = EmbeddingEngine(cfg, db)
    yield eng
    db.close()


def _set_tier(engine, memory_id: str, tier: str) -> None:
    engine._db.execute(
        "UPDATE memories SET ingestion_tier = ? WHERE id = ?", (tier, memory_id)
    )
    engine._db.connection.commit()


def test_watermark_detects_offsetting_tier_flips(engine):
    """Releasing one memory while quarantining another leaves MAX and COUNT
    identical. Without a third term the cache is accepted, the released
    memory stays unsearchable and the quarantined one stays searchable —
    permanently, since a restart reloads the same cache."""
    # State A: one memory held back.
    _set_tier(engine, "mem-010", "quarantine")
    state_a = engine._embedding_watermark()

    # State B: that one released, a different one held back. Same number of
    # rows, same highest id — different membership.
    _set_tier(engine, "mem-010", "critical")
    _set_tier(engine, "mem-020", "quarantine")
    state_b = engine._embedding_watermark()

    assert state_a[0] == state_b[0] and state_a[1] == state_b[1], (
        "precondition: MAX and COUNT are blind to this swap"
    )
    assert state_a != state_b, (
        "watermark missed an offsetting tier swap — the cache would be "
        "accepted while index membership differs"
    )


def test_watermark_detects_release_from_quarantine(engine):
    """A tier change touches no embeddings row, so a key over that table
    alone would not move."""
    _set_tier(engine, "mem-005", "quarantine")
    quarantined = engine._embedding_watermark()
    _set_tier(engine, "mem-005", "critical")
    assert engine._embedding_watermark() != quarantined


def test_clear_memory_cache_bumps_epoch(engine):
    before = engine._index_epoch
    engine._clear_memory_cache()
    assert engine._index_epoch == before + 1


def test_invalidation_during_build_is_not_overwritten(engine, monkeypatch):
    """A builder that finishes after an archive must not reinstate the index
    it snapshotted beforehand."""
    real = engine._load_or_build_index

    def build_then_invalidate():
        vi = real()
        engine._clear_memory_cache()      # as an archive would, mid-build
        return vi

    monkeypatch.setattr(engine, "_load_or_build_index", build_then_invalidate)

    q = np.zeros(DIM, dtype=np.float32)
    q[0] = 1.0
    engine.search_similar(q, top_k=5)

    assert engine._vector_index is None, (
        "builder republished an index that was invalidated while it built"
    )


def test_cache_not_written_when_corpus_moves_mid_build(engine, monkeypatch):
    """If the corpus changes between snapshot and save, no cache is written
    rather than one stamped with coverage it does not have."""
    real_matrix = engine._load_chunk_matrix

    def matrix_then_change():
        out = real_matrix()
        _set_tier(engine, "mem-030", "quarantine")   # after the snapshot
        return out

    monkeypatch.setattr(engine, "_load_chunk_matrix", matrix_then_change)

    q = np.zeros(DIM, dtype=np.float32)
    q[0] = 1.0
    engine.search_similar(q, top_k=5)

    assert not engine._index_cache_path().exists(), (
        "cached an index whose watermark does not match what it indexed"
    )


def test_saved_cache_is_actually_reloaded(engine):
    """End-to-end: a cache written on one pass must be used on the next.

    Regression for a guard that tested the base path for existence. Under
    the pointer layout nothing lives at that path, so the check rejected
    every valid cache — silently turning the feature off while every
    correctness test still passed.
    """
    q = np.zeros(DIM, dtype=np.float32)
    q[0] = 1.0
    first = engine.search_similar(q, top_k=5)
    assert engine._index_cache_path().parent.exists()

    engine._vector_index = None          # as a fresh process would start
    reloaded = engine._try_load_cached_index()
    assert reloaded is not None, "saved cache was not reloaded"
    assert reloaded.search(q, 5) == first, "reloaded index answers differently"


def test_cache_survives_repeated_save_load_cycles(engine):
    """Publishing must be idempotent across generations, leaving one set."""
    q = np.zeros(DIM, dtype=np.float32)
    q[0] = 1.0
    for _ in range(3):
        engine._vector_index = None
        engine.search_similar(q, top_k=5)

    base = engine._index_cache_path()
    sets = [d for d in base.parent.iterdir()
            if d.is_dir() and d.name.startswith(base.name + ".")]
    assert len(sets) == 1, f"cache sets accumulated: {[s.name for s in sets]}"
