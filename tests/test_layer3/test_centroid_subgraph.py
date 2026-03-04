"""Tests for retrieval-driven centroid sub-graph inhibition."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer1.tag_manager import TagManager
from chicory.layer3.centroid_subgraph import (
    CentroidSubgraph,
    _array_to_blob,
    _blob_to_array,
)
from tests.conftest import MockEmbeddingEngine


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def stack():
    """Build a test stack with DB, mock embeddings, tag manager, and subgraph."""
    config = ChicoryConfig(
        db_path=Path(":memory:"),
        anthropic_api_key="test",
        llm_model="test-model",
        embedding_model="mock",
        embedding_dimension=32,
        centroid_ema_alpha=0.2,
        centroid_edge_ema_alpha=0.3,
        centroid_inhibition_scale=1.0,
        centroid_inhibition_enabled=True,
    )

    db = DatabaseEngine(config)
    db.connect()
    apply_schema(db)

    mock_emb = MockEmbeddingEngine(dimension=32)
    mock_emb.set_db(db)
    tag_mgr = TagManager(db)

    subgraph = CentroidSubgraph(config, db, mock_emb)

    yield {
        "config": config,
        "db": db,
        "emb": mock_emb,
        "tags": tag_mgr,
        "subgraph": subgraph,
    }

    db.close()


def _make_unit_vec(dim: int = 32, seed: int = 0) -> np.ndarray:
    """Create a deterministic unit vector."""
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _insert_tag(db, name: str) -> int:
    db.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (name,))
    return db.execute(
        "SELECT id FROM tags WHERE name = ?", (name,),
    ).fetchone()["id"]


def _insert_memory(db, memory_id: str, content: str = "test") -> None:
    db.execute(
        "INSERT OR IGNORE INTO memories (id, content, source_model) VALUES (?, ?, 'test')",
        (memory_id, content),
    )


def _assign_tag(db, memory_id: str, tag_id: int) -> None:
    db.execute(
        "INSERT OR IGNORE INTO memory_tags (memory_id, tag_id) VALUES (?, ?)",
        (memory_id, tag_id),
    )


# ── Centroid Maintenance ────────────────────────────────────────────────


class TestCentroidMaintenance:

    def test_first_store_initializes_centroid(self, stack):
        db, subgraph = stack["db"], stack["subgraph"]
        tag_id = _insert_tag(db, "alpha")
        vec = _make_unit_vec(seed=1)

        subgraph.update_centroid_on_store(tag_id, vec)

        centroids = subgraph.get_centroids_batch([tag_id])
        assert tag_id in centroids
        # First store: centroid should be the normalized input
        np.testing.assert_allclose(centroids[tag_id], vec, atol=1e-5)

    def test_ema_update_moves_centroid(self, stack):
        db, subgraph = stack["db"], stack["subgraph"]
        tag_id = _insert_tag(db, "beta")
        v1 = _make_unit_vec(seed=10)
        v2 = _make_unit_vec(seed=20)

        subgraph.update_centroid_on_store(tag_id, v1)
        subgraph.update_centroid_on_store(tag_id, v2)

        centroids = subgraph.get_centroids_batch([tag_id])
        result = centroids[tag_id]

        # EMA: alpha=0.2 → new = 0.2*v2 + 0.8*v1, then normalized
        expected = 0.2 * v2 + 0.8 * v1
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_memory_count_increments(self, stack):
        db, subgraph = stack["db"], stack["subgraph"]
        tag_id = _insert_tag(db, "gamma")
        for i in range(5):
            subgraph.update_centroid_on_store(tag_id, _make_unit_vec(seed=i))

        row = db.execute(
            "SELECT memory_count FROM tag_centroids WHERE tag_id = ?",
            (tag_id,),
        ).fetchone()
        assert row["memory_count"] == 5

    def test_get_centroids_batch_empty(self, stack):
        result = stack["subgraph"].get_centroids_batch([])
        assert result == {}

    def test_get_centroids_batch_missing_tags(self, stack):
        result = stack["subgraph"].get_centroids_batch([9999])
        assert result == {}

    def test_rebuild_centroids(self, stack):
        db, subgraph, emb = stack["db"], stack["subgraph"], stack["emb"]

        # Set up: 2 tags, 2 memories, each with 1 tag
        t1 = _insert_tag(db, "tag-one")
        t2 = _insert_tag(db, "tag-two")
        _insert_memory(db, "m1")
        _insert_memory(db, "m2")
        _assign_tag(db, "m1", t1)
        _assign_tag(db, "m2", t2)

        v1 = _make_unit_vec(seed=100)
        v2 = _make_unit_vec(seed=200)
        emb.store_cached("m1", v1)
        emb.store_cached("m2", v2)

        count = subgraph.rebuild_centroids()
        assert count == 2

        centroids = subgraph.get_centroids_batch([t1, t2])
        assert t1 in centroids
        assert t2 in centroids


# ── Co-Retrieval Edge Tracking ──────────────────────────────────────────


class TestCoRetrievalEdges:

    def test_single_tag_no_edges(self, stack):
        db, subgraph = stack["db"], stack["subgraph"]
        t1 = _insert_tag(db, "lone")
        subgraph.record_co_retrieval([t1])

        count = db.execute("SELECT COUNT(*) as c FROM centroid_edges").fetchone()["c"]
        assert count == 0

    def test_two_tags_creates_edge(self, stack):
        db, subgraph = stack["db"], stack["subgraph"]
        t1 = _insert_tag(db, "a")
        t2 = _insert_tag(db, "b")

        subgraph.record_co_retrieval([t1, t2])

        row = db.execute(
            "SELECT edge_strength, co_retrieval_count FROM centroid_edges "
            "WHERE tag_a_id = ? AND tag_b_id = ?",
            (min(t1, t2), max(t1, t2)),
        ).fetchone()
        assert row is not None
        # First retrieval: alpha * 1.0 + (1-alpha) * 0.0 = 0.3
        assert abs(row["edge_strength"] - 0.3) < 1e-6
        assert row["co_retrieval_count"] == 1

    def test_ema_accumulates(self, stack):
        db, subgraph = stack["db"], stack["subgraph"]
        t1 = _insert_tag(db, "x")
        t2 = _insert_tag(db, "y")

        subgraph.record_co_retrieval([t1, t2])  # 0.3
        subgraph.record_co_retrieval([t1, t2])  # 0.3*1.0 + 0.7*0.3 = 0.51

        row = db.execute(
            "SELECT edge_strength, co_retrieval_count FROM centroid_edges "
            "WHERE tag_a_id = ? AND tag_b_id = ?",
            (min(t1, t2), max(t1, t2)),
        ).fetchone()
        assert abs(row["edge_strength"] - 0.51) < 1e-6
        assert row["co_retrieval_count"] == 2

    def test_three_tags_creates_three_edges(self, stack):
        db, subgraph = stack["db"], stack["subgraph"]
        t1 = _insert_tag(db, "p")
        t2 = _insert_tag(db, "q")
        t3 = _insert_tag(db, "r")

        subgraph.record_co_retrieval([t1, t2, t3])

        count = db.execute("SELECT COUNT(*) as c FROM centroid_edges").fetchone()["c"]
        assert count == 3

    def test_rebuild_from_history(self, stack):
        db, subgraph = stack["db"], stack["subgraph"]
        t1 = _insert_tag(db, "h1")
        t2 = _insert_tag(db, "h2")

        # Simulate retrieval history
        db.execute(
            "INSERT INTO retrieval_events (query_text, method, result_count, model_version) "
            "VALUES ('test', 'hybrid', 1, 'v1')"
        )
        rid = db.execute("SELECT last_insert_rowid() as id").fetchone()["id"]
        db.execute(
            "INSERT INTO retrieval_tag_hits (retrieval_id, tag_id, hit_type) VALUES (?, ?, 'direct_match')",
            (rid, t1),
        )
        db.execute(
            "INSERT INTO retrieval_tag_hits (retrieval_id, tag_id, hit_type) VALUES (?, ?, 'direct_match')",
            (rid, t2),
        )
        db.connection.commit()

        edge_count = subgraph.rebuild_edges_from_history()
        assert edge_count == 1

        row = db.execute(
            "SELECT edge_strength FROM centroid_edges "
            "WHERE tag_a_id = ? AND tag_b_id = ?",
            (min(t1, t2), max(t1, t2)),
        ).fetchone()
        assert row is not None
        assert row["edge_strength"] > 0


# ── Retrieval Reweighting ───────────────────────────────────────────────


class TestReweighting:

    def test_returns_net_deltas(self, stack):
        db, subgraph = stack["db"], stack["subgraph"]
        t1 = _insert_tag(db, "rw-a")
        t2 = _insert_tag(db, "rw-b")
        subgraph.update_centroid_on_store(t1, _make_unit_vec(seed=0))
        subgraph.update_centroid_on_store(t2, _make_unit_vec(seed=0))

        a, b = min(t1, t2), max(t1, t2)
        db.execute(
            "INSERT INTO tag_relational_tensor (tag_a_id, tag_b_id, synchronicity_strength) "
            "VALUES (?, ?, 5.0)", (a, b),
        )
        db.connection.commit()

        result = subgraph.update_on_retrieval([t1, t2], mean_relevance=0.8)
        assert isinstance(result, dict)

    def test_no_update_with_single_tag(self, stack):
        db, subgraph = stack["db"], stack["subgraph"]
        tid = _insert_tag(db, "solo")
        subgraph.update_centroid_on_store(tid, _make_unit_vec(seed=0))

        result = subgraph.update_on_retrieval([tid], mean_relevance=0.5)
        assert result == {}

    def test_single_pair_pure_additive(self, stack):
        """With only one pair, parallelness=0 → zero subtraction → pure add."""
        db, subgraph = stack["db"], stack["subgraph"]
        t1 = _insert_tag(db, "solo-a")
        t2 = _insert_tag(db, "solo-b")
        subgraph.update_centroid_on_store(t1, _make_unit_vec(seed=5))
        subgraph.update_centroid_on_store(t2, _make_unit_vec(seed=5))

        a, b = min(t1, t2), max(t1, t2)
        db.execute(
            "INSERT INTO tag_relational_tensor (tag_a_id, tag_b_id, synchronicity_strength) "
            "VALUES (?, ?, 5.0)", (a, b),
        )
        db.connection.commit()

        result = subgraph.update_on_retrieval([t1, t2], mean_relevance=0.8)

        # One pair: no stronger pair to be parallel with → parallelness=0
        # → subtraction=0 → net = add_val (positive)
        if (a, b) in result:
            assert result[(a, b)] > 0

        row = db.execute(
            "SELECT synchronicity_strength FROM tag_relational_tensor "
            "WHERE tag_a_id = ? AND tag_b_id = ?", (a, b),
        ).fetchone()
        assert row["synchronicity_strength"] > 5.0

    def test_strongest_pair_grows_parallel_shrinks(self, stack):
        """Parallel weak pairs shrink; the dominant pair grows."""
        db, subgraph = stack["db"], stack["subgraph"]

        t1 = _insert_tag(db, "grow-a")
        t2 = _insert_tag(db, "grow-b")
        t3 = _insert_tag(db, "grow-c")

        # All 3 centroids along similar directions so all pairs are parallel.
        # t1 ≈ t2 (very close), t3 is a scaled perturbation of t1 — still
        # roughly parallel but further away, creating a clear strength ranking.
        v1 = _make_unit_vec(seed=42)
        v2 = v1 + 0.05 * _make_unit_vec(seed=43)
        v2 = (v2 / np.linalg.norm(v2)).astype(np.float32)
        v3 = v1 + 0.4 * _make_unit_vec(seed=43)
        v3 = (v3 / np.linalg.norm(v3)).astype(np.float32)

        subgraph.update_centroid_on_store(t1, v1)
        subgraph.update_centroid_on_store(t2, v2)
        subgraph.update_centroid_on_store(t3, v3)

        # Insert tensor entries for all 3 pairs at the same starting value
        pairs = sorted([(min(a, b), max(a, b)) for a, b in
                        itertools.combinations([t1, t2, t3], 2)])
        for a, b in pairs:
            db.execute(
                "INSERT INTO tag_relational_tensor "
                "(tag_a_id, tag_b_id, synchronicity_strength) VALUES (?, ?, 5.0)",
                (a, b),
            )
        db.connection.commit()

        result = subgraph.update_on_retrieval([t1, t2, t3], mean_relevance=0.8)

        # The most similar pair (t1, t2) should have positive net delta
        key_similar = (min(t1, t2), max(t1, t2))
        assert result.get(key_similar, 0) > 0

        # At least one parallel weaker pair should have negative net delta
        other_deltas = [v for k, v in result.items() if k != key_similar]
        assert any(d < 0 for d in other_deltas)

    def test_orthogonal_pair_preserved(self, stack):
        """Orthogonal pairs get near-zero subtraction — independent patterns survive."""
        db, subgraph = stack["db"], stack["subgraph"]

        t1 = _insert_tag(db, "orth-a")
        t2 = _insert_tag(db, "orth-b")
        t3 = _insert_tag(db, "orth-c")

        # t1 ≈ t2 (parallel centroids), t3 orthogonal to both
        rng = np.random.RandomState(42)
        v1 = rng.randn(32).astype(np.float32)
        v1 /= np.linalg.norm(v1)
        v2 = v1 + 0.05 * rng.randn(32).astype(np.float32)
        v2 = (v2 / np.linalg.norm(v2)).astype(np.float32)
        # Construct v3 orthogonal to v1 via Gram-Schmidt
        v3_raw = rng.randn(32).astype(np.float32)
        v3_raw -= v3_raw.dot(v1) * v1
        v3_raw -= v3_raw.dot(v2) * v2
        v3 = (v3_raw / np.linalg.norm(v3_raw)).astype(np.float32)

        subgraph.update_centroid_on_store(t1, v1)
        subgraph.update_centroid_on_store(t2, v2)
        subgraph.update_centroid_on_store(t3, v3)

        # Insert tensor entries — orthogonal pair starts at 5.0
        pairs = sorted([(min(a, b), max(a, b)) for a, b in
                        itertools.combinations([t1, t2, t3], 2)])
        for a, b in pairs:
            db.execute(
                "INSERT INTO tag_relational_tensor "
                "(tag_a_id, tag_b_id, synchronicity_strength) VALUES (?, ?, 5.0)",
                (a, b),
            )
        db.connection.commit()

        subgraph.update_on_retrieval([t1, t2, t3], mean_relevance=0.8)

        # The (t1,t2) pair should grow (dominant, zero subtraction)
        key_par = (min(t1, t2), max(t1, t2))
        row_par = db.execute(
            "SELECT synchronicity_strength FROM tag_relational_tensor "
            "WHERE tag_a_id = ? AND tag_b_id = ?", key_par,
        ).fetchone()
        assert row_par["synchronicity_strength"] > 5.0

        # Orthogonal pairs (involving t3) should have near-zero subtraction.
        # With cosine sim near 0 between t3 and others, the incoming strength
        # is near 0 — so both add and subtract are tiny. Value stays near 5.0.
        for pair in pairs:
            if t3 in pair:
                row = db.execute(
                    "SELECT synchronicity_strength FROM tag_relational_tensor "
                    "WHERE tag_a_id = ? AND tag_b_id = ?", pair,
                ).fetchone()
                # Not meaningfully decreased — within a small tolerance of 5.0
                assert row["synchronicity_strength"] >= 4.5, (
                    f"Orthogonal pair {pair} should be preserved, "
                    f"got {row['synchronicity_strength']}"
                )

    def test_tensor_modification(self, stack):
        """Verify tensor values actually change after reweighting."""
        db, subgraph = stack["db"], stack["subgraph"]

        t1 = _insert_tag(db, "tm-a")
        t2 = _insert_tag(db, "tm-b")
        t3 = _insert_tag(db, "tm-c")

        # All roughly parallel so subtraction is meaningful
        v1 = _make_unit_vec(seed=10)
        v2 = v1 + 0.05 * _make_unit_vec(seed=11)
        v2 = (v2 / np.linalg.norm(v2)).astype(np.float32)
        v3 = v1 + 0.4 * _make_unit_vec(seed=11)
        v3 = (v3 / np.linalg.norm(v3)).astype(np.float32)

        subgraph.update_centroid_on_store(t1, v1)
        subgraph.update_centroid_on_store(t2, v2)
        subgraph.update_centroid_on_store(t3, v3)

        pairs = sorted([(min(a, b), max(a, b)) for a, b in
                        itertools.combinations([t1, t2, t3], 2)])
        for a, b in pairs:
            db.execute(
                "INSERT INTO tag_relational_tensor "
                "(tag_a_id, tag_b_id, synchronicity_strength) VALUES (?, ?, 5.0)",
                (a, b),
            )
        db.connection.commit()

        subgraph.update_on_retrieval([t1, t2, t3], mean_relevance=0.8)

        # Check that values diverged from the uniform 5.0
        values = []
        for a, b in pairs:
            row = db.execute(
                "SELECT synchronicity_strength FROM tag_relational_tensor "
                "WHERE tag_a_id = ? AND tag_b_id = ?", (a, b),
            ).fetchone()
            values.append(row["synchronicity_strength"])

        # Not all the same anymore
        assert max(values) > min(values)
        # Strongest pair grew (dominant + zero subtraction)
        assert max(values) > 5.0

    def test_resonance_modification(self, stack):
        """Verify resonance_strength changes after reweighting."""
        db, subgraph = stack["db"], stack["subgraph"]

        t1 = _insert_tag(db, "rm-a")
        t2 = _insert_tag(db, "rm-b")
        t3 = _insert_tag(db, "rm-c")

        v1 = _make_unit_vec(seed=20)
        v2 = v1 + 0.05 * _make_unit_vec(seed=21)
        v2 = (v2 / np.linalg.norm(v2)).astype(np.float32)
        v3 = _make_unit_vec(seed=70)

        subgraph.update_centroid_on_store(t1, v1)
        subgraph.update_centroid_on_store(t2, v2)
        subgraph.update_centroid_on_store(t3, v3)

        # Create two sync events: one with [t1, t2], one with [t3]
        # Cross-event pairs: (t1,t3) and (t2,t3) — at least one has non-zero net
        db.execute(
            "INSERT INTO synchronicity_events "
            "(event_type, description, strength, quadrant, involved_tags, reinforcement_count) "
            "VALUES ('test', 'test', 3.0, 'Q1', ?, 0)",
            (json.dumps([t1, t2]),),
        )
        eid1 = db.execute("SELECT last_insert_rowid() as id").fetchone()["id"]
        db.execute(
            "INSERT INTO synchronicity_events "
            "(event_type, description, strength, quadrant, involved_tags, reinforcement_count) "
            "VALUES ('test', 'test', 3.0, 'Q1', ?, 0)",
            (json.dumps([t3]),),
        )
        eid2 = db.execute("SELECT last_insert_rowid() as id").fetchone()["id"]

        ea, eb = min(eid1, eid2), max(eid1, eid2)
        db.execute(
            "INSERT INTO resonances "
            "(event_a_id, event_b_id, event_ids, shared_primes, resonance_strength, description) "
            "VALUES (?, ?, ?, '[2,3]', 5.0, 'test')",
            (ea, eb, json.dumps([ea, eb])),
        )
        db.connection.commit()

        subgraph.update_on_retrieval([t1, t2, t3], mean_relevance=0.8)

        row = db.execute(
            "SELECT resonance_strength FROM resonances WHERE event_a_id = ? AND event_b_id = ?",
            (ea, eb),
        ).fetchone()
        # Value should have changed from 5.0
        assert row["resonance_strength"] != 5.0

    def test_strength_floors_at_zero(self, stack):
        db, subgraph = stack["db"], stack["subgraph"]

        t1 = _insert_tag(db, "floor-a")
        t2 = _insert_tag(db, "floor-b")
        t3 = _insert_tag(db, "floor-c")

        # All parallel so subtraction is meaningful, but different enough
        # to create a clear ranking
        v1 = _make_unit_vec(seed=0)
        v2 = v1 + 0.05 * _make_unit_vec(seed=1)
        v2 = (v2 / np.linalg.norm(v2)).astype(np.float32)
        v3 = v1 + 0.4 * _make_unit_vec(seed=1)
        v3 = (v3 / np.linalg.norm(v3)).astype(np.float32)
        subgraph.update_centroid_on_store(t1, v1)
        subgraph.update_centroid_on_store(t2, v2)
        subgraph.update_centroid_on_store(t3, v3)

        pairs = sorted([(min(a, b), max(a, b)) for a, b in
                        itertools.combinations([t1, t2, t3], 2)])
        for a, b in pairs:
            db.execute(
                "INSERT INTO tag_relational_tensor "
                "(tag_a_id, tag_b_id, synchronicity_strength) VALUES (?, ?, 0.001)",
                (a, b),
            )
        db.connection.commit()

        # High scale to ensure subtraction exceeds existing value
        stack["config"].centroid_inhibition_scale = 50.0
        subgraph.update_on_retrieval([t1, t2, t3], mean_relevance=0.9)

        for a, b in pairs:
            row = db.execute(
                "SELECT synchronicity_strength FROM tag_relational_tensor "
                "WHERE tag_a_id = ? AND tag_b_id = ?", (a, b),
            ).fetchone()
            assert row["synchronicity_strength"] >= 0.0


# ── Blob Helpers ────────────────────────────────────────────────────────


class TestBlobHelpers:

    def test_roundtrip(self):
        v = _make_unit_vec(seed=42)
        blob = _array_to_blob(v)
        restored = _blob_to_array(blob)
        np.testing.assert_array_equal(v, restored)
