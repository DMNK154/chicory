"""Canopy observer tests."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer4.canopy import CanopyObserver
from chicory.layer4.forest import ForestReorganizer


def _build_observer(**overrides):
    config = ChicoryConfig(
        db_path=Path(":memory:"),
        embedding_model="mock",
        embedding_dimension=32,
        **overrides,
    )
    db = DatabaseEngine(config)
    db.connect()
    apply_schema(db)
    forest = ForestReorganizer(db, config)
    return config, db, CanopyObserver(db, config, forest)


def _insert_memories(db: DatabaseEngine, count: int) -> list[str]:
    """Insert stub memories and return their IDs."""
    mids: list[str] = []
    for i in range(count):
        mid = str(uuid.uuid4())
        db.execute(
            "INSERT INTO memories (id, content, summary, source_model) VALUES (?, ?, ?, 'test')",
            (mid, f"memory content {i}", f"memory {i}"),
        )
        mids.append(mid)
    return mids


def _insert_episodic_edge(
    db: DatabaseEngine,
    mid_a: str,
    mid_b: str,
    co_retrieval: float = 1.0,
    bridge: float = 0.0,
) -> None:
    """Insert an episodic tensor edge between two memories."""
    a, b = min(mid_a, mid_b), max(mid_a, mid_b)
    db.execute(
        """INSERT INTO memory_relational_tensor
           (memory_a_id, memory_b_id, co_retrieval_strength, bridge_strength,
            semantic_strength, tag_projected_strength, temporal_proximity,
            source_proximity, edge_status, activation_count)
           VALUES (?, ?, ?, ?, 0.0, 0.0, 0.0, 0.0, 'candidate', 1)""",
        (a, b, co_retrieval, bridge),
    )


def test_discover_clusters_from_episodic_edges():
    """Memories connected by co-retrieval edges form clusters."""
    _, db, observer = _build_observer()
    try:
        mids = _insert_memories(db, 4)

        # Two edges: 0-1, 1-2 → component {0,1,2}; 3 is isolated
        _insert_episodic_edge(db, mids[0], mids[1])
        _insert_episodic_edge(db, mids[1], mids[2])

        shapes = observer._discover_clusters(mids, "retrieval", "1")

        assert len(shapes) == 1
        assert sorted(shapes[0].memory_ids) == sorted(mids[:3])
        assert shapes[0].block_type == "memory_cluster"
    finally:
        db.close()


def test_discover_clusters_multiple_components():
    """Disjoint edge groups form separate clusters."""
    _, db, observer = _build_observer()
    try:
        mids = _insert_memories(db, 4)

        # Two separate components: {0,1} and {2,3}
        _insert_episodic_edge(db, mids[0], mids[1])
        _insert_episodic_edge(db, mids[2], mids[3])

        shapes = observer._discover_clusters(mids, "retrieval", "1")

        assert len(shapes) == 2
        cluster_sets = [set(s.memory_ids) for s in shapes]
        assert {mids[0], mids[1]} in cluster_sets
        assert {mids[2], mids[3]} in cluster_sets
    finally:
        db.close()


def test_discover_clusters_no_edges_returns_empty():
    """No episodic edges means no clusters discovered."""
    _, db, observer = _build_observer()
    try:
        mids = _insert_memories(db, 3)
        shapes = observer._discover_clusters(mids, "retrieval", "1")
        assert shapes == []
    finally:
        db.close()


def test_observe_records_canopy_blocks():
    """observe() should discover clusters and record canopy blocks."""
    _, db, observer = _build_observer()
    try:
        mids = _insert_memories(db, 3)
        _insert_episodic_edge(db, mids[0], mids[1])
        _insert_episodic_edge(db, mids[1], mids[2])

        grown = observer.observe(
            source="retrieval",
            source_id="1",
            memory_ids=mids,
        )

        blocks = db.execute("SELECT * FROM canopy_blocks").fetchall()
        assert len(blocks) >= 1
        assert blocks[0]["block_type"] == "memory_cluster"
    finally:
        db.close()


def test_observe_disabled_returns_empty():
    """observe() with canopy_enabled=False returns empty."""
    _, db, observer = _build_observer(canopy_enabled=False)
    try:
        mids = _insert_memories(db, 3)
        _insert_episodic_edge(db, mids[0], mids[1])

        grown = observer.observe(
            source="retrieval",
            source_id="1",
            memory_ids=mids,
        )
        assert grown == []
    finally:
        db.close()
