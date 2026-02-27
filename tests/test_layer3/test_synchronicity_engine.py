"""Tests for the prime Ramsey lattice synchronicity engine."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer1.tag_manager import TagManager
from chicory.layer3.synchronicity_engine import SynchronicityEngine
from chicory.models.synchronicity import SynchronicityEvent
from tests.conftest import MockEmbeddingEngine


@pytest.fixture
def stack():
    """Build a test stack with engine, db, tags, and mock embeddings."""
    config = ChicoryConfig(
        db_path=Path(":memory:"),
        anthropic_api_key="test",
        llm_model="test-model",
        embedding_model="mock",
        embedding_dimension=32,
        lattice_primes=[2, 3, 5, 7, 11],
        lattice_min_resonance_primes=3,
        lattice_void_radius=0.5,
    )

    db = DatabaseEngine(config)
    db.connect()
    apply_schema(db)

    mock_emb = MockEmbeddingEngine(dimension=32)
    mock_emb.set_db(db)
    tag_mgr = TagManager(db)

    engine = SynchronicityEngine(config, db, mock_emb, tag_mgr)

    yield {
        "config": config,
        "db": db,
        "emb": mock_emb,
        "tags": tag_mgr,
        "engine": engine,
    }

    db.close()


def _create_memory(db, emb, memory_id: str, content: str, tag_ids: list[int]):
    """Helper: insert a memory with embedding and tag associations."""
    db.execute(
        "INSERT INTO memories (id, content, source_model) VALUES (?, ?, 'test')",
        (memory_id, content),
    )
    for tid in tag_ids:
        db.execute(
            "INSERT INTO memory_tags (memory_id, tag_id) VALUES (?, ?)",
            (memory_id, tid),
        )
    emb.store_cached(memory_id, emb.embed(content))
    db.connection.commit()


def _create_sync_event(db, tag_ids: list[int], **kwargs) -> SynchronicityEvent:
    """Helper: insert a sync event and return it with its persisted ID."""
    defaults = {
        "event_type": "test",
        "description": "test event",
        "strength": 1.0,
        "quadrant": "test",
    }
    defaults.update(kwargs)

    db.execute(
        """INSERT INTO synchronicity_events
           (event_type, description, strength, quadrant, involved_tags)
           VALUES (?, ?, ?, ?, ?)""",
        (
            defaults["event_type"],
            defaults["description"],
            defaults["strength"],
            defaults["quadrant"],
            json.dumps(tag_ids),
        ),
    )
    db.connection.commit()
    event_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

    return SynchronicityEvent(
        id=event_id,
        event_type=defaults["event_type"],
        description=defaults["description"],
        strength=defaults["strength"],
        quadrant=defaults["quadrant"],
        involved_tags=json.dumps(tag_ids),
    )


# ── Placement tests ──────────────────────────────────────────────────


class TestPlaceEvent:
    def test_returns_position_with_valid_angle(self, stack):
        tag = stack["tags"].get_or_create("test-topic")
        _create_memory(stack["db"], stack["emb"], "m1", "test content", [tag.id])

        event = _create_sync_event(stack["db"], [tag.id])
        pos = stack["engine"].place_event(event)

        assert pos is not None
        assert 0 <= pos.angle < 2 * math.pi
        assert pos.sync_event_id == event.id

    def test_prime_slots_within_bounds(self, stack):
        tag = stack["tags"].get_or_create("slot-bounds")
        _create_memory(stack["db"], stack["emb"], "m1", "content a", [tag.id])

        event = _create_sync_event(stack["db"], [tag.id])
        pos = stack["engine"].place_event(event)

        assert pos is not None
        slots = json.loads(pos.prime_slots)
        for p in stack["config"].lattice_primes:
            assert 0 <= slots[str(p)] < p

    def test_no_id_returns_none(self, stack):
        event = SynchronicityEvent(
            event_type="test",
            description="no id",
            strength=1.0,
            quadrant="test",
            involved_tags="[1]",
        )
        assert stack["engine"].place_event(event) is None

    def test_empty_tags_returns_none(self, stack):
        event = _create_sync_event(stack["db"], [])
        assert stack["engine"].place_event(event) is None

    def test_tag_with_no_memories_returns_none(self, stack):
        tag = stack["tags"].get_or_create("orphan-tag")
        event = _create_sync_event(stack["db"], [tag.id])
        assert stack["engine"].place_event(event) is None

    def test_idempotent_placement(self, stack):
        tag = stack["tags"].get_or_create("idem")
        _create_memory(stack["db"], stack["emb"], "m1", "idempotent", [tag.id])

        event = _create_sync_event(stack["db"], [tag.id])
        pos1 = stack["engine"].place_event(event)
        pos2 = stack["engine"].place_event(event)

        assert pos1 is not None
        assert pos2 is not None
        assert pos1.angle == pos2.angle
        assert pos1.prime_slots == pos2.prime_slots

    def test_different_tags_get_different_angles(self, stack):
        tag_a = stack["tags"].get_or_create("alpha")
        tag_b = stack["tags"].get_or_create("beta")
        _create_memory(stack["db"], stack["emb"], "m1", "alpha content", [tag_a.id])
        _create_memory(stack["db"], stack["emb"], "m2", "beta content", [tag_b.id])

        event_a = _create_sync_event(stack["db"], [tag_a.id])
        event_b = _create_sync_event(stack["db"], [tag_b.id])

        stack["engine"].invalidate_pca_cache()
        pos_a = stack["engine"].place_event(event_a)
        pos_b = stack["engine"].place_event(event_b)

        assert pos_a is not None and pos_b is not None
        # Different tag embeddings should produce different angles
        # (with high probability given the mock hash-based embeddings)
        assert pos_a.angle != pos_b.angle


# ── Batch placement ──────────────────────────────────────────────────


class TestPlaceEventsBatch:
    def test_batch_places_multiple(self, stack):
        tag_a = stack["tags"].get_or_create("batch-a")
        tag_b = stack["tags"].get_or_create("batch-b")
        _create_memory(stack["db"], stack["emb"], "m1", "batch a", [tag_a.id])
        _create_memory(stack["db"], stack["emb"], "m2", "batch b", [tag_b.id])

        events = [
            _create_sync_event(stack["db"], [tag_a.id]),
            _create_sync_event(stack["db"], [tag_b.id]),
        ]
        placed = stack["engine"].place_events_batch(events)
        assert len(placed) == 2

    def test_batch_skips_unplaceable(self, stack):
        tag = stack["tags"].get_or_create("good")
        _create_memory(stack["db"], stack["emb"], "m1", "good content", [tag.id])

        events = [
            _create_sync_event(stack["db"], [tag.id]),
            _create_sync_event(stack["db"], []),  # No tags — unplaceable
        ]
        placed = stack["engine"].place_events_batch(events)
        assert len(placed) == 1


# ── Resonance detection ──────────────────────────────────────────────


class TestFindResonances:
    def test_empty_lattice_returns_empty(self, stack):
        assert stack["engine"].find_resonances() == []

    def test_single_event_returns_empty(self, stack):
        tag = stack["tags"].get_or_create("solo")
        _create_memory(stack["db"], stack["emb"], "m1", "solo content", [tag.id])
        event = _create_sync_event(stack["db"], [tag.id])
        stack["engine"].place_event(event)

        assert stack["engine"].find_resonances() == []

    def test_identical_tags_produce_full_resonance(self, stack):
        """Two events with the same tag should land at the same angle
        and resonate across ALL primes."""
        tag = stack["tags"].get_or_create("shared")
        _create_memory(stack["db"], stack["emb"], "m1", "shared content", [tag.id])

        event_a = _create_sync_event(stack["db"], [tag.id], description="event a")
        event_b = _create_sync_event(stack["db"], [tag.id], description="event b")

        stack["engine"].place_events_batch([event_a, event_b])
        resonances = stack["engine"].find_resonances()

        assert len(resonances) == 1
        r = resonances[0]
        shared = json.loads(r.shared_primes)
        assert len(shared) == len(stack["config"].lattice_primes)

    def test_resonance_strength_is_log_sum(self, stack):
        """Resonance strength should equal sum(log(p)) for shared primes."""
        tag = stack["tags"].get_or_create("strength-test")
        _create_memory(stack["db"], stack["emb"], "m1", "strength", [tag.id])

        event_a = _create_sync_event(stack["db"], [tag.id])
        event_b = _create_sync_event(stack["db"], [tag.id])

        stack["engine"].place_events_batch([event_a, event_b])
        resonances = stack["engine"].find_resonances()

        assert len(resonances) == 1
        primes = stack["config"].lattice_primes
        expected = sum(math.log(p) for p in primes)
        assert abs(resonances[0].resonance_strength - expected) < 1e-6

    def test_min_shared_primes_filters(self, stack):
        """Raising min_shared_primes should filter out weaker resonances."""
        tag = stack["tags"].get_or_create("filter-test")
        _create_memory(stack["db"], stack["emb"], "m1", "filter", [tag.id])

        event_a = _create_sync_event(stack["db"], [tag.id])
        event_b = _create_sync_event(stack["db"], [tag.id])

        stack["engine"].place_events_batch([event_a, event_b])

        # With max threshold, should still find (they share all 5 primes)
        assert len(stack["engine"].find_resonances(min_shared_primes=5)) == 1
        # With impossibly high threshold, should find nothing
        assert len(stack["engine"].find_resonances(min_shared_primes=6)) == 0


# ── Void profiling ───────────────────────────────────────────────────


class TestVoidProfile:
    def test_too_few_events_returns_none(self, stack):
        assert stack["engine"].compute_void_profile() is None

    def test_two_events_returns_none(self, stack):
        tag = stack["tags"].get_or_create("two")
        _create_memory(stack["db"], stack["emb"], "m1", "two a", [tag.id])
        _create_memory(stack["db"], stack["emb"], "m2", "two b", [tag.id])

        for i in range(2):
            event = _create_sync_event(stack["db"], [tag.id])
            stack["engine"].place_event(event)

        assert stack["engine"].compute_void_profile() is None

    def test_three_events_returns_profile(self, stack):
        tags = [
            stack["tags"].get_or_create(f"void-{i}") for i in range(3)
        ]
        for i, tag in enumerate(tags):
            _create_memory(
                stack["db"], stack["emb"], f"m{i}", f"void content {i}", [tag.id]
            )

        for tag in tags:
            event = _create_sync_event(stack["db"], [tag.id])
            stack["engine"].place_event(event)

        profile = stack["engine"].compute_void_profile()
        assert profile is not None
        assert profile.void_radius >= 0
        edge_tags = json.loads(profile.edge_tags)
        assert isinstance(edge_tags, list)
        assert len(profile.description) > 0


# ── Lattice state ────────────────────────────────────────────────────


class TestGetLatticeState:
    def test_empty_state(self, stack):
        state = stack["engine"].get_lattice_state()
        assert state["position_count"] == 0
        assert state["resonance_count"] == 0
        assert state["void_profile"] is None
        assert state["primes_used"] == stack["config"].lattice_primes

    def test_populated_state(self, stack):
        tags = [
            stack["tags"].get_or_create(f"state-{i}") for i in range(4)
        ]
        for i, tag in enumerate(tags):
            _create_memory(
                stack["db"], stack["emb"], f"m{i}", f"state content {i}", [tag.id]
            )

        for tag in tags:
            event = _create_sync_event(stack["db"], [tag.id])
            stack["engine"].place_event(event)

        state = stack["engine"].get_lattice_state()
        assert state["position_count"] == 4
        assert state["void_profile"] is not None

        for pos in state["positions"]:
            assert "sync_event_id" in pos
            assert "angle" in pos
            assert "prime_slots" in pos
            assert "event_type" in pos
