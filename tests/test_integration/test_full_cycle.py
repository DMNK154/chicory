"""Integration test: full cycle from storage through synchronicity detection."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pytest

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer1.memory_store import MemoryStore
from chicory.layer1.salience import SalienceScorer
from chicory.layer1.tag_manager import TagManager
from chicory.layer2.retrieval_tracker import RetrievalTracker
from chicory.layer2.trend_engine import TrendEngine
from chicory.layer3.phase_space import PhaseSpace
from chicory.layer3.synchronicity_detector import SynchronicityDetector
from chicory.layer4.adaptive_thresholds import AdaptiveThresholds
from chicory.layer4.feedback import FeedbackEngine
from chicory.layer4.meta_analyzer import MetaAnalyzer
from chicory.models.phase import Quadrant

from tests.conftest import MockEmbeddingEngine


@pytest.fixture
def full_stack():
    """Build the full stack with mock embeddings."""
    from pathlib import Path

    config = ChicoryConfig(
        db_path=Path(":memory:"),
        anthropic_api_key="test",
        llm_model="test-model",
        embedding_model="mock",
        embedding_dimension=32,
        similarity_threshold=0.0,  # Accept all for testing
    )

    db = DatabaseEngine(config)
    db.connect()
    apply_schema(db)

    mock_emb = MockEmbeddingEngine(dimension=32)
    mock_emb.set_db(db)

    tag_mgr = TagManager(db)
    salience = SalienceScorer(config, db)
    mem_store = MemoryStore(config, db, mock_emb, tag_mgr, salience)
    trend_engine = TrendEngine(config, db)
    ret_tracker = RetrievalTracker(config, db)
    phase_space = PhaseSpace(config, db, trend_engine, ret_tracker)
    sync_detector = SynchronicityDetector(
        config, db, phase_space, trend_engine, ret_tracker, tag_mgr, mock_emb
    )
    adaptive = AdaptiveThresholds(config, db)
    meta_analyzer = MetaAnalyzer(config, db, adaptive)
    feedback = FeedbackEngine(db, tag_mgr, salience)

    yield {
        "config": config,
        "db": db,
        "tag_mgr": tag_mgr,
        "salience": salience,
        "mem_store": mem_store,
        "mock_emb": mock_emb,
        "trend_engine": trend_engine,
        "ret_tracker": ret_tracker,
        "phase_space": phase_space,
        "sync_detector": sync_detector,
        "adaptive": adaptive,
        "meta_analyzer": meta_analyzer,
        "feedback": feedback,
    }

    db.close()


class TestDatabaseSchema:
    def test_schema_applied(self, db):
        """All tables should exist after schema application."""
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {r["name"] for r in tables}

        expected = {
            "schema_version", "memories", "embeddings", "tags", "memory_tags",
            "tag_events", "retrieval_events", "retrieval_results",
            "retrieval_tag_hits", "trend_snapshots", "synchronicity_events",
            "meta_patterns", "adaptive_thresholds", "model_versions",
        }
        assert expected.issubset(table_names)

    def test_schema_version(self, db):
        row = db.execute("SELECT MAX(version) as version FROM schema_version").fetchone()
        assert row["version"] == 11


class TestTagManager:
    def test_create_and_retrieve(self, db):
        tm = TagManager(db)
        tag = tm.get_or_create("Test Tag", created_by="test")
        assert tag.name == "test-tag"  # Normalized
        assert tag.is_active

    def test_find_similar(self, db):
        tm = TagManager(db)
        tm.get_or_create("machine-learning")
        similar = tm.find_similar_tags("machine-learnin")  # Typo
        assert len(similar) > 0

    def test_merge(self, db):
        tm = TagManager(db)
        t1 = tm.get_or_create("ml")
        t2 = tm.get_or_create("machine-learning")

        # Create a memory with t1
        db.execute(
            "INSERT INTO memories (id, content, source_model) VALUES ('m1', 'test', 'test')"
        )
        db.execute(
            "INSERT INTO memory_tags (memory_id, tag_id) VALUES ('m1', ?)",
            (t1.id,),
        )
        db.connection.commit()

        tm.merge_tags(t1.id, t2.id)

        # t1 should be inactive and merged
        t1_after = tm.get_by_id(t1.id)
        assert not t1_after.is_active
        assert t1_after.merged_into == t2.id


class TestMemoryStore:
    def test_store_and_retrieve(self, full_stack):
        ms = full_stack["mem_store"]

        mem = ms.store(
            content="Neural networks learn hierarchical representations",
            tags=["deep-learning", "representations"],
            salience_model=0.7,
        )

        assert mem.id
        assert mem.tags == ["deep-learning", "representations"]
        assert mem.salience_model == 0.7

        # Retrieve by ID
        retrieved = ms.get_by_id(mem.id)
        assert retrieved.content == mem.content

    def test_semantic_retrieval(self, full_stack):
        ms = full_stack["mem_store"]

        ms.store("Cats are great pets", tags=["animals"])
        ms.store("Dogs are loyal companions", tags=["animals"])
        ms.store("Python is a programming language", tags=["coding"])

        results = ms.retrieve_semantic("pets and animals", top_k=3)
        assert len(results) > 0

    def test_tag_retrieval(self, full_stack):
        ms = full_stack["mem_store"]

        ms.store("Memory about math", tags=["mathematics"])
        ms.store("Memory about code", tags=["coding"])
        ms.store("Memory about math and code", tags=["mathematics", "coding"])

        results = ms.retrieve_by_tags(["mathematics"])
        assert len(results) == 2  # math only + math and code


class TestTrendEngine:
    def test_compute_trend(self, full_stack):
        te = full_stack["trend_engine"]
        tm = full_stack["tag_mgr"]

        tag = tm.get_or_create("test-topic")
        te.record_event(tag.id, "assignment", weight=1.0)
        te.record_event(tag.id, "assignment", weight=1.0)
        te.record_event(tag.id, "retrieval", weight=1.0)

        trend = te.compute_trend(tag.id)
        assert trend.event_count == 3
        assert trend.level > 0
        assert 0 <= trend.temperature <= 1

    def test_empty_trend(self, full_stack):
        te = full_stack["trend_engine"]
        trend = te.compute_trend(999)
        assert trend.level == 0
        assert trend.temperature == 0
        assert trend.event_count == 0


class TestPhaseSpace:
    def test_quadrant_classification(self, full_stack):
        ps = full_stack["phase_space"]
        te = full_stack["trend_engine"]
        tm = full_stack["tag_mgr"]

        # Create a tag with no activity — should be inactive
        tag = tm.get_or_create("dormant-topic")
        coord = ps.compute_coordinate(tag.id)
        assert coord.quadrant == Quadrant.INACTIVE

    def test_all_coordinates(self, full_stack):
        ps = full_stack["phase_space"]
        tm = full_stack["tag_mgr"]

        tm.get_or_create("topic-a")
        tm.get_or_create("topic-b")

        coords = ps.compute_all_coordinates()
        assert len(coords) == 2


class TestSalienceScorer:
    def test_composite_scoring(self, full_stack):
        sal = full_stack["salience"]
        config = full_stack["config"]

        composite = sal.compute_composite(0.8, 0.6)
        expected = config.salience_model_weight * 0.8 + config.salience_usage_weight * 0.6
        assert abs(composite - expected) < 0.001

    def test_update_on_access(self, full_stack):
        ms = full_stack["mem_store"]
        sal = full_stack["salience"]

        mem = ms.store("Test memory", tags=["test"], salience_model=0.5)
        sal.update_on_access(mem.id)

        updated = ms.get_by_id(mem.id)
        assert updated.access_count == 1
        assert updated.last_accessed is not None


class TestAdaptiveThresholds:
    def test_default_threshold(self, full_stack):
        at = full_stack["adaptive"]
        val = at.get_threshold("sync_detection_sigma")
        assert val == 2.0  # Default

    def test_update_and_retrieve(self, full_stack):
        at = full_stack["adaptive"]
        config = full_stack["config"]

        at.update_threshold("test_metric", 5.0)
        val = at.get_threshold("test_metric")
        assert val == 5.0  # First value is exact

        at.update_threshold("test_metric", 10.0)
        val = at.get_threshold("test_metric")
        # EMA: 0.1 * 10.0 + 0.9 * 5.0 = 5.5
        assert abs(val - 5.5) < 0.01

    def test_burn_in_widens_threshold(self, full_stack):
        at = full_stack["adaptive"]
        config = full_stack["config"]

        at.update_threshold("test_metric", 3.0)
        normal = at.get_threshold("test_metric")

        at.enter_burn_in("new-model")
        burn_in = at.get_threshold("test_metric")
        assert burn_in > normal  # Should be wider


class TestRetrievalTracker:
    def test_log_and_frequency(self, full_stack):
        rt = full_stack["ret_tracker"]
        tm = full_stack["tag_mgr"]
        ms = full_stack["mem_store"]

        tag = tm.get_or_create("tracked-topic")

        # Need a real memory for FK constraint
        mem = ms.store("Tracked test memory", tags=["tracked-topic"])

        rid = rt.log_retrieval(
            query_text="test query",
            method="semantic",
            results=[(mem.id, 1, 0.9)],
            model_version="test",
        )

        rt.log_tag_hits(rid, [(tag.id, "direct_match")])

        freq = rt.get_tag_retrieval_frequency(tag.id)
        assert freq > 0


class TestFullCycle:
    def test_store_retrieve_trend_phase(self, full_stack):
        """Full cycle: store memories, retrieve, check trends and phase space."""
        ms = full_stack["mem_store"]
        te = full_stack["trend_engine"]
        tm = full_stack["tag_mgr"]
        ps = full_stack["phase_space"]
        rt = full_stack["ret_tracker"]

        # Store memories across different domains
        ms.store("Topology studies shapes and spaces", tags=["mathematics", "topology"])
        ms.store("Group theory is about symmetry", tags=["mathematics", "algebra"])
        ms.store("Consciousness is the hard problem", tags=["philosophy", "consciousness"])
        ms.store("Neural nets are universal approximators", tags=["ai", "deep-learning"])
        ms.store("Jung proposed the collective unconscious", tags=["psychology", "jung"])

        # Simulate retrieval activity biased toward mathematics
        # Use tag-based retrieval since mock embeddings aren't semantically meaningful
        results = ms.retrieve_by_tags(["mathematics"])
        assert len(results) > 0, "Should find math memories by tag"

        rid = rt.log_retrieval(
            query_text="mathematical structures",
            method="tag",
            results=[(m.id, i + 1, 1.0) for i, m in enumerate(results)],
            model_version="test",
        )

        # Record tag hits for the retrieval
        for mem in results:
            tag_ids = tm.get_tag_ids_for_memory(mem.id)
            rt.log_tag_hits(rid, [(tid, "direct_match") for tid in tag_ids])
            for tid in tag_ids:
                te.record_event(tid, "retrieval", memory_id=mem.id)

        # Check trends
        math_tag = tm.get_by_name("mathematics")
        assert math_tag is not None
        trend = te.compute_trend(math_tag.id)
        assert trend.event_count > 0

        # Check phase space
        coords = ps.compute_all_coordinates()
        assert len(coords) > 0

        # All tags should have coordinates
        for tag in tm.list_active():
            assert tag.id in coords
