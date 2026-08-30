"""Tests for the librarian: write gate, linear sweep, subject roster,
and the provenance/quarantine plumbing through the memory store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer1.salience import SalienceScorer
from chicory.layer1.memory_store import MemoryStore
from chicory.layer1.tag_manager import TagManager
from chicory.layer3.centroid_subgraph import CentroidSubgraph, _array_to_blob
from chicory.layer4.librarian import LinearSweep, SubjectRoster, WriteGate
from tests.conftest import MockEmbeddingEngine

DIM = 32


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def stack():
    config = ChicoryConfig(
        db_path=Path(":memory:"),
        anthropic_api_key="test",
        llm_model="test-model",
        embedding_model="mock",
        embedding_dimension=DIM,
        librarian_gate_min_centroids=1,
        librarian_gate_threshold=0.5,
        librarian_sweep_batch_size=10,
        librarian_sweep_near_dup_threshold=0.95,
        librarian_sweep_drift_threshold=0.15,
        subject_min_edge_strength=0.05,
        subject_min_cluster_size=2,
        subject_hub_tag_max_fraction=999.0,  # disabled; hub rule tested explicitly
    )
    db = DatabaseEngine(config)
    db.connect()
    apply_schema(db)

    emb = MockEmbeddingEngine(dimension=DIM)
    emb.set_db(db)
    tags = TagManager(db)
    centroids = CentroidSubgraph(config, db, emb)

    yield {
        "config": config,
        "db": db,
        "emb": emb,
        "tags": tags,
        "centroids": centroids,
        "gate": WriteGate(config, db, tags, centroids),
        "sweep": LinearSweep(config, db, emb, tags, centroids),
        "roster": SubjectRoster(config, db, centroids),
    }
    db.close()


def _basis_vec(i: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def _insert_tag(db, name: str) -> int:
    db.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (name,))
    return db.execute(
        "SELECT id FROM tags WHERE name = ?", (name,),
    ).fetchone()["id"]


def _insert_centroid(db, tag_id: int, vec: np.ndarray, memory_count: int = 5):
    db.execute(
        "INSERT OR REPLACE INTO tag_centroids (tag_id, centroid, memory_count) "
        "VALUES (?, ?, ?)",
        (tag_id, _array_to_blob(vec), memory_count),
    )


def _insert_memory(db, memory_id: str, content: str = "test",
                   tier: str = "critical") -> None:
    db.execute(
        "INSERT OR IGNORE INTO memories (id, content, source_model, ingestion_tier) "
        "VALUES (?, ?, 'test', ?)",
        (memory_id, content, tier),
    )


def _assign_tag(db, memory_id: str, tag_id: int) -> None:
    db.execute(
        "INSERT OR IGNORE INTO memory_tags (memory_id, tag_id) VALUES (?, ?)",
        (memory_id, tag_id),
    )


# ── WriteGate ───────────────────────────────────────────────────────────


class TestWriteGate:
    def test_cold_start_passes_everything(self, stack):
        stack["config"].librarian_gate_min_centroids = 10
        verdict = stack["gate"].evaluate(_basis_vec(0), ["anything"])
        assert verdict.verdict == "cold_start"
        assert not verdict.diverted

    def test_on_topic_passes(self, stack):
        tid = _insert_tag(stack["db"], "physics")
        _insert_centroid(stack["db"], tid, _basis_vec(0))
        verdict = stack["gate"].evaluate(_basis_vec(0), ["physics"])
        assert verdict.verdict == "pass"
        assert verdict.best_tag_id == tid
        assert verdict.score > 0.9

    def test_off_topic_shadow_quarantine(self, stack):
        tid = _insert_tag(stack["db"], "physics")
        _insert_centroid(stack["db"], tid, _basis_vec(0))
        verdict = stack["gate"].evaluate(_basis_vec(1), ["physics"])
        assert verdict.verdict == "shadow_quarantine"
        assert not verdict.diverted  # shadow mode never diverts

    def test_off_topic_enforced_quarantine(self, stack):
        stack["config"].librarian_gate_enforce = True
        tid = _insert_tag(stack["db"], "physics")
        _insert_centroid(stack["db"], tid, _basis_vec(0))
        verdict = stack["gate"].evaluate(_basis_vec(1), ["physics"])
        assert verdict.verdict == "quarantine"
        assert verdict.diverted

    def test_global_fallback_passes_new_tags(self, stack):
        # Claimed tag has no centroid, but content matches another shelf:
        # belongs-somewhere passes — mis-shelving is the sweep's job.
        tid = _insert_tag(stack["db"], "physics")
        _insert_centroid(stack["db"], tid, _basis_vec(0))
        verdict = stack["gate"].evaluate(_basis_vec(0), ["brand-new-tag"])
        assert verdict.verdict == "pass"
        assert verdict.best_tag_id == tid

    def test_cache_refreshes_on_centroid_update(self, stack):
        # EMA updates shift centroid values without changing the row count;
        # the global-fallback matrix must still refresh (keyed on updated_at)
        tid = _insert_tag(stack["db"], "physics")
        _insert_centroid(stack["db"], tid, _basis_vec(0))
        v1 = stack["gate"].evaluate(_basis_vec(1), ["new-tag"])
        assert v1.verdict == "shadow_quarantine"
        # Move the sole centroid to match the content, count unchanged.
        # Force a distinct updated_at — same-millisecond REPLACEs would
        # leave MAX(updated_at) unchanged and this test flaky.
        _insert_centroid(stack["db"], tid, _basis_vec(1))
        stack["db"].execute(
            "UPDATE tag_centroids SET updated_at = '2999-01-01T00:00:00' "
            "WHERE tag_id = ?", (tid,),
        )
        v2 = stack["gate"].evaluate(_basis_vec(1), ["new-tag"])
        assert v2.verdict == "pass"

    def test_multi_chunk_worst_chunk_governs(self, stack):
        tid = _insert_tag(stack["db"], "physics")
        _insert_centroid(stack["db"], tid, _basis_vec(0))
        verdict = stack["gate"].evaluate_content(
            [_basis_vec(0), _basis_vec(1)], ["physics"],
        )
        assert verdict.verdict == "shadow_quarantine"

    def test_log_writes_row(self, stack):
        tid = _insert_tag(stack["db"], "physics")
        _insert_centroid(stack["db"], tid, _basis_vec(0))
        verdict = stack["gate"].evaluate(_basis_vec(1), ["physics"])
        stack["gate"].log("mem-1", verdict)
        row = stack["db"].execute(
            "SELECT * FROM write_gate_log WHERE memory_id = 'mem-1'"
        ).fetchone()
        assert row["verdict"] == "shadow_quarantine"
        assert row["enforced"] == 0


# ── LinearSweep ─────────────────────────────────────────────────────────


class TestLinearSweep:
    def test_cursor_advances_and_is_durable(self, stack):
        for i in range(3):
            _insert_memory(stack["db"], f"m{i}")
            stack["emb"].store_cached(f"m{i}", _basis_vec(i))
        stats = stack["sweep"].run_batch(batch_size=2)
        assert stats["swept"] == 2
        swept = stack["db"].execute(
            "SELECT COUNT(*) AS n FROM memories WHERE swept_at IS NOT NULL"
        ).fetchone()["n"]
        assert swept == 2
        # Second batch picks up the remaining unswept memory first
        stack["sweep"].run_batch(batch_size=2)
        swept = stack["db"].execute(
            "SELECT COUNT(*) AS n FROM memories WHERE swept_at IS NOT NULL"
        ).fetchone()["n"]
        assert swept == 3

    def test_missing_embedding_finding(self, stack):
        _insert_memory(stack["db"], "no-emb")
        stats = stack["sweep"].run_batch()
        assert stats["findings"].get("missing_embedding") == 1
        row = stack["db"].execute(
            "SELECT * FROM sweep_findings WHERE memory_id = 'no-emb'"
        ).fetchone()
        assert row["finding_type"] == "missing_embedding"

    def test_near_duplicate_finding_dedupes_pair(self, stack):
        _insert_memory(stack["db"], "dup-a")
        _insert_memory(stack["db"], "dup-b")
        stack["emb"].store_cached("dup-a", _basis_vec(0))
        stack["emb"].store_cached("dup-b", _basis_vec(0))
        stats = stack["sweep"].run_batch()
        assert stats["findings"].get("near_duplicate", 0) >= 1
        rows = stack["db"].execute(
            "SELECT memory_id, detail FROM sweep_findings "
            "WHERE finding_type = 'near_duplicate'"
        ).fetchall()
        # Both directions collapse onto ONE row filed under the first id
        assert len(rows) == 1
        assert rows[0]["memory_id"] == "dup-a"
        assert rows[0]["detail"] == "dup-a|dup-b"

    def test_tag_drift_finding(self, stack):
        tid = _insert_tag(stack["db"], "physics")
        _insert_centroid(stack["db"], tid, _basis_vec(0))
        _insert_memory(stack["db"], "drifted")
        _assign_tag(stack["db"], "drifted", tid)
        stack["emb"].store_cached("drifted", _basis_vec(1))  # orthogonal
        stats = stack["sweep"].run_batch()
        assert stats["findings"].get("tag_drift") == 1
        # Shift the centroid slightly and re-sweep: the stable detail key
        # must dedupe onto the SAME open finding, not accumulate a new row
        _insert_centroid(stack["db"], tid, _basis_vec(2))
        stack["sweep"].run_batch()
        rows = stack["db"].execute(
            "SELECT COUNT(*) AS n FROM sweep_findings "
            "WHERE memory_id = 'drifted' AND finding_type = 'tag_drift'"
        ).fetchone()
        assert rows["n"] == 1

    def test_progress_counts(self, stack):
        _insert_memory(stack["db"], "p1")
        stack["emb"].store_cached("p1", _basis_vec(0))
        stack["sweep"].run_batch()
        progress = stack["sweep"].progress()
        assert progress["total_memories"] == 1
        assert progress["swept_memories"] == 1


# ── SubjectRoster ───────────────────────────────────────────────────────


def _edge(db, a: int, b: int, strength: float = 0.5) -> None:
    lo, hi = min(a, b), max(a, b)
    db.execute(
        "INSERT OR REPLACE INTO centroid_edges "
        "(tag_a_id, tag_b_id, edge_strength, co_retrieval_count) VALUES (?, ?, ?, 1)",
        (lo, hi, strength),
    )


class TestSubjectRoster:
    def _seed_cluster(self, stack, names: list[str], base: int) -> list[int]:
        ids = []
        for i, name in enumerate(names):
            tid = _insert_tag(stack["db"], name)
            _insert_centroid(stack["db"], tid, _basis_vec(base), memory_count=10 - i)
            ids.append(tid)
        for i in range(len(ids) - 1):
            _edge(stack["db"], ids[i], ids[i + 1])
        return ids

    def test_rebuild_creates_named_subjects(self, stack):
        self._seed_cluster(stack, ["physics", "quantum", "energy"], base=0)
        self._seed_cluster(stack, ["cooking", "recipes"], base=1)
        stats = stack["roster"].rebuild()
        assert stats["subjects_active"] == 2
        subjects = stack["roster"].list_subjects()
        names = {s["name"] for s in subjects}
        assert "physics+quantum+energy" in names
        assert "cooking+recipes" in names
        phys = next(s for s in subjects if s["name"].startswith("physics"))
        assert len(phys["member_tag_ids"]) == 3

    def test_small_clusters_excluded(self, stack):
        tid = _insert_tag(stack["db"], "loner")
        _insert_centroid(stack["db"], tid, _basis_vec(0))
        stats = stack["roster"].rebuild()
        assert stats["subjects_active"] == 0

    def test_temporal_tags_excluded(self, stack):
        ids = []
        for name in ["day-2026-01-01", "day-2026-01-02"]:
            tid = _insert_tag(stack["db"], name)
            _insert_centroid(stack["db"], tid, _basis_vec(0))
            ids.append(tid)
        _edge(stack["db"], ids[0], ids[1])
        stats = stack["roster"].rebuild()
        assert stats["subjects_active"] == 0

    def test_vanished_cluster_retired_not_deleted(self, stack):
        ids = self._seed_cluster(stack, ["physics", "quantum"], base=0)
        stack["roster"].rebuild()
        stack["db"].execute("DELETE FROM centroid_edges")
        stats = stack["roster"].rebuild()
        assert stats["subjects_retired"] == 1
        retired = stack["roster"].list_subjects(status="retired")
        assert len(retired) == 1
        assert retired[0]["name"] == "physics+quantum"

    def test_subject_centroid_is_unit(self, stack):
        self._seed_cluster(stack, ["physics", "quantum"], base=0)
        stack["roster"].rebuild()
        row = stack["db"].execute(
            "SELECT centroid FROM subjects WHERE status = 'active'"
        ).fetchone()
        vec = np.frombuffer(row["centroid"], dtype=np.float32)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


# ── Provenance + quarantine through MemoryStore ─────────────────────────


@pytest.fixture
def store_stack(stack):
    salience = SalienceScorer(stack["config"], stack["db"])
    store = MemoryStore(
        stack["config"], stack["db"], stack["emb"], stack["tags"], salience,
    )
    return {**stack, "store": store}


class TestProvenanceAndQuarantine:
    def test_author_origin_roundtrip(self, store_stack):
        mem = store_stack["store"].store(
            content="the sky is blue",
            tags=["sky"],
            author="claude",
            origin="session-42",
            skip_embedding=True,
        )
        assert mem.author == "claude"
        assert mem.origin == "session-42"
        loaded = store_stack["store"].get_by_id(mem.id)
        assert loaded.author == "claude"
        assert loaded.origin == "session-42"
        assert loaded.ingestion_tier == "critical"

    def test_quarantined_excluded_from_tag_retrieval(self, store_stack):
        good = store_stack["store"].store(
            content="on-topic fact", tags=["physics"], skip_embedding=True,
        )
        bad = store_stack["store"].store(
            content="suspicious payload", tags=["physics"],
            ingestion_tier="quarantine", skip_embedding=True,
        )
        results = store_stack["store"].retrieve_by_tags(["physics"])
        ids = {m.id for m in results}
        assert good.id in ids
        assert bad.id not in ids

    def test_quarantined_excluded_from_semantic_retrieval(self, store_stack):
        vec = _basis_vec(0)
        good = store_stack["store"].store(
            content="on-topic", tags=["physics"],
            precomputed_embeddings=[vec],
        )
        bad = store_stack["store"].store(
            content="off-topic", tags=["physics"],
            ingestion_tier="quarantine", precomputed_embeddings=[vec],
        )
        results = store_stack["store"].retrieve_semantic(
            "anything", top_k=10, query_vec=vec,
        )
        ids = {m.id for m, _ in results}
        assert good.id in ids
        assert bad.id not in ids


# ---- Front desk strata (custody, FAQ, commissions, demand) ----

from datetime import datetime as _dt

from chicory.layer4.librarian import (
    CommissionDesk,
    DemandLedger,
    FaqDesk,
    record_custody_transition,
)
from chicory.models.memory import Memory


def _mem(mid: str) -> Memory:
    now = _dt.utcnow()
    return Memory(
        id=mid, content=f"content of {mid}", summary=f"summary of {mid}",
        created_at=now, updated_at=now, source_model="test",
    )


class _StubStore:
    """Stub memory_store: DemandLedger only calls retrieve_hybrid."""

    def __init__(self, results):
        self.results = results

    def retrieve_hybrid(self, query, tags=None, top_k=None):
        return self.results


class TestCustodyTransitions:
    def test_transition_writes_event_and_noops_on_same(self, stack):
        _insert_memory(stack["db"], "c1")
        assert record_custody_transition(stack["db"], "c1", "clean", "sweep", "x")
        assert not record_custody_transition(stack["db"], "c1", "clean", "sweep", "x")
        rows = stack["db"].execute(
            "SELECT * FROM custody_events WHERE memory_id = 'c1'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["from_status"] == "unswept"
        assert rows[0]["to_status"] == "clean"
        assert rows[0]["actor"] == "sweep"

    def test_sweep_sets_clean_and_flagged(self, stack):
        _insert_memory(stack["db"], "ok")
        stack["emb"].store_cached("ok", _basis_vec(0))
        _insert_memory(stack["db"], "bad")  # no embedding -> finding
        stack["sweep"].run_batch()
        statuses = {
            r["id"]: r["custody_status"]
            for r in stack["db"].execute(
                "SELECT id, custody_status FROM memories"
            ).fetchall()
        }
        assert statuses["ok"] == "clean"
        assert statuses["bad"] == "flagged"

    def test_sweep_leaves_quarantined_and_certified_alone(self, stack):
        _insert_memory(stack["db"], "q1", tier="quarantine")
        stack["db"].execute(
            "UPDATE memories SET custody_status = 'quarantined' WHERE id = 'q1'"
        )
        _insert_memory(stack["db"], "cert")
        stack["emb"].store_cached("cert", _basis_vec(0))
        stack["db"].execute(
            "UPDATE memories SET custody_status = 'certified' WHERE id = 'cert'"
        )
        stack["sweep"].run_batch()
        statuses = {
            r["id"]: r["custody_status"]
            for r in stack["db"].execute(
                "SELECT id, custody_status FROM memories"
            ).fetchall()
        }
        assert statuses["q1"] == "quarantined"
        assert statuses["cert"] == "certified"  # clean pass keeps certification


class TestFaqDesk:
    def _desk(self, stack):
        return FaqDesk(stack["config"], stack["db"])

    def _seed_supporting(self, stack):
        _insert_memory(stack["db"], "s1")
        _insert_memory(stack["db"], "s2")

    def test_candidate_not_served_until_approved(self, stack):
        desk = self._desk(stack)
        self._seed_supporting(stack)
        fid = desk.create_candidate(
            "what is physics", _basis_vec(0), "draft", ["s1", "s2"], [],
        )
        assert desk.serve(_basis_vec(0)) is None
        assert desk.approve(fid, answer="Physics is the shelf at basis 0.")
        hit = desk.serve(_basis_vec(0))
        assert hit is not None
        assert hit["answer"] == "Physics is the shelf at basis 0."
        assert hit["supporting_memory_ids"] == ["s1", "s2"]
        row = stack["db"].execute(
            "SELECT hit_count FROM faq_entries WHERE id = ?", (fid,)
        ).fetchone()
        assert row["hit_count"] == 1

    def test_serve_requires_similarity(self, stack):
        desk = self._desk(stack)
        self._seed_supporting(stack)
        fid = desk.create_candidate("q", _basis_vec(0), "a", ["s1"], [])
        desk.approve(fid)
        assert desk.serve(_basis_vec(1)) is None  # orthogonal query

    def test_broken_supporting_goes_stale(self, stack):
        desk = self._desk(stack)
        self._seed_supporting(stack)
        fid = desk.create_candidate("q", _basis_vec(0), "a", ["s1", "s2"], [])
        desk.approve(fid)
        stack["db"].execute(
            "UPDATE memories SET ingestion_tier = 'quarantine' WHERE id = 's2'"
        )
        assert desk.serve(_basis_vec(0)) is None
        row = stack["db"].execute(
            "SELECT status FROM faq_entries WHERE id = ?", (fid,)
        ).fetchone()
        assert row["status"] == "stale"

    def test_retire(self, stack):
        desk = self._desk(stack)
        self._seed_supporting(stack)
        fid = desk.create_candidate("q", _basis_vec(0), "a", ["s1"], [])
        assert desk.retire(fid)
        assert not desk.approve(fid)  # retired stays retired


class TestCommissionDesk:
    def _desk(self, stack):
        return CommissionDesk(stack["config"], stack["db"])

    def test_file_and_lifecycle(self, stack):
        desk = self._desk(stack)
        r = desk.file("what about gaps", _basis_vec(0), source="manual")
        assert r["status"] == "filed"
        cid = r["commission_id"]
        assert desk.set_status(cid, "claimed")
        _insert_memory(stack["db"], "report-1")
        assert desk.set_status(cid, "reported", report_memory_id="report-1")
        assert desk.set_status(cid, "closed")
        rows = desk.list("closed")
        assert rows[0]["report_memory_id"] == "report-1"

    def test_file_dedupes_similar_open(self, stack):
        desk = self._desk(stack)
        r1 = desk.file("gap question", _basis_vec(0))
        r2 = desk.file("gap question again", _basis_vec(0))
        assert r2["status"] == "exists"
        assert r2["commission_id"] == r1["commission_id"]

    def test_match_open(self, stack):
        desk = self._desk(stack)
        desk.file("open gap", _basis_vec(0))
        hits = desk.match_open(_basis_vec(0))
        assert len(hits) == 1
        assert desk.match_open(_basis_vec(1)) == []


class TestDemandLedger:
    def _ledger(self, stack, results=None):
        faq = FaqDesk(stack["config"], stack["db"])
        commissions = CommissionDesk(stack["config"], stack["db"])
        store = _StubStore(results or [])
        return DemandLedger(
            stack["config"], stack["db"], store, faq, commissions, stack["roster"],
        ), faq, commissions

    def test_record_and_cluster(self, stack):
        ledger, _, _ = self._ledger(stack)
        for _i in range(3):
            ledger.record("same question", _basis_vec(0), 5, 0.2)
        ledger.record("different question", _basis_vec(1), 5, 0.2)
        stats = ledger.process_batch(allow_retrieval=False)
        assert stats["processed"] == 4
        clusters = stack["db"].execute(
            "SELECT hit_count FROM query_clusters ORDER BY hit_count DESC"
        ).fetchall()
        assert len(clusters) == 2
        assert clusters[0]["hit_count"] == 3

    def test_recurring_good_promotes_faq_candidate(self, stack):
        _insert_memory(stack["db"], "sup1")
        _insert_memory(stack["db"], "sup2")
        ledger, faq, _ = self._ledger(
            stack, results=[(_mem("sup1"), 0.9), (_mem("sup2"), 0.8)],
        )
        for _i in range(3):
            ledger.record("popular question", _basis_vec(0), 5, 0.8)
        stats = ledger.process_batch(allow_retrieval=True)
        assert stats["faq_candidates"] == 1
        cands = faq.candidates()
        assert len(cands) == 1
        assert cands[0]["question"] == "popular question"
        assert cands[0]["supporting_memory_ids"] == ["sup1", "sup2"]
        # Re-processing similar demand does not duplicate the candidate
        for _i in range(3):
            ledger.record("popular question", _basis_vec(0), 5, 0.8)
        stats2 = ledger.process_batch(allow_retrieval=True)
        assert stats2["faq_candidates"] == 0

    def test_recurring_weak_files_commission(self, stack):
        ledger, _, commissions = self._ledger(stack)
        for _i in range(3):
            ledger.record("unanswerable question", _basis_vec(2), 0, 0.05)
        stats = ledger.process_batch(allow_retrieval=False)
        assert stats["commissions_filed"] == 1
        open_rows = commissions.list("open")
        assert open_rows[0]["question"] == "unanswerable question"
        assert open_rows[0]["source"] == "recurring-weak"


# ---- Review-fix regression tests (round 2) ----


class TestReviewFixes:
    def test_retrieve_semantic_sets_last_query_vec(self, store_stack):
        vec = _basis_vec(0)
        store_stack["store"].store(
            content="anchor", tags=["physics"], precomputed_embeddings=[vec],
        )
        store_stack["store"].retrieve_semantic("anything", top_k=3, query_vec=vec)
        got = getattr(store_stack["store"], "_last_query_vec", None)
        assert got is not None
        assert float(np.dot(got, vec)) > 0.99

    def test_cross_batch_near_dup_flags_both(self, stack):
        # 'aaa' swept long ago and clean; duplicate 'zzz' arrives later.
        _insert_memory(stack["db"], "aaa")
        stack["emb"].store_cached("aaa", _basis_vec(0))
        stack["db"].execute(
            "UPDATE memories SET swept_at = '2026-01-01T00:00:00', "
            "custody_status = 'clean' WHERE id = 'aaa'"
        )
        _insert_memory(stack["db"], "zzz")
        stack["emb"].store_cached("zzz", _basis_vec(0))
        # Batch of 1 sweeps zzz (NULL swept_at sorts first)
        stack["sweep"].run_batch(batch_size=1)
        statuses = {
            r["id"]: r["custody_status"]
            for r in stack["db"].execute(
                "SELECT id, custody_status FROM memories"
            ).fetchall()
        }
        assert statuses["zzz"] == "flagged"
        assert statuses["aaa"] == "flagged"  # out-of-batch partner flagged too

    def test_missing_embedding_auto_resolves_and_reopens(self, stack):
        _insert_memory(stack["db"], "late")
        stack["sweep"].run_batch()
        row = stack["db"].execute(
            "SELECT resolved FROM sweep_findings WHERE memory_id = 'late'"
        ).fetchone()
        assert row["resolved"] == 0
        # Embedding arrives (phase-2 style); re-sweep auto-resolves
        stack["emb"].store_cached("late", _basis_vec(0))
        stack["db"].execute("UPDATE memories SET swept_at = NULL WHERE id = 'late'")
        stack["sweep"].run_batch()
        row = stack["db"].execute(
            "SELECT resolved FROM sweep_findings WHERE memory_id = 'late'"
        ).fetchone()
        assert row["resolved"] == 1
        status = stack["db"].execute(
            "SELECT custody_status FROM memories WHERE id = 'late'"
        ).fetchone()["custody_status"]
        assert status == "clean"
        # Condition returns: embedding removed -> resolved row reopens
        stack["db"].execute("DELETE FROM embeddings WHERE memory_id = 'late'")
        stack["db"].execute("UPDATE memories SET swept_at = NULL WHERE id = 'late'")
        stack["sweep"].run_batch()
        row = stack["db"].execute(
            "SELECT resolved FROM sweep_findings WHERE memory_id = 'late'"
        ).fetchone()
        assert row["resolved"] == 0

    def test_tag_drift_auto_resolves_when_reshelved(self, stack):
        tid = _insert_tag(stack["db"], "physics")
        _insert_centroid(stack["db"], tid, _basis_vec(0))
        _insert_memory(stack["db"], "drifter")
        _assign_tag(stack["db"], "drifter", tid)
        stack["emb"].store_cached("drifter", _basis_vec(1))
        stack["sweep"].run_batch()
        assert stack["db"].execute(
            "SELECT resolved FROM sweep_findings WHERE memory_id = 'drifter'"
        ).fetchone()["resolved"] == 0
        # Centroid moves to match the memory: drift conclusively passes
        _insert_centroid(stack["db"], tid, _basis_vec(1))
        stack["db"].execute("UPDATE memories SET swept_at = NULL WHERE id = 'drifter'")
        stack["sweep"].run_batch()
        assert stack["db"].execute(
            "SELECT resolved FROM sweep_findings WHERE memory_id = 'drifter'"
        ).fetchone()["resolved"] == 1

    def test_stale_faq_refreshes_instead_of_blocking(self, stack):
        _insert_memory(stack["db"], "old1")
        _insert_memory(stack["db"], "new1")
        faq = FaqDesk(stack["config"], stack["db"])
        commissions = CommissionDesk(stack["config"], stack["db"])
        ledger = DemandLedger(
            stack["config"], stack["db"],
            _StubStore([(_mem("new1"), 0.9)]),
            faq, commissions, stack["roster"],
        )
        fid = faq.create_candidate("q", _basis_vec(0), "old draft", ["old1"], [])
        faq.approve(fid)
        stack["db"].execute(
            "UPDATE memories SET ingestion_tier = 'quarantine' WHERE id = 'old1'"
        )
        assert faq.serve(_basis_vec(0)) is None  # marks stale
        for _i in range(3):
            ledger.record("q again", _basis_vec(0), 5, 0.8)
        stats = ledger.process_batch(allow_retrieval=True)
        assert stats["faq_candidates"] == 1
        row = stack["db"].execute(
            "SELECT status, supporting_memory_ids FROM faq_entries WHERE id = ?",
            (fid,),
        ).fetchone()
        assert row["status"] == "candidate"  # refreshed for re-judgment
        assert json.loads(row["supporting_memory_ids"]) == ["new1"]

    def test_deferred_promotion_retries_next_batch(self, stack):
        _insert_memory(stack["db"], "sup1")
        faq = FaqDesk(stack["config"], stack["db"])
        commissions = CommissionDesk(stack["config"], stack["db"])
        ledger = DemandLedger(
            stack["config"], stack["db"],
            _StubStore([(_mem("sup1"), 0.9)]),
            faq, commissions, stack["roster"],
        )
        for _i in range(3):
            ledger.record("popular", _basis_vec(0), 5, 0.8)
        stats = ledger.process_batch(allow_retrieval=False)  # index not built
        assert stats["faq_candidates"] == 0
        ledger.record("popular", _basis_vec(0), 5, 0.8)
        stats2 = ledger.process_batch(allow_retrieval=True)
        assert stats2["faq_candidates"] == 1  # retried, not lost

    def test_closed_commission_not_refiled_by_demand(self, stack):
        faq = FaqDesk(stack["config"], stack["db"])
        commissions = CommissionDesk(stack["config"], stack["db"])
        ledger = DemandLedger(
            stack["config"], stack["db"], _StubStore([]),
            faq, commissions, stack["roster"],
        )
        for _i in range(3):
            ledger.record("gap", _basis_vec(2), 0, 0.05)
        stats = ledger.process_batch(allow_retrieval=False)
        assert stats["commissions_filed"] == 1
        cid = commissions.list("open")[0]["commission_id"]
        commissions.set_status(cid, "closed")
        for _i in range(3):
            ledger.record("gap", _basis_vec(2), 0, 0.05)
        stats2 = ledger.process_batch(allow_retrieval=False)
        assert stats2["commissions_filed"] == 0  # closed ticket stays closed

    def test_query_history_retention_prunes_old_rows(self, stack):
        faq = FaqDesk(stack["config"], stack["db"])
        commissions = CommissionDesk(stack["config"], stack["db"])
        ledger = DemandLedger(
            stack["config"], stack["db"], _StubStore([]),
            faq, commissions, stack["roster"],
        )
        ledger.record("ancient", _basis_vec(0), 5, 0.5)
        ledger.process_batch(allow_retrieval=False)
        stack["db"].execute(
            "UPDATE query_history SET created_at = '2020-01-01T00:00:00'"
        )
        ledger.record("fresh", _basis_vec(1), 5, 0.5)
        ledger.process_batch(allow_retrieval=False)
        texts = {
            r["query_text"] for r in stack["db"].execute(
                "SELECT query_text FROM query_history"
            ).fetchall()
        }
        assert "ancient" not in texts
        assert "fresh" in texts


# ---- Helper strata: penumbra ledger + analogic recall geometry ----

from chicory.layer4.librarian import (
    PenumbraLedger,
    analogic_target,
    parallelness,
)


@pytest.fixture
def penumbra(stack):
    ledger = PenumbraLedger(stack["config"], stack["db"])
    return ledger


class TestPenumbraLedger:
    def test_record_upserts_counts(self, stack, penumbra):
        for mid in ("a", "b", "c"):
            _insert_memory(stack["db"], mid)
        penumbra.record([("a", 0.4), ("b", 0.3)], ["c"])
        penumbra.record([("a", 0.5)], ["c", "a"])
        rows = {
            r["memory_id"]: (r["near_count"], r["retrieved_count"])
            for r in stack["db"].execute(
                "SELECT * FROM retrieval_penumbra"
            ).fetchall()
        }
        assert rows["a"] == (2, 1)   # near twice, retrieved once
        assert rows["b"] == (1, 0)
        assert rows["c"] == (0, 2)

    def test_buried_and_rescued(self, stack, penumbra):
        stack["config"].penumbra_buried_min_near = 3
        _insert_memory(stack["db"], "lost")
        _insert_memory(stack["db"], "seen")
        for _i in range(3):
            penumbra.record([("lost", 0.4)], [])
        penumbra.record([("seen", 0.4)], [])
        penumbra.record([], ["seen"])
        assert penumbra.buried_candidates(["lost", "seen"]) == {"lost"}
        assert penumbra.rescued(["lost", "seen"]) == {"seen"}

    def test_sweep_files_buried_without_custody_flag(self, stack):
        stack["config"].penumbra_buried_min_near = 2
        ledger = PenumbraLedger(stack["config"], stack["db"])
        sweep = LinearSweep(
            stack["config"], stack["db"], stack["emb"],
            stack["tags"], stack["centroids"], penumbra=ledger,
        )
        _insert_memory(stack["db"], "hidden")
        stack["emb"].store_cached("hidden", _basis_vec(0))
        for _i in range(2):
            ledger.record([("hidden", 0.4)], [])
        sweep.run_batch()
        finding = stack["db"].execute(
            "SELECT * FROM sweep_findings WHERE memory_id = 'hidden' "
            "AND finding_type = 'buried'"
        ).fetchone()
        assert finding is not None and finding["resolved"] == 0
        # Buried is a service finding, not an integrity flag: custody stays clean
        status = stack["db"].execute(
            "SELECT custody_status FROM memories WHERE id = 'hidden'"
        ).fetchone()["custody_status"]
        assert status == "clean"
        # Once actually retrieved, the next sweep auto-resolves it
        ledger.record([], ["hidden"])
        stack["db"].execute("UPDATE memories SET swept_at = NULL WHERE id = 'hidden'")
        sweep.run_batch()
        finding = stack["db"].execute(
            "SELECT resolved FROM sweep_findings WHERE memory_id = 'hidden' "
            "AND finding_type = 'buried'"
        ).fetchone()
        assert finding["resolved"] == 1


class TestAnalogicGeometry:
    def test_target_transports_offset(self):
        home = _basis_vec(0)
        other = _basis_vec(1)
        # Query displaced from home along e2
        query = home + 0.5 * _basis_vec(2)
        target = analogic_target(home, query, other)
        # Target should lean along e1 (other's frame) plus the e2 offset
        assert target[1] > 0.5
        assert target[2] > 0.1
        assert abs(np.linalg.norm(target) - 1.0) < 1e-5

    def test_parallelness_scores_matching_role_high(self):
        home, other = _basis_vec(0), _basis_vec(1)
        query = _unit_np(home + 0.5 * _basis_vec(2))
        # Counterpart with the SAME offset in the other frame
        good = _unit_np(other + 0.5 * _basis_vec(2))
        # Counterpart with an orthogonal offset
        bad = _unit_np(other + 0.5 * _basis_vec(3))
        assert parallelness(good, other, home, query) > 0.8
        assert parallelness(bad, other, home, query) < 0.3

    def test_parallelness_zero_offset_is_zero(self):
        home, other = _basis_vec(0), _basis_vec(1)
        assert parallelness(other, other, home, home) == 0.0


def _unit_np(v):
    return v / np.linalg.norm(v)


# ---- Review-fix regression tests (round 3) ----


class TestHelperReviewFixes:
    def test_penumbra_record_filters_quarantined_and_archived(self, stack, penumbra):
        _insert_memory(stack["db"], "live")
        _insert_memory(stack["db"], "held", tier="quarantine")
        _insert_memory(stack["db"], "gone")
        stack["db"].execute("UPDATE memories SET is_archived = 1 WHERE id = 'gone'")
        penumbra.record([("live", 0.4), ("held", 0.4), ("gone", 0.4)], [])
        rows = {
            r["memory_id"] for r in stack["db"].execute(
                "SELECT memory_id FROM retrieval_penumbra"
            ).fetchall()
        }
        assert rows == {"live"}

    def test_sweep_never_files_buried_for_quarantined(self, stack):
        stack["config"].penumbra_buried_min_near = 1
        ledger = PenumbraLedger(stack["config"], stack["db"])
        sweep = LinearSweep(
            stack["config"], stack["db"], stack["emb"],
            stack["tags"], stack["centroids"], penumbra=ledger,
        )
        _insert_memory(stack["db"], "heldq", tier="quarantine")
        stack["emb"].store_cached("heldq", _basis_vec(0))
        # Force a penumbra row past the record-time filter to prove the
        # sweep-side guard holds on its own
        stack["db"].execute(
            "INSERT INTO retrieval_penumbra (memory_id, near_count, retrieved_count) "
            "VALUES ('heldq', 5, 0)"
        )
        sweep.run_batch()
        finding = stack["db"].execute(
            "SELECT * FROM sweep_findings WHERE memory_id = 'heldq' "
            "AND finding_type = 'buried'"
        ).fetchone()
        assert finding is None


# ---- Roster granularity: hub-tag rule + centroid-space blob splitting ----


class TestRosterGranularity:
    def test_hub_tags_excluded_from_subjects(self, stack):
        stack["config"].subject_hub_tag_max_fraction = 0.5
        # 10 live memories; 'document' sits on 9 of them (hub), topic tags on 2
        for i in range(10):
            _insert_memory(stack["db"], f"hm{i}")
        doc = _insert_tag(stack["db"], "document")
        _insert_centroid(stack["db"], doc, _basis_vec(0), memory_count=9)
        a = _insert_tag(stack["db"], "alchemy")
        b = _insert_tag(stack["db"], "hermetics")
        _insert_centroid(stack["db"], a, _basis_vec(1), memory_count=2)
        _insert_centroid(stack["db"], b, _basis_vec(1), memory_count=2)
        _edge(stack["db"], doc, a)
        _edge(stack["db"], doc, b)
        _edge(stack["db"], a, b)
        stack["roster"].rebuild()
        subjects = stack["roster"].list_subjects()
        assert len(subjects) == 1
        member_ids = set(subjects[0]["member_tag_ids"])
        assert doc not in member_ids  # hub tag filtered out
        assert member_ids == {a, b}

    def test_giant_component_splits_in_centroid_space(self, stack):
        # Two geometrically distinct tag groups welded into ONE component
        # by co-retrieval edges — components alone would make one blob.
        stack["config"].subject_max_component_fraction = 0.3
        stack["config"].subject_split_k = 2
        group_a, group_b = [], []
        for i in range(4):
            ta = _insert_tag(stack["db"], f"phys-{i}")
            _insert_centroid(stack["db"], ta, _basis_vec(0), memory_count=5)
            group_a.append(ta)
            tb = _insert_tag(stack["db"], f"cook-{i}")
            _insert_centroid(stack["db"], tb, _basis_vec(1), memory_count=5)
            group_b.append(tb)
        all_tags = group_a + group_b
        # Chain everything into one component, including a cross-group weld
        for x, y in zip(all_tags, all_tags[1:]):
            _edge(stack["db"], x, y)
        stats = stack["roster"].rebuild()
        # Hierarchy semantics: the parent is KEPT and the split becomes its
        # children (1 root + 2 children), rather than replacing it.
        assert stats["subjects_active"] == 3
        subjects = stack["roster"].list_subjects()
        child_sets = [
            set(s["member_tag_ids"]) for s in subjects if s["depth"] == 1
        ]
        assert set(group_a) in child_sets
        assert set(group_b) in child_sets
        root = [s for s in subjects if s["depth"] == 0][0]
        assert set(root["member_tag_ids"]) == set(group_a) | set(group_b)


# ---- Subject hierarchy (parent_id / depth / path) ----


class TestSubjectHierarchy:
    def _cluster(self, stack, names, base, memories_per_tag=0):
        """Seed a tag group sharing a centroid direction."""
        ids = []
        for i, nm in enumerate(names):
            tid = _insert_tag(stack["db"], nm)
            _insert_centroid(stack["db"], tid, _basis_vec(base), memory_count=10 - i)
            ids.append(tid)
            for j in range(memories_per_tag):
                mid = f"m-{nm}-{j}"
                _insert_memory(stack["db"], mid)
                _assign_tag(stack["db"], mid, tid)
        for a, b in zip(ids, ids[1:]):
            _edge(stack["db"], a, b)
        return ids

    def test_small_roster_stays_flat(self, stack):
        self._cluster(stack, ["alpha", "beta", "gamma"], base=0)
        stack["roster"].rebuild()
        nodes = stack["roster"].tree()
        assert len(nodes) == 1
        assert nodes[0]["depth"] == 0
        assert nodes[0]["parent_id"] is None
        assert nodes[0]["path"] == nodes[0]["name"]

    def test_oversized_subject_splits_into_children(self, stack):
        # Two geometrically distinct groups welded into one component
        stack["config"].subject_max_component_fraction = 0.3
        stack["config"].subject_split_k = 2
        stack["config"].subject_max_depth = 2
        a = self._cluster(stack, [f"phys-{i}" for i in range(4)], base=0)
        b = self._cluster(stack, [f"cook-{i}" for i in range(4)], base=1)
        _edge(stack["db"], a[-1], b[0])  # weld them
        stats = stack["roster"].rebuild()
        assert stats["max_depth"] >= 1
        nodes = stack["roster"].tree()
        roots = [n for n in nodes if n["depth"] == 0]
        kids = [n for n in nodes if n["depth"] == 1]
        # The parent is KEPT, not replaced by its children
        assert len(roots) == 1
        assert len(kids) == 2
        assert all(k["parent_id"] == roots[0]["id"] for k in kids)
        assert all(k["path"].startswith(roots[0]["path"] + "/") for k in kids)
        # Children partition the parent's tags
        parent_tags = set(roots[0]["member_tag_ids"])
        child_tags = set()
        for k in kids:
            child_tags |= set(k["member_tag_ids"])
        assert child_tags == parent_tags

    def test_memory_share_triggers_a_split_that_tag_count_would_miss(self, stack):
        # Few tags, but they carry most of the library — the defect the
        # tag-fraction cap alone could never catch.
        stack["config"].subject_max_component_fraction = 0.99  # tag cap off
        stack["config"].subject_max_memory_fraction = 0.20
        stack["config"].subject_split_k = 2
        stack["config"].subject_max_depth = 2
        a = self._cluster(stack, ["big-a", "big-b"], base=0, memories_per_tag=6)
        b = self._cluster(stack, ["big-c", "big-d"], base=1, memories_per_tag=6)
        _edge(stack["db"], a[-1], b[0])
        stats = stack["roster"].rebuild()
        assert stats["max_depth"] >= 1, "memory-share criterion did not fire"

    def test_depth_is_bounded(self, stack):
        stack["config"].subject_max_component_fraction = 0.01  # always "too big"
        stack["config"].subject_max_depth = 1
        stack["config"].subject_split_k = 2
        a = self._cluster(stack, [f"t{i}" for i in range(8)], base=0)
        _edge(stack["db"], a[0], a[-1])
        stats = stack["roster"].rebuild()
        assert stats["max_depth"] <= 1
        assert all(n["depth"] <= 1 for n in stack["roster"].tree())

    def test_routing_stays_coarse_while_teaching_goes_deep(self, stack):
        stack["config"].subject_max_component_fraction = 0.3
        stack["config"].subject_split_k = 2
        stack["config"].subject_max_depth = 2
        a = self._cluster(stack, [f"phys-{i}" for i in range(4)], base=0)
        b = self._cluster(stack, [f"cook-{i}" for i in range(4)], base=1)
        _edge(stack["db"], a[-1], b[0])
        stack["roster"].rebuild()
        # Routing at depth 0 sees only the root
        coarse = stack["roster"].match(_basis_vec(0), threshold=-1.0, top_m=10,
                                       max_depth=0)
        assert all(s["depth"] == 0 for s, _ in coarse)
        # Unrestricted match sees the children too
        deep = stack["roster"].match(_basis_vec(0), threshold=-1.0, top_m=10)
        assert any(s["depth"] == 1 for s, _ in deep)

    def test_addressing_by_path_and_children(self, stack):
        stack["config"].subject_max_component_fraction = 0.3
        stack["config"].subject_split_k = 2
        stack["config"].subject_max_depth = 2
        a = self._cluster(stack, [f"phys-{i}" for i in range(4)], base=0)
        b = self._cluster(stack, [f"cook-{i}" for i in range(4)], base=1)
        _edge(stack["db"], a[-1], b[0])
        stack["roster"].rebuild()
        root = [n for n in stack["roster"].tree() if n["depth"] == 0][0]
        assert stack["roster"].get_by_path(root["path"])["id"] == root["id"]
        # A bare name resolves too
        assert stack["roster"].get_by_path(root["name"])["id"] == root["id"]
        kids = stack["roster"].children_of(root["path"])
        assert len(kids) == 2
        # And a child is addressable by its full path
        child = stack["roster"].get_by_path(kids[0]["path"])
        assert child is not None and child["depth"] == 1
        assert stack["roster"].get_by_path("no/such/path") is None

    def test_vanished_subtree_is_retired(self, stack):
        stack["config"].subject_max_component_fraction = 0.3
        stack["config"].subject_split_k = 2
        stack["config"].subject_max_depth = 2
        a = self._cluster(stack, [f"phys-{i}" for i in range(4)], base=0)
        b = self._cluster(stack, [f"cook-{i}" for i in range(4)], base=1)
        _edge(stack["db"], a[-1], b[0])
        stack["roster"].rebuild()
        before = len(stack["roster"].tree())
        assert before >= 3
        stack["db"].execute("DELETE FROM centroid_edges")
        stack["roster"].rebuild()
        assert stack["roster"].tree() == []
        retired = stack["roster"].list_subjects(status="retired", limit=50)
        assert len(retired) == before

    def test_auto_route_depth_skips_a_lone_root(self, stack):
        stack["config"].subject_max_component_fraction = 0.3
        stack["config"].subject_split_k = 2
        stack["config"].subject_max_depth = 2
        stack["config"].consult_route_depth = -1
        a = self._cluster(stack, [f"phys-{i}" for i in range(4)], base=0)
        b = self._cluster(stack, [f"cook-{i}" for i in range(4)], base=1)
        _edge(stack["db"], a[-1], b[0])
        stack["roster"].rebuild()
        # One root spanning everything is not a routing choice; auto
        # descends to the first level that offers one.
        assert stack["roster"].route_depth() == 1
        stack["config"].consult_route_depth = 0
        assert stack["roster"].route_depth() == 0  # explicit config wins

    def test_routing_frontier_descends_past_oversized_nodes_only(self, stack):
        # The frontier rule: route to a node when it is under the size cap
        # and its parent was over it.  Depth is irrelevant — an unbalanced
        # tree needs the boundary drawn by size, not by level.
        stack["config"].subject_max_component_fraction = 0.3
        stack["config"].subject_split_k = 2
        stack["config"].subject_max_depth = 2
        stack["config"].subject_max_memory_fraction = 0.5
        a = self._cluster(stack, [f"phys-{i}" for i in range(4)], base=0,
                          memories_per_tag=3)
        b = self._cluster(stack, [f"cook-{i}" for i in range(4)], base=1,
                          memories_per_tag=3)
        _edge(stack["db"], a[-1], b[0])
        stack["roster"].rebuild()

        routed = stack["roster"].match(
            _basis_vec(0), threshold=-1.0, top_m=20, max_depth=1,
        )
        paths = {s["path"] for s, _ in routed}
        cap = stack["roster"]._routing_cap()
        nodes = {n["path"]: n for n in stack["roster"].tree()}

        # Every routed node is under the cap (or a leaf)
        for p in paths:
            n = nodes[p]
            assert (
                n["memory_count"] <= cap
                or not stack["roster"].children_of(p)
            ), f"routed an oversized internal node: {p}"
        # The oversized root is excluded; its children carry the routing
        root = [n for n in nodes.values() if n["depth"] == 0][0]
        assert root["memory_count"] > cap
        assert root["path"] not in paths
        assert len(paths) == 2

    def test_routing_stops_at_a_node_small_enough_to_choose(self, stack):
        # Inverse case: when the parent is already under the cap, routing
        # stops there rather than descending — no redundant children.
        stack["config"].subject_max_component_fraction = 0.3
        stack["config"].subject_split_k = 2
        stack["config"].subject_max_depth = 2
        stack["config"].subject_max_memory_fraction = 1.0  # nothing is oversized
        a = self._cluster(stack, [f"phys-{i}" for i in range(4)], base=0,
                          memories_per_tag=2)
        b = self._cluster(stack, [f"cook-{i}" for i in range(4)], base=1,
                          memories_per_tag=2)
        _edge(stack["db"], a[-1], b[0])
        stack["roster"].rebuild()
        routed = stack["roster"].match(
            _basis_vec(0), threshold=-1.0, top_m=20, max_depth=1,
        )
        paths = {s["path"] for s, _ in routed}
        assert len(paths) == 1
        assert "/" not in next(iter(paths)), "descended past a usable shelf"

    def test_children_partition_the_parent_even_with_tiny_splinters(self, stack):
        # A k-means group below min_cluster_size must be folded into its
        # nearest sibling, not dropped: dropped tags would live only on an
        # internal node, which the routing frontier never reaches.
        stack["config"].subject_max_component_fraction = 0.3
        stack["config"].subject_split_k = 3
        stack["config"].subject_max_depth = 2
        stack["config"].subject_min_cluster_size = 2
        a = self._cluster(stack, [f"phys-{i}" for i in range(4)], base=0)
        b = self._cluster(stack, [f"cook-{i}" for i in range(4)], base=1)
        lone = _insert_tag(stack["db"], "lone-outlier")
        _insert_centroid(stack["db"], lone, _basis_vec(9), memory_count=1)
        _edge(stack["db"], a[-1], b[0])
        _edge(stack["db"], b[-1], lone)
        stack["roster"].rebuild()
        root = [n for n in stack["roster"].tree() if n["depth"] == 0][0]
        kids = stack["roster"].children_of(root["path"])
        if kids:
            covered = set()
            for k in kids:
                covered |= set(k["member_tag_ids"])
            assert covered == set(root["member_tag_ids"]), "tags fell out of the tree"
            assert lone in covered

    def test_degenerate_split_is_refused(self, stack):
        # If the largest child is nearly the whole parent, the split
        # separated nothing — the node stays a leaf instead of spawning
        # a near-identity child (the X~2 / X~3 chains).
        stack["config"].subject_max_component_fraction = 0.01  # always "too big"
        stack["config"].subject_split_k = 2
        stack["config"].subject_max_depth = 3
        stack["config"].subject_min_cluster_size = 2
        stack["config"].subject_split_min_separation = 0.9
        # 10 tags all on the same centroid direction: k-means cannot
        # meaningfully separate them
        ids = self._cluster(stack, [f"same-{i}" for i in range(10)], base=0)
        _edge(stack["db"], ids[0], ids[-1])
        stack["roster"].rebuild()
        for n in stack["roster"].tree():
            assert "~" not in n["name"], f"near-identity child persisted: {n['name']}"

    def test_empty_eligible_tags_leaves_roster_alone(self, stack):
        self._cluster(stack, ["alpha", "beta", "gamma"], base=0)
        stack["roster"].rebuild()
        before = len(stack["roster"].tree())
        assert before >= 1
        # Hub rule now excludes everything
        stack["config"].subject_hub_tag_max_fraction = 0.0
        stats = stack["roster"].rebuild()
        assert stats["subjects_active"] == before
        assert "note" in stats
        assert len(stack["roster"].tree()) == before

    def test_chain_route_depth_falls_back_to_shallowest(self, stack):
        stack["config"].consult_route_depth = -1
        stack["db"].execute(
            "INSERT INTO subjects (name, member_tag_ids, memory_count, "
            "status, depth, path, parent_id) "
            "VALUES ('root', '[]', 10, 'active', 0, 'root', NULL)"
        )
        rid = stack["db"].execute(
            "SELECT id FROM subjects WHERE name='root'"
        ).fetchone()["id"]
        stack["db"].execute(
            "INSERT INTO subjects (name, member_tag_ids, memory_count, "
            "status, depth, path, parent_id) "
            "VALUES ('mid', '[]', 5, 'active', 1, 'root/mid', ?)", (rid,),
        )
        # A chain offers no choice at any level; the shallowest node is the
        # one that covers everything.
        assert stack["roster"].route_depth() == 0

    def test_degenerate_split_refused_on_memory_share_too(self, stack):
        # A child can shed tags while keeping nearly all the parent's
        # memories — that is not a split either.
        stack["config"].subject_max_component_fraction = 0.01
        stack["config"].subject_split_k = 2
        stack["config"].subject_max_depth = 3
        stack["config"].subject_min_cluster_size = 2
        stack["config"].subject_split_min_separation = 0.9
        # Four tags, geometrically separable, but all pointing at the same
        # memories — any split keeps 100% of the material on both sides.
        ids = []
        for i, nm in enumerate(["big-a", "big-b", "small-c", "small-d"]):
            tid = _insert_tag(stack["db"], nm)
            _insert_centroid(stack["db"], tid, _basis_vec(i), memory_count=5)
            ids.append(tid)
        for j in range(10):
            mid = f"shared-{j}"
            _insert_memory(stack["db"], mid)
            for tid in ids:
                _assign_tag(stack["db"], mid, tid)
        for x, y in zip(ids, ids[1:]):
            _edge(stack["db"], x, y)
        stack["roster"].rebuild()
        depths = {n["depth"] for n in stack["roster"].tree()}
        assert depths == {0}, "split despite identical memory coverage"


class TestFuzzyAddressing:
    def _roster(self, stack):
        # Two shelves, one nested, with distinctive member tags
        phys = self._named_cluster(stack, ["quantum", "relativity", "field"], 0)
        cook = self._named_cluster(stack, ["baking", "pastry", "sourdough"], 1)
        _edge(stack["db"], phys[-1], cook[0])
        stack["config"].subject_max_component_fraction = 0.3
        stack["config"].subject_split_k = 2
        stack["config"].subject_max_depth = 2
        stack["roster"].rebuild()

    def _named_cluster(self, stack, names, base):
        ids = []
        for i, nm in enumerate(names):
            tid = _insert_tag(stack["db"], nm)
            _insert_centroid(stack["db"], tid, _basis_vec(base), memory_count=10 - i)
            ids.append(tid)
        for a, b in zip(ids, ids[1:]):
            _edge(stack["db"], a, b)
        return ids

    def test_exact_path_and_name_still_work(self, stack):
        self._roster(stack)
        node = stack["roster"].tree()[0]
        assert stack["roster"].resolve(node["path"])["match"]["id"] == node["id"]
        assert stack["roster"].resolve(node["name"])["match"]["id"] == node["id"]

    def test_topic_resolves_to_the_most_specific_shelf(self, stack):
        self._roster(stack)
        res = stack["roster"].resolve("sourdough")
        assert "match" in res, res
        node = res["match"]
        # Deepest shelf holding the tag, not the library-sized ancestor
        children = stack["roster"].children_of(node["path"])
        assert not children or node["depth"] > 0
        tid = _insert_tag(stack["db"], "sourdough")
        assert tid in node["member_tag_ids"]

    def test_unknown_topic_returns_suggestions_not_an_error(self, stack):
        self._roster(stack)
        res = stack["roster"].resolve("underwater basket weaving")
        assert res.get("no_match") is True
        assert isinstance(res.get("suggestions"), list)

    def test_rename_survives_a_rebuild(self, stack):
        self._roster(stack)
        node = stack["roster"].tree()[0]
        renamed = stack["roster"].rename(node["path"], "The Whole Library")
        assert renamed["label"] == "The Whole Library"
        stack["roster"].rebuild()  # derived fields refresh...
        again = stack["roster"].get_by_path(node["path"])
        assert again["display_name"] == "The Whole Library"  # ...authored one stays
        assert again["label"] == "The Whole Library"

    def test_given_name_is_addressable(self, stack):
        self._roster(stack)
        node = stack["roster"].tree()[0]
        stack["roster"].rename(node["path"], "Everything Shelf")
        assert stack["roster"].resolve("everything shelf")["match"]["id"] == node["id"]
        # and a substring of it
        assert stack["roster"].resolve("Everything")["match"]["id"] == node["id"]

    def test_rename_can_be_cleared(self, stack):
        self._roster(stack)
        node = stack["roster"].tree()[0]
        stack["roster"].rename(node["path"], "Temp")
        stack["roster"].rename(node["path"], None)
        after = stack["roster"].get_by_path(node["path"])
        assert after["display_name"] is None
        assert after["label"] == after["name"]

    def test_rename_of_unknown_shelf_returns_none(self, stack):
        self._roster(stack)
        assert stack["roster"].rename("no-such-shelf-anywhere", "X") is None

    def test_spaces_match_hyphenated_tags(self, stack):
        self._roster(stack)
        tid = _insert_tag(stack["db"], "rocket-league")
        _insert_centroid(stack["db"], tid, _basis_vec(4), memory_count=5)
        base = stack["roster"].tree()[0]
        members = sorted(set(base["member_tag_ids"]) | {tid})
        stack["db"].execute(
            "UPDATE subjects SET member_tag_ids = ? WHERE id = ?",
            (json.dumps(members), base["id"]),
        )
        res = stack["roster"].resolve("rocket league")
        assert "match" in res, res
        assert tid in res["match"]["member_tag_ids"]
