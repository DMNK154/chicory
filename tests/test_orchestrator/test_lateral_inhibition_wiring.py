"""Orchestrator wiring for data-derived lateral inhibition.

What triggers a recompute (run_sync, the background growth gate), what
must not (store, ingest finalize), how a pre-empted write resumes, and
the cache invalidations and boot marker that went in alongside.  The
evidence is scenario A, inserted with direct SQL: 60 units {python,
asyncio, bug} and 40 units {javascript, promise, bug}, each also carrying
a month tag and two letters, plus rare units — four cross-language pairs
come out inhibitory.
"""

from __future__ import annotations

import json

import pytest

import chicory.layer3.lateral_inhibition as lateral_inhibition
import chicory.orchestrator.orchestrator as orchestrator_module
from chicory.config import ChicoryConfig
from chicory.diagnostics.activation_trace import ActivationTrace
from chicory.layer3.lateral_inhibition import WATERMARK_KEY
from chicory.layer3.synchronicity_engine import SynchronicityEngine
from tests.conftest import BagOfWordsModel


@pytest.fixture
def make_orchestrator(tmp_path):
    """The conftest `orchestrator` fixture, with config overrides."""
    from chicory.orchestrator.orchestrator import Orchestrator

    made = []

    def _make(**overrides):
        cfg = ChicoryConfig(
            db_path=tmp_path / f"chicory{len(made)}.db",
            anthropic_api_key="test-key",
            llm_model="test-model",
            embedding_model="mock",
            embedding_dimension=32,
            **overrides,
        )
        orch = Orchestrator(cfg)
        orch._embedding_engine._model = BagOfWordsModel(dimension=32)
        orch._sync_detection_background = lambda idle=False: None
        made.append(orch)
        return orch

    yield _make
    for orch in made:
        orch.close()


def _tag(db, name: str) -> int:
    row = db.query_one("SELECT id FROM tags WHERE name = ?", (name,))
    if row:
        return row["id"]
    return db.execute("INSERT INTO tags (name) VALUES (?)", (name,)).lastrowid


def _mem(db, mid: str, names: list[str]) -> None:
    db.execute(
        "INSERT INTO memories (id, content, source_model) VALUES (?, ?, 'test')",
        (mid, f"content of {mid}"),
    )
    for name in names:
        db.execute(
            "INSERT OR IGNORE INTO memory_tags (memory_id, tag_id) VALUES (?, ?)",
            (mid, _tag(db, name)),
        )


def _scenario_a(db) -> None:
    for i in range(60):
        _mem(db, f"py{i}", ["python", "asyncio", "bug", "month-2026-10", "p", "y"])
    for i in range(40):
        _mem(db, f"js{i}", ["javascript", "promise", "bug", "month-2026-10", "j", "a"])
    _mem(db, "rare-a", ["rare1", "rare2"])
    _mem(db, "rare-b", ["rare3"])
    db.connection.commit()


def _watermark(db):
    row = db.query_one(
        "SELECT value FROM librarian_state WHERE key = ?", (WATERMARK_KEY,),
    )
    return json.loads(row["value"]) if row else None


def _inhibition_rows(db) -> int:
    return db.query_one("SELECT COUNT(*) AS n FROM tag_inhibition")["n"]


@pytest.fixture
def count_computes(monkeypatch):
    """Count calls of compute_tag_inhibition made through the orchestrator."""
    calls = []
    real = orchestrator_module.compute_tag_inhibition

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(orchestrator_module, "compute_tag_inhibition", counting)
    return calls


# ── Retired API ─────────────────────────────────────────────────────────


def test_retired_api_is_gone():
    import chicory.models.lattice as lattice

    assert not hasattr(SynchronicityEngine, "compute_all_parallelness")
    assert not hasattr(SynchronicityEngine, "get_inhibition_score")
    assert not hasattr(lattice, "TagResonanceEntry")


# ── Synchronous recompute ───────────────────────────────────────────────


def test_run_sync_reports_inhibition(orchestrator):
    orch = orchestrator
    db = orch._db
    _scenario_a(db)

    stats = orch.run_sync()

    assert stats["inhibition_pairs"] == 4
    assert stats["inhibition_written"] == 4
    assert stats["inhibition_deleted"] == 0
    assert "inhibition_seconds" in stats
    assert not [k for k in stats if k.startswith("parallelness")]

    py, js = _tag(db, "python"), _tag(db, "javascript")
    index = orch._tensor_cache.inhibitor_index()
    assert js in index.get(py, {})
    assert index[py][js] == pytest.approx(0.8337, abs=1e-4)
    assert _inhibition_rows(db) == 4
    wm = _watermark(db)
    assert wm is not None and wm["pairs"] == 4


# ── Store path ──────────────────────────────────────────────────────────


def test_store_invalidates_cache_after_glyph_placement(make_orchestrator):
    # With the centroid branch off, glyph placement is the only thing on
    # the store path that writes tensor rows — and nothing else would
    # drop the cache after it.
    orch = make_orchestrator(centroid_inhibition_enabled=False)
    orch._tensor_cache.rows_for_a([1])
    assert orch._tensor_cache.is_loaded

    result = orch.handle_store_memory(
        "a note about zymurgy", tags=["zymurgyfermentation"],
    )

    assert result["status"] == "stored"
    placed = orch._db.query_one(
        "SELECT 1 FROM glyph_positions gp JOIN tags t ON t.id = gp.tag_id "
        "WHERE t.name = 'zymurgyfermentation'",
    )
    assert placed is not None
    assert orch._tensor_cache.is_loaded is False


def test_store_never_rewrites_tensor_parallelness(orchestrator):
    orch = orchestrator
    orch.handle_store_memory("first note on kettles", tags=["kettle", "teapot"])
    db = orch._db
    a, b = sorted((_tag(db, "kettle"), _tag(db, "teapot")))
    # Both ends are on the glyph lattice, which is what the retired
    # store-time pass rewrote parallelness from.
    assert db.query_one(
        "SELECT COUNT(*) AS n FROM glyph_positions WHERE tag_id IN (?, ?)", (a, b),
    )["n"] == 2
    db.execute(
        "INSERT INTO tag_relational_tensor (tag_a_id, tag_b_id, parallelness) "
        "VALUES (?, ?, 0.123) "
        "ON CONFLICT(tag_a_id, tag_b_id) DO UPDATE SET parallelness = 0.123",
        (a, b),
    )
    db.connection.commit()

    orch.handle_store_memory(
        "second note on saucers", tags=["saucer", "cupboard"],
    )

    row = db.query_one(
        "SELECT parallelness FROM tag_relational_tensor "
        "WHERE tag_a_id = ? AND tag_b_id = ?", (a, b),
    )
    assert row["parallelness"] == 0.123


# ── Background recompute ────────────────────────────────────────────────


def test_background_recompute_is_growth_gated(orchestrator, count_computes):
    orch = orchestrator
    db = orch._db
    _scenario_a(db)

    def run():
        with orch._foreground_db_lock:
            orch._maybe_recompute_inhibition()

    # No watermark yet: the first pass counts.
    orch._last_inhibition_check = None
    run()
    assert len(count_computes) == 1
    assert _inhibition_rows(db) == 4
    assert _watermark(db) is not None

    # Within the minimum interval: nothing.
    run()
    assert len(count_computes) == 1

    # Interval elapsed, but memory_tags has not grown.
    orch._last_inhibition_check = None
    run()
    assert len(count_computes) == 1

    # memory_tags must grow by 2% of the watermark's max rowid: one row
    # short of that is not enough, the last row is.
    base = _watermark(db)["max_rowid"]
    grow = max(1, -(-base * 2 // 100))
    for i in range(grow - 1):
        _mem(db, f"grow{i}", [f"growth-{i}"])
    db.connection.commit()
    orch._last_inhibition_check = None
    run()
    assert len(count_computes) == 1

    _mem(db, "grow-last", ["growth-last"])
    db.connection.commit()
    orch._last_inhibition_check = None
    run()
    assert len(count_computes) == 2
    assert _watermark(db)["max_rowid"] == base + grow


def test_background_backlog_resumes(orchestrator, monkeypatch):
    orch = orchestrator
    db = orch._db
    _scenario_a(db)
    monkeypatch.setattr(lateral_inhibition, "WRITE_CHUNK", 1)
    before = orch._tensor_cache.inhibitor_index()

    # A foreground call is waiting: one chunk is written, then it yields.
    orch._foreground_waiting.set()
    orch._last_inhibition_check = None
    with orch._foreground_db_lock:
        orch._maybe_recompute_inhibition()

    backlog = orch._inhibition_backlog
    assert backlog is not None and not backlog.done
    assert backlog.written == 1
    assert _inhibition_rows(db) == 1
    assert orch._tensor_cache.inhibitor_index() is before
    assert _watermark(db) is None

    # Foreground gone: the next pass finishes the same plan.
    orch._foreground_waiting.clear()
    orch._last_inhibition_check = None
    with orch._foreground_db_lock:
        orch._maybe_recompute_inhibition()

    assert orch._inhibition_backlog is None
    assert backlog.done and backlog.written == 4
    assert _inhibition_rows(db) == 4
    published = orch._tensor_cache.inhibitor_index()
    assert published is backlog.index
    assert _tag(db, "javascript") in published[_tag(db, "python")]
    assert _watermark(db) is not None


# ── Boot seeding ────────────────────────────────────────────────────────


def test_glyph_xref_seeded_once(orchestrator, monkeypatch):
    from chicory.layer1.glyph_bridge import GlyphBridge

    orch = orchestrator
    seeded = []
    orch._glyph_encoder = object.__new__(GlyphBridge)
    monkeypatch.setattr(
        orch._sync_engine, "seed_from_cross_reference",
        lambda: seeded.append("xref") or 3,
    )
    monkeypatch.setattr(
        orch._sync_engine, "seed_semiotic_from_pairs",
        lambda: seeded.append("semiotic") or 0,
    )

    orch._maybe_seed_glyph_cross_reference()
    orch._maybe_seed_glyph_cross_reference()

    assert seeded == ["xref", "semiotic"]
    marker = orch._db.query_one(
        "SELECT value FROM librarian_state WHERE key = 'glyph_xref_seeded'",
    )
    assert marker is not None and marker["value"]


def test_glyph_xref_not_marked_when_nothing_seeded(orchestrator, monkeypatch):
    from chicory.layer1.glyph_bridge import GlyphBridge

    orch = orchestrator
    orch._glyph_encoder = object.__new__(GlyphBridge)
    monkeypatch.setattr(orch._sync_engine, "seed_from_cross_reference", lambda: 0)
    monkeypatch.setattr(orch._sync_engine, "seed_semiotic_from_pairs", lambda: 0)

    orch._maybe_seed_glyph_cross_reference()

    assert orch._db.query_one(
        "SELECT 1 FROM librarian_state WHERE key = 'glyph_xref_seeded'",
    ) is None


def test_seed_from_cross_reference_writes_no_inhibition(orchestrator, monkeypatch):
    from chicory.layer1.glyph_bridge import GlyphBridge

    orch = orchestrator
    db = orch._db
    a, b = sorted((_tag(db, "upward"), _tag(db, "downward")))
    db.connection.commit()

    bridge = object.__new__(GlyphBridge)
    bridge.get_cross_reference_data = lambda: {"oppositions": {"A": {"B"}}}
    engine = orch._sync_engine
    monkeypatch.setattr(engine, "_glyph_encoder", bridge)
    monkeypatch.setattr(
        engine, "_build_glyph_to_tag_ids", lambda _bridge: {"A": [a], "B": [b]},
    )
    monkeypatch.setattr(engine, "_load_resonant_tag_set", lambda: {(a, b)})

    assert engine.seed_from_cross_reference() == 1

    row = db.query_one(
        "SELECT semantic_strength, inhibition_strength FROM tag_relational_tensor "
        "WHERE tag_a_id = ? AND tag_b_id = ?", (a, b),
    )
    assert row["semantic_strength"] == 0.6
    assert row["inhibition_strength"] == 0.0


# ── Ingest ──────────────────────────────────────────────────────────────


def test_ingest_finalize_does_no_inhibition_and_refreshes_tag_space(
    orchestrator, count_computes,
):
    orch = orchestrator
    _scenario_a(orch._db)
    stored = orch.handle_store_memory(
        "a deferred note on ingest", tags=["ingested", "deferred"],
        defer_side_effects=True,
    )
    orch._tag_space._get_lexical_index()
    assert orch._tag_space._lexical_index is not None

    orch._finalize_ingested_memories([stored["memory_id"]])

    assert count_computes == []
    assert _inhibition_rows(orch._db) == 0
    assert orch._tag_space._lexical_index is None


# ── Activation trace ────────────────────────────────────────────────────


def test_activation_trace_reports_inhibition_only_when_present():
    quiet = ActivationTrace(query="q", method="hybrid", tags_requested=[])
    assert "tag_inhibition" not in quiet.to_dict()

    trace = ActivationTrace(
        query="q", method="hybrid", tags_requested=[],
        tag_inhibition={"promise": 0.41666666, "javascript": 0.83373737},
    )
    assert list(trace.to_dict()["tag_inhibition"].items()) == [
        ("javascript", 0.8337), ("promise", 0.4167),
    ]


def test_retrieval_trace_names_inhibited_partners(orchestrator):
    # bug excites javascript (glyph 14.0) and asyncio (11.0); python, on
    # every top-5 result, inhibits javascript without any edge to it.
    orch = orchestrator
    db = orch._db
    _scenario_a(db)
    bug, js, asyncio = (_tag(db, n) for n in ("bug", "javascript", "asyncio"))
    for partner, glyph in ((js, 14.0), (asyncio, 11.0)):
        db.execute(
            "INSERT INTO tag_relational_tensor (tag_a_id, tag_b_id, glyph_strength) "
            "VALUES (?, ?, ?)", (min(bug, partner), max(bug, partner), glyph),
        )
    db.connection.commit()
    orch._tensor_cache.invalidate()
    orch.recompute_inhibition()
    for i in range(3):
        orch.handle_store_memory(
            f"python bug number {i} in the event loop", tags=["python", "bug"],
        )

    orch.handle_retrieve_memories("python bug event loop", method="hybrid")

    trace = orch._last_trace
    assert set(trace.tag_inhibition) == {"javascript"}
    assert trace.tag_inhibition["javascript"] == pytest.approx(0.8337, abs=1e-4)
    assert trace.to_dict()["tag_inhibition"] == {"javascript": 0.8337}
