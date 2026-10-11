"""Orchestrator wiring for data-derived lateral inhibition.

What triggers a recompute (run_sync, the background growth gate), what
must not (store, ingest finalize), how a pre-empted write resumes, when a
plan has gone stale and is dropped, how a process picks up another
process's recompute, and the cache invalidations and boot marker that
went in alongside.  The evidence is scenario A, inserted with direct SQL:
60 units {python, asyncio, bug} and 40 units {javascript, promise, bug},
each also carrying a month tag and two letters, plus rare units — four
cross-language pairs come out inhibitory.
"""

from __future__ import annotations

import json

import pytest

import chicory.layer3.lateral_inhibition as lateral_inhibition
import chicory.orchestrator.orchestrator as orchestrator_module
from chicory.config import ChicoryConfig
from chicory.diagnostics.activation_trace import ActivationTrace
from chicory.layer3.lateral_inhibition import WATERMARK_KEY, load_inhibitor_index
from chicory.layer3.synchronicity_engine import SynchronicityEngine
from tests.conftest import BagOfWordsModel


@pytest.fixture
def make_orchestrator(tmp_path):
    """The conftest `orchestrator` fixture, with config overrides."""
    from chicory.orchestrator.orchestrator import Orchestrator

    made = []

    def _make(**overrides):
        settings = {
            "db_path": tmp_path / f"chicory{len(made)}.db",
            "anthropic_api_key": "test-key",
            "llm_model": "test-model",
            "embedding_model": "mock",
            "embedding_dimension": 32,
        }
        settings.update(overrides)
        orch = Orchestrator(ChicoryConfig(**settings))
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


def _table(db) -> dict[tuple[int, int], float]:
    """tag_inhibition as {(a, b): I}."""
    return {
        (r["tag_a_id"], r["tag_b_id"]): r["inhibition_strength"]
        for r in db.query(
            "SELECT tag_a_id, tag_b_id, inhibition_strength FROM tag_inhibition",
        )
    }


def _flatten(index) -> dict[tuple[int, int], float]:
    """An inhibitor index in the table's shape, to compare the two."""
    return {
        (a, b): w for a, row in index.items() for b, w in row.items() if a < b
    }


def _growth_needed(db) -> int:
    """memory_tags rows that make the background pass due again."""
    base = _watermark(db)["max_rowid"]
    return max(1, -(-base * 2 // 100))


def _grow(db, prefix: str, n: int) -> None:
    """Add n memory_tags rows of fresh, rare tags: growth, not evidence."""
    for i in range(n):
        _mem(db, f"{prefix}{i}", [f"{prefix}-tag-{i}"])
    db.connection.commit()


def _background_pass(orch) -> None:
    """One background inhibition phase, as _sync_detection_background runs it."""
    orch._last_inhibition_check = None
    with orch._foreground_db_lock:
        orch._maybe_recompute_inhibition()


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

    # Within the minimum interval: nothing, although memory_tags has grown
    # enough — so it is the interval that holds the pass back.
    _grow(db, "early", _growth_needed(db))
    run()
    assert len(count_computes) == 1

    # The same growth once the interval has elapsed counts.
    orch._last_inhibition_check = None
    run()
    assert len(count_computes) == 2

    # Interval elapsed, but memory_tags has not grown.
    orch._last_inhibition_check = None
    run()
    assert len(count_computes) == 2

    # memory_tags must grow by 2% of the watermark's max rowid: one row
    # short of that is not enough, the last row is.
    base = _watermark(db)["max_rowid"]
    grow = _growth_needed(db)
    _grow(db, "grow", grow - 1)
    orch._last_inhibition_check = None
    run()
    assert len(count_computes) == 2

    _mem(db, "grow-last", ["growth-last"])
    db.connection.commit()
    orch._last_inhibition_check = None
    run()
    assert len(count_computes) == 3
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


def _queue_backlog(orch, monkeypatch) -> None:
    """A background pass pre-empted after its first write chunk."""
    monkeypatch.setattr(lateral_inhibition, "WRITE_CHUNK", 1)
    orch._foreground_waiting.set()
    try:
        _background_pass(orch)
    finally:
        orch._foreground_waiting.clear()
    backlog = orch._inhibition_backlog
    assert backlog is not None and backlog.written == 1 and not backlog.done


def _co_occur(db, n: int) -> None:
    """Units carrying both languages: new evidence that moves I(python, javascript)."""
    for i in range(n):
        _mem(db, f"both{i}", ["python", "javascript", "bug"])
    db.connection.commit()


def test_counting_runs_without_the_foreground_lock(orchestrator, monkeypatch):
    orch = orchestrator
    _scenario_a(orch._db)
    lock = orch._foreground_db_lock
    seen: list[tuple[str, bool]] = []

    def watch(name):
        real = getattr(orchestrator_module, name)

        def watched(*args, **kwargs):
            seen.append((name, lock._is_owned()))
            return real(*args, **kwargs)

        monkeypatch.setattr(orchestrator_module, name, watched)

    for name in ("compute_tag_inhibition", "plan_inhibition", "persist_inhibition"):
        watch(name)

    # Background: the count and its diff run with the lock released, and
    # the write under it again.
    _background_pass(orch)
    assert seen == [
        ("compute_tag_inhibition", False),
        ("plan_inhibition", False),
        ("persist_inhibition", True),
    ]

    # Synchronous: the count runs without the lock too, but the diff is
    # taken under it, so no background write can land between the diff
    # and the write that relies on it.
    seen.clear()
    orch.recompute_inhibition()
    assert seen == [
        ("compute_tag_inhibition", False),
        ("plan_inhibition", True),
        ("persist_inhibition", True),
    ]
    assert not orch._foreground_waiting.is_set()


def test_inhibition_disabled_computes_nothing(make_orchestrator, count_computes):
    orch = make_orchestrator(inhibition_enabled=False)
    db = orch._db
    _scenario_a(db)

    assert orch.recompute_inhibition() == {"pairs": 0, "written": 0, "deleted": 0}
    _background_pass(orch)

    assert count_computes == []
    assert _inhibition_rows(db) == 0
    assert _watermark(db) is None


def test_recompute_clears_a_stale_backlog(orchestrator, monkeypatch):
    orch = orchestrator
    db = orch._db
    _scenario_a(db)
    _queue_backlog(orch, monkeypatch)
    _co_occur(db, 10)

    orch.recompute_inhibition()

    assert orch._inhibition_backlog is None
    fresh = orch._tensor_cache.inhibitor_index()
    assert _table(db) == _flatten(fresh)

    # Nothing is left for the next pass to write over the fresh count.
    _background_pass(orch)
    assert orch._tensor_cache.inhibitor_index() is fresh
    assert _table(db) == _flatten(fresh)


def test_recompute_while_counting_drops_the_background_plan(
    orchestrator, monkeypatch,
):
    # run_sync on the MCP thread while the heartbeat's pass counts with the
    # lock released: the background plan is older than what run_sync wrote.
    orch = orchestrator
    db = orch._db
    _scenario_a(db)
    orch.recompute_inhibition()
    _grow(db, "grow", _growth_needed(db))
    real = orchestrator_module.compute_tag_inhibition
    calls = []
    fresh = {}

    def count_then_sync(*args, **kwargs):
        calls.append(1)
        comp = real(*args, **kwargs)
        if len(calls) == 1:
            _co_occur(db, 10)
            orch.recompute_inhibition()
            fresh["index"] = orch._tensor_cache.inhibitor_index()
            fresh["watermark"] = _watermark(db)
        return comp

    monkeypatch.setattr(orchestrator_module, "compute_tag_inhibition", count_then_sync)

    _background_pass(orch)

    assert len(calls) == 2
    assert orch._inhibition_backlog is None
    assert orch._tensor_cache.inhibitor_index() is fresh["index"]
    assert _table(db) == _flatten(fresh["index"])
    assert _watermark(db) == fresh["watermark"]


def _merge(orch, source: str, target: str) -> None:
    db = orch._db
    orch._tag_manager.merge_tags(_tag(db, source), _tag(db, target))


def _touching(db, *names: str) -> list[tuple[int, int]]:
    ids = {_tag(db, n) for n in names}
    return [pair for pair in _table(db) if ids & set(pair)]


def test_merge_while_counting_drops_the_plan_and_recounts(
    orchestrator, monkeypatch,
):
    orch = orchestrator
    db = orch._db
    _scenario_a(db)
    orch.recompute_inhibition()
    before = orch._tensor_cache.inhibitor_index()
    _grow(db, "grow", _growth_needed(db))
    real = orchestrator_module.compute_tag_inhibition
    calls = []

    def count_then_merge(*args, **kwargs):
        comp = real(*args, **kwargs)
        if not calls:
            _merge(orch, "promise", "javascript")
        calls.append(1)
        return comp

    monkeypatch.setattr(orchestrator_module, "compute_tag_inhibition", count_then_merge)

    _background_pass(orch)

    # The merge's deletes stand, and so does the recount it asked for.
    assert len(calls) == 1
    assert _touching(db, "promise", "javascript") == []
    assert _watermark(db)["max_rowid"] == 0
    assert orch._inhibition_backlog is None
    # The dropped plan was never published; the reset watermark is, so the
    # index now matches the table without the merged tag's rows.
    assert orch._tensor_cache.inhibitor_index() is not before
    assert _flatten(orch._tensor_cache.inhibitor_index()) == _table(db)

    _background_pass(orch)

    assert len(calls) == 2
    assert _touching(db, "promise") == []
    py, js = _tag(db, "python"), _tag(db, "javascript")
    assert (min(py, js), max(py, js)) in _table(db)
    assert _flatten(orch._tensor_cache.inhibitor_index()) == _table(db)
    assert _watermark(db)["max_rowid"] > 0


def test_merge_during_the_first_ever_count_drops_the_plan(
    orchestrator, monkeypatch,
):
    # No watermark yet, as on every store right after the upgrade.  A merge
    # that only deleted the watermark would leave None == None and the
    # stale plan would be written.
    orch = orchestrator
    db = orch._db
    _scenario_a(db)
    assert _watermark(db) is None
    real = orchestrator_module.compute_tag_inhibition
    calls = []

    def count_then_merge(*args, **kwargs):
        comp = real(*args, **kwargs)
        if not calls:
            _merge(orch, "promise", "javascript")
        calls.append(1)
        return comp

    monkeypatch.setattr(orchestrator_module, "compute_tag_inhibition", count_then_merge)

    _background_pass(orch)

    assert len(calls) == 1
    assert _table(db) == {}
    assert _watermark(db)["max_rowid"] == 0

    _background_pass(orch)

    assert len(calls) == 2
    assert _touching(db, "promise") == []
    assert _table(db)
    assert _watermark(db)["max_rowid"] > 0


def test_merge_with_a_backlog_queued_drops_it(orchestrator, monkeypatch):
    orch = orchestrator
    db = orch._db
    _scenario_a(db)
    orch.recompute_inhibition()
    # Enough new evidence to be due, and to move every stored strength.
    _co_occur(db, 10)
    _grow(db, "grow", _growth_needed(db))
    _queue_backlog(orch, monkeypatch)

    _merge(orch, "promise", "javascript")
    after_merge = _table(db)
    _background_pass(orch)

    assert orch._inhibition_backlog is None
    assert _table(db) == after_merge
    assert _touching(db, "promise", "javascript") == []
    assert _watermark(db)["max_rowid"] == 0


# ── Several processes on one database ───────────────────────────────────


def test_peer_recompute_is_picked_up(make_orchestrator, tmp_path):
    # One MCP server per client, plus `chicory sync`, share one file; each
    # keeps its own inhibitor index.
    shared = tmp_path / "shared.db"
    server = make_orchestrator(db_path=shared)
    db = server._db
    _scenario_a(db)
    assert server._tensor_cache.inhibitor_index() == {}

    cli = make_orchestrator(db_path=shared)
    assert cli.recompute_inhibition()["pairs"] == 4

    # Not due here — the peer's watermark is fresh — but its rows are new
    # to this process.
    _background_pass(server)
    published = server._tensor_cache.inhibitor_index()
    assert published == load_inhibitor_index(db)
    assert _flatten(published) == _table(db) and len(_table(db)) == 4

    # Nothing changed since: no reload.
    _background_pass(server)
    assert server._tensor_cache.inhibitor_index() is published


def test_peer_recompute_between_backlog_passes_drops_the_backlog(
    make_orchestrator, tmp_path, monkeypatch,
):
    shared = tmp_path / "shared.db"
    server = make_orchestrator(db_path=shared)
    db = server._db
    _scenario_a(db)
    _queue_backlog(server, monkeypatch)

    _co_occur(db, 10)
    cli = make_orchestrator(db_path=shared)
    cli.recompute_inhibition()
    fresh = _table(db)
    assert fresh == _flatten(cli._tensor_cache.inhibitor_index())

    # The backlog was diffed against a table the peer has since rewritten:
    # dropped, and the peer's count taken instead.
    _background_pass(server)
    assert server._inhibition_backlog is None
    assert _table(db) == fresh
    published = server._tensor_cache.inhibitor_index()
    assert _flatten(published) == fresh

    _background_pass(server)
    assert server._tensor_cache.inhibitor_index() is published
    assert _table(db) == fresh


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


def test_sync_and_codebase_ingest_drop_the_lexical_index(orchestrator, tmp_path):
    # Both add tags and lattice positions the lexical index cannot see
    # until it is rebuilt.
    orch = orchestrator
    orch.handle_store_memory("a note on kettles", tags=["kettle", "teapot"])
    orch._tag_space._get_lexical_index()
    assert orch._tag_space._lexical_index is not None

    orch.run_sync()
    assert orch._tag_space._lexical_index is None

    project = tmp_path / "project"
    project.mkdir()
    (project / "kettle.py").write_text(
        '"""Kettle helpers."""\n\n\n'
        "def boil(water: int) -> int:\n"
        '    """Boil the water."""\n'
        "    return water + 1\n",
    )
    orch._tag_space._get_lexical_index()
    assert orch._tag_space._lexical_index is not None

    stats = orch.handle_ingest_codebase(str(project))
    assert stats["memories_created"] >= 1
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


def _inhibiting_store(orch) -> None:
    """Scenario A and three notes, set up for "python bug event loop".

    bug excites javascript (glyph 14.0) and asyncio (11.0); python, on
    every top-5 result, inhibits javascript without any edge to it.
    """
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


def test_retrieval_trace_names_inhibited_partners(orchestrator):
    orch = orchestrator
    _inhibiting_store(orch)

    orch.handle_retrieve_memories("python bug event loop", method="hybrid")

    trace = orch._last_trace
    assert set(trace.tag_inhibition) == {"javascript"}
    assert trace.tag_inhibition["javascript"] == pytest.approx(0.8337, abs=1e-4)
    assert trace.to_dict()["tag_inhibition"] == {"javascript": 0.8337}


def test_trace_never_carries_the_previous_querys_inhibition(orchestrator):
    # A semantic query never reaches the centroid path, which is the only
    # thing that rewrites the partner activations and inhibition.
    orch = orchestrator
    _inhibiting_store(orch)
    orch.handle_retrieve_memories("python bug event loop", method="hybrid")
    assert orch._last_trace.tag_inhibition

    orch.handle_retrieve_memories(
        "completely unrelated gardening tomatoes", method="semantic",
    )

    trace = orch._last_trace
    assert trace.method == "semantic"
    assert trace.tag_inhibition == {}
    assert trace.tag_activations == {}
    assert "tag_inhibition" not in trace.to_dict()


# ── Kill switch ─────────────────────────────────────────────────────────


def test_env_switches_inhibition_off_for_the_cli(tmp_path, monkeypatch):
    # An MCP server or the CLI builds its config with load_config(), so
    # the environment is the only place an operator can turn it off; the
    # CLI's square scoring must then apply no gain either.
    import typer.testing

    import chicory.layer3.square_finder as square_finder
    from chicory.cli.app import app
    from chicory.config import load_config

    monkeypatch.setenv("CHICORY_TENSOR_INHIBITION_WEIGHT", "3.5")
    monkeypatch.setenv("CHICORY_INHIBITION_ENABLED", "true")
    on = load_config(db_path=tmp_path / "on.db")
    assert on.inhibition_enabled is True
    assert on.tensor_inhibition_weight == 3.5

    monkeypatch.setenv("CHICORY_INHIBITION_ENABLED", "false")
    assert load_config(db_path=tmp_path / "off.db").inhibition_enabled is False

    gains = []

    def fake_find_square_motifs(_db, **kwargs):
        gains.append(kwargs["layer_weights"]["inhibition"])
        return []

    monkeypatch.setattr(square_finder, "find_square_motifs", fake_find_square_motifs)
    runner = typer.testing.CliRunner()
    for enabled in ("true", "false"):
        monkeypatch.setenv("CHICORY_INHIBITION_ENABLED", enabled)
        result = runner.invoke(
            app, ["squares", "--db", str(tmp_path / "cli.db"), "--json"],
        )
        assert result.exit_code == 0, result.output

    assert gains == [3.5, 0.0]


def test_every_cli_weight_site_goes_through_effective_gain():
    # The kill switch and the gain clamp live in effective_gain; a site that
    # reads the raw weight would bypass both.
    from pathlib import Path

    source = (Path(orchestrator_module.__file__).parents[1] / "cli" / "app.py").read_text()
    assert "config.tensor_inhibition_weight" not in source
    assert source.count('"inhibition": effective_gain(config)') == 9
