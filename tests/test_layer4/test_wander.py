"""Tests for the wandering librarian: distributed cursor, errands, heartbeat.

The properties worth pinning here are the ones that make an unattended
patrol safe to leave running:

- it always advances (nothing it examined can return to the head of the
  queue, including things it failed on),
- it covers everything before it repeats,
- it yields to the foreground, and
- it stops when told to.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer4.wander import (
    ERRANDS,
    Errand,
    Heartbeat,
    Patrol,
    WanderState,
)
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
        wander_batch_size=3,
        wander_tick_seconds=0.01,
        wander_idle_ticks=2,
        wander_interval_minutes=0.0,  # no rate limit in tests
        wander_errands=["missing_embedding"],
    )
    db = DatabaseEngine(config)
    db.connect()
    apply_schema(db)

    emb = MockEmbeddingEngine(dimension=DIM)
    emb.set_db(db)

    yield {
        "config": config,
        "db": db,
        "emb": emb,
        "patrol": Patrol(config, db, emb),
    }
    db.close()


def make_memory(db, mid: str, content: str = "some content", archived: int = 0):
    db.execute(
        "INSERT INTO memories (id, content, source_model, is_archived) "
        "VALUES (?, ?, 'test', ?)",
        (mid, content, archived),
    )
    return mid


# ── Schema ──────────────────────────────────────────────────────────────


def test_tables_exist(stack):
    names = {
        r["name"]
        for r in stack["db"].execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "patrol_visits" in names
    assert "wander_state" in names


# ── State ───────────────────────────────────────────────────────────────


def test_state_created_on_first_read(stack):
    state = WanderState(stack["db"])
    rec = state.get("missing_embedding")
    assert rec.errand == "missing_embedding"
    assert rec.lap == 0
    assert rec.enabled is True


def test_seed_is_stable_across_reads(stack):
    state = WanderState(stack["db"])
    assert state.get("errand-a").seed == state.get("errand-a").seed


def test_disabled_errand_does_not_run(stack):
    db, patrol = stack["db"], stack["patrol"]
    make_memory(db, "m1")
    patrol.state.set_enabled("missing_embedding", False)
    result = patrol.run_errand("missing_embedding")
    assert result.get("skipped") == "disabled"
    assert db.execute("SELECT COUNT(*) AS n FROM patrol_visits").fetchone()["n"] == 0


def test_unknown_errand_is_reported_not_raised(stack):
    assert stack["patrol"].run_errand("no-such-errand")["error"] == "unknown"


# ── Cursor ──────────────────────────────────────────────────────────────


def _visit_errand(name="probe", population=None, when="always"):
    """An errand that examines without changing anything."""
    return Errand(
        name=name,
        description="test probe",
        run=lambda memory_ids, **_: {m: ("ok", None) for m in memory_ids},
        population=population,
        when=when,
    )


def test_unvisited_memories_lead(stack):
    db, patrol = stack["db"], stack["patrol"]
    errand = _visit_errand()
    ERRANDS[errand.name] = errand
    for i in range(5):
        make_memory(db, f"m{i}")

    first = patrol.select(errand, 3)
    assert len(first) == 3
    patrol.record_visits(errand.name, 0, {m: ("ok", None) for m in first})

    # The two never-visited memories must come before anything already
    # stamped — the NULLs-first property the whole design rests on.
    second = patrol.select(errand, 3)
    assert set(second[:2]) == {"m0", "m1", "m2", "m3", "m4"} - set(first)


def test_full_coverage_before_repeat(stack):
    db, patrol = stack["db"], stack["patrol"]
    errand = _visit_errand()
    ERRANDS[errand.name] = errand
    for i in range(7):
        make_memory(db, f"m{i}")

    seen = set()
    for _ in range(3):  # 3 batches of 3 covers 7
        batch = patrol.select(errand, 3)
        patrol.record_visits(errand.name, 0, {m: ("ok", None) for m in batch})
        seen.update(batch)
    assert seen == {f"m{i}" for i in range(7)}


def test_archived_memories_are_not_visited(stack):
    db, patrol = stack["db"], stack["patrol"]
    errand = _visit_errand()
    ERRANDS[errand.name] = errand
    make_memory(db, "live")
    make_memory(db, "gone", archived=1)
    assert patrol.select(errand, 10) == ["live"]


def test_population_narrows_the_walk(stack):
    db, patrol = stack["db"], stack["patrol"]
    errand = _visit_errand(name="narrow", population="m.content LIKE 'keep%'")
    ERRANDS[errand.name] = errand
    make_memory(db, "a", content="keep me")
    make_memory(db, "b", content="skip me")
    assert patrol.select(errand, 10) == ["a"]


def test_errands_have_independent_cursors(stack):
    db, patrol = stack["db"], stack["patrol"]
    one, two = _visit_errand("one"), _visit_errand("two")
    ERRANDS[one.name], ERRANDS[two.name] = one, two
    for i in range(3):
        make_memory(db, f"m{i}")

    batch = patrol.select(one, 3)
    patrol.record_visits(one.name, 0, {m: ("ok", None) for m in batch})

    # The second errand has walked nothing, so everything is still new to it.
    rows = db.execute(
        "SELECT COUNT(*) AS n FROM patrol_visits WHERE errand = 'two'"
    ).fetchone()
    assert rows["n"] == 0
    assert len(patrol.select(two, 3)) == 3


# ── Progress guarantee ──────────────────────────────────────────────────


def test_failures_still_advance_the_cursor(stack):
    """A memory an errand could not fix must not block the queue forever."""
    db, patrol = stack["db"], stack["patrol"]
    errand = Errand(
        name="always-fails",
        description="test",
        run=lambda memory_ids, **_: {m: ("failed", "nope") for m in memory_ids},
    )
    ERRANDS[errand.name] = errand
    for i in range(4):
        make_memory(db, f"m{i}")

    first = patrol.run_errand("always-fails", batch_size=2)
    assert first["examined"] == 2
    assert first["failed"] == 2

    # Next batch must be the OTHER two, not a retry of the same two.
    stamped = {
        r["memory_id"]
        for r in db.execute(
            "SELECT memory_id FROM patrol_visits WHERE errand = 'always-fails'"
        ).fetchall()
    }
    assert len(stamped) == 2
    assert set(patrol.select(errand, 2)) & stamped == set()


def test_empty_content_is_recorded_not_retried(stack):
    db, patrol = stack["db"], stack["patrol"]
    make_memory(db, "blank", content="   ")
    result = patrol.run_errand("missing_embedding")
    assert result["examined"] == 1
    row = db.execute(
        "SELECT verdict FROM patrol_visits WHERE memory_id = 'blank'"
    ).fetchone()
    assert row["verdict"] == "skipped"


# ── The missing_embedding errand ────────────────────────────────────────


def test_missing_embedding_repairs_and_leaves_population(stack):
    db, patrol = stack["db"], stack["patrol"]
    for i in range(3):
        make_memory(db, f"m{i}", content=f"content number {i}")

    errand = ERRANDS["missing_embedding"]
    assert patrol.population_size(errand) == 3

    result = patrol.run_errand("missing_embedding")
    assert result["acted"] == 3
    # Repaired memories drop out of the errand's own population.
    assert patrol.population_size(errand) == 0
    n = db.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()["n"]
    assert n >= 3


def test_missing_embedding_resolves_sweep_findings(stack):
    db, patrol = stack["db"], stack["patrol"]
    make_memory(db, "m1", content="real content")
    db.execute(
        "INSERT INTO sweep_findings (memory_id, finding_type, detail) "
        "VALUES ('m1', 'missing_embedding', '')"
    )
    patrol.run_errand("missing_embedding")
    row = db.execute(
        "SELECT resolved FROM sweep_findings WHERE memory_id = 'm1'"
    ).fetchone()
    assert row["resolved"] == 1


def test_unrelated_findings_are_left_alone(stack):
    db, patrol = stack["db"], stack["patrol"]
    make_memory(db, "m1", content="real content")
    db.execute(
        "INSERT INTO sweep_findings (memory_id, finding_type, detail) "
        "VALUES ('m1', 'tag_drift', 'below_threshold')"
    )
    patrol.run_errand("missing_embedding")
    row = db.execute(
        "SELECT resolved FROM sweep_findings WHERE finding_type = 'tag_drift'"
    ).fetchone()
    assert row["resolved"] == 0


def test_run_records_counters(stack):
    db, patrol = stack["db"], stack["patrol"]
    for i in range(2):
        make_memory(db, f"m{i}", content=f"body {i}")
    patrol.run_errand("missing_embedding")
    rec = patrol.state.get("missing_embedding")
    assert rec.runs == 1
    assert rec.visited_count == 2
    assert rec.acted_count == 2
    assert "acted=2" in (rec.last_result or "")


def test_lap_advances_when_population_empties(stack):
    """Completion is discovered on the next run, not predicted on this one.

    Confirming it in the same run would mean re-counting the population
    right after mutating it — the 3.4s scan the cursor exists to avoid.
    """
    db, patrol = stack["db"], stack["patrol"]
    make_memory(db, "m1", content="body")
    assert patrol.state.get("missing_embedding").lap == 0

    first = patrol.run_errand("missing_embedding")
    assert first["acted"] == 1
    assert first["lap"] == 0

    second = patrol.run_errand("missing_embedding")
    assert second["examined"] == 0
    assert second["lap"] == 1

    # And it must not keep ticking on every empty run afterwards.
    assert patrol.run_errand("missing_embedding")["lap"] == 1


def test_lap_advances_when_a_survey_wraps(stack):
    """A whole-library errand advances once every memory carries the lap."""
    db, patrol = stack["db"], stack["patrol"]
    errand = _visit_errand("survey")
    ERRANDS[errand.name] = errand
    patrol._config.wander_errands = ["survey"]
    for i in range(4):
        make_memory(db, f"m{i}")

    # Two batches of 2 cover all four; neither completes the lap.
    assert patrol.run_errand("survey", batch_size=2)["lap"] == 0
    assert patrol.run_errand("survey", batch_size=2)["lap"] == 0
    # The third batch finds the head already stamped at lap 0 — wrapped.
    assert patrol.run_errand("survey", batch_size=2)["lap"] == 1


# ── Yielding ────────────────────────────────────────────────────────────


def test_should_stop_halts_mid_batch(stack):
    db, patrol = stack["db"], stack["patrol"]
    for i in range(5):
        make_memory(db, f"m{i}", content=f"body {i}")

    calls = {"n": 0}

    def should_stop():
        calls["n"] += 1
        return calls["n"] > 2  # let two memories through, then preempt

    result = patrol.run_errand(
        "missing_embedding", batch_size=5, should_stop=should_stop,
    )
    assert result["examined"] < 5
    # What it did examine is durably stamped; the rest stay at the head.
    stamped = db.execute(
        "SELECT COUNT(*) AS n FROM patrol_visits"
    ).fetchone()["n"]
    assert stamped == result["examined"]


def test_idle_only_errands_wait_for_quiet(stack):
    db, patrol = stack["db"], stack["patrol"]
    errand = _visit_errand("idle-only", when="idle")
    ERRANDS[errand.name] = errand
    make_memory(db, "m1")
    patrol._config.wander_errands = ["idle-only"]

    busy = patrol.run_due(idle=False)
    assert busy["errands"] == []

    quiet = patrol.run_due(idle=True)
    assert quiet["errands"][0]["examined"] == 1


# ── Heartbeat ───────────────────────────────────────────────────────────


def test_heartbeat_tick_passes_idle_state(stack):
    seen = []
    hb = Heartbeat(stack["config"], tick=seen.append)
    for _ in range(4):
        hb.run_once()
    # wander_idle_ticks=2: the first two beats are still warming up.
    assert seen == [False, True, True, True]


def test_heartbeat_skips_and_resets_when_busy(stack):
    seen = []
    busy = {"v": False}
    hb = Heartbeat(stack["config"], tick=seen.append, busy=lambda: busy["v"])

    hb.run_once()
    hb.run_once()
    assert hb.idle is True

    busy["v"] = True
    assert hb.run_once() is False       # tick skipped entirely
    assert hb.idle is False             # and the idle run reset

    busy["v"] = False
    hb.run_once()
    assert hb.idle is False             # one quiet tick isn't idle again yet


def test_heartbeat_survives_a_failing_tick(stack):
    def boom(_idle):
        raise RuntimeError("tick exploded")

    hb = Heartbeat(stack["config"], tick=boom)
    assert hb.run_once() is True        # the loop must not die
    assert hb.ticks == 1


def test_heartbeat_disabled_by_config(stack):
    stack["config"].wander_enabled = False
    hb = Heartbeat(stack["config"], tick=lambda _idle: None)
    assert hb.start() is False


def test_heartbeat_start_is_idempotent_and_stops(stack):
    beats = []
    hb = Heartbeat(stack["config"], tick=lambda idle: beats.append(idle))
    assert hb.start() is True
    assert hb.start() is True           # second start must not spawn a rival
    hb.stop()
    stopped_at = hb.ticks
    hb.stop()                           # stopping twice is safe
    assert hb.ticks == stopped_at
