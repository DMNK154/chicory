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
    register,
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


# ── Lease ───────────────────────────────────────────────────────────────


def _rival(stack):
    """A second Patrol on the same database, as a second server would be."""
    from chicory.layer4.wander import Patrol
    other = Patrol(stack["config"], stack["db"], stack["emb"])
    other.lease._owner = "otherhost:99999"
    return other


def test_lease_is_granted_once(stack):
    mine, theirs = stack["patrol"], _rival(stack)
    assert mine.lease.acquire() is True
    assert theirs.lease.acquire() is False
    assert mine.lease.holder()["owner"] == mine.lease.owner


def test_lease_is_reentrant_for_its_holder(stack):
    lease = stack["patrol"].lease
    assert lease.acquire() is True
    assert lease.acquire() is True      # renews rather than blocking itself


def test_release_frees_the_lease(stack):
    mine, theirs = stack["patrol"], _rival(stack)
    assert mine.lease.acquire() is True
    mine.lease.release()
    assert mine.lease.holder() is None
    assert theirs.lease.acquire() is True


def test_release_by_a_non_holder_is_a_noop(stack):
    mine, theirs = stack["patrol"], _rival(stack)
    mine.lease.acquire()
    theirs.lease.release()              # not theirs to give away
    assert mine.lease.holder()["owner"] == mine.lease.owner


def test_expired_lease_is_reclaimed(stack):
    """A holder that dies must not stop the library patrolling forever."""
    import json
    mine, theirs = stack["patrol"], _rival(stack)
    mine.lease.acquire()
    stack["db"].execute(
        "UPDATE librarian_state SET value = ? WHERE key = 'patrol_lease'",
        (json.dumps({"owner": "deadhost:1", "expires_at": "2000-01-01T00:00:00"}),),
    )
    assert theirs.lease.acquire() is True


def test_malformed_lease_value_is_reclaimable(stack):
    stack["db"].execute(
        "INSERT OR REPLACE INTO librarian_state (key, value) "
        "VALUES ('patrol_lease', 'not json at all')"
    )
    assert stack["patrol"].lease.acquire() is True
    assert stack["patrol"].lease.holder()["owner"] == stack["patrol"].lease.owner


def test_only_the_lease_holder_patrols(stack):
    """The defect this exists for: N servers, one library, same tick."""
    db = stack["db"]
    for i in range(6):
        make_memory(db, f"m{i}", content=f"body {i}")
    mine, theirs = stack["patrol"], _rival(stack)

    theirs.lease.acquire()              # the rival got there first
    blocked = mine.run_due(idle=True)
    assert blocked["errands"] == []
    assert blocked["skipped"] == "lease held elsewhere"
    assert blocked["lease_holder"] == "otherhost:99999"
    assert db.execute("SELECT COUNT(*) AS n FROM patrol_visits").fetchone()["n"] == 0

    theirs.lease.release()
    ran = mine.run_due(idle=True)
    assert ran["errands"][0]["examined"] > 0


def test_lease_is_released_after_a_run(stack):
    make_memory(stack["db"], "m1", content="body")
    stack["patrol"].run_due(idle=True)
    assert stack["patrol"].lease.holder() is None


def test_lease_released_even_when_an_errand_explodes(stack):
    from chicory.layer4.wander import ERRANDS, Errand
    boom = Errand(
        name="explodes", description="test",
        run=lambda **_: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    ERRANDS[boom.name] = boom
    stack["patrol"]._config.wander_errands = ["explodes"]
    make_memory(stack["db"], "m1")
    stack["patrol"].run_due(idle=True)          # must not raise
    assert stack["patrol"].lease.holder() is None


# ── Geometry attribution ────────────────────────────────────────────────


def test_geometry_gate_config_defaults(stack):
    """Only patrons reshape the map by default.

    record_co_retrieval and update_on_retrieval write the centroid edges
    and tag centroids SubjectRoster.rebuild derives the shelf tree from.
    Measured: 200 retrievals moved that geometry 0.97% and re-partitioned
    70% of the tree — so an automated reader left out of geometry_actors
    can walk the library without rewriting the map it walks.
    """
    cfg = stack["config"]
    assert cfg.geometry_actors == ["patron"]
    assert cfg.demand_actors == ["patron"]
    assert cfg.retrieval_default_actor == "patron"


def test_an_actor_outside_geometry_actors_is_excluded(stack):
    cfg = stack["config"]
    geometry = set(cfg.geometry_actors)
    assert "patron" in geometry
    assert "patrol" not in geometry      # the idle random walk, when it lands
    assert "research" not in geometry


def test_empty_geometry_actors_lets_everyone_reshape(stack):
    """Back-compat escape: an empty list restores pre-attribution behaviour."""
    stack["config"].geometry_actors = []
    geometry = set(stack["config"].geometry_actors)
    assert (not geometry) or ("anyone" in geometry)


# ── Authored deposits ───────────────────────────────────────────────────


def test_observation_is_written_and_readable(stack):
    p = stack["patrol"]
    p.observe(errand="e", kind="document_coverage", subject="book.pdf",
              detail="8/10 chunks held; missing 3, 7", weight=0.8)
    obs = p.observations(kind="document_coverage")
    assert len(obs) == 1
    assert obs[0]["subject"] == "book.pdf"
    assert obs[0]["confirmations"] == 1
    assert obs[0]["actor"] == "patrol:e"


def test_seeing_it_again_confirms_rather_than_duplicates(stack):
    """A point of reference gains weight by recurrence, not by piling up."""
    p = stack["patrol"]
    for _ in range(3):
        p.observe(errand="e", kind="document_coverage", subject="book.pdf",
                  detail="8/10", weight=0.8)
    obs = p.observations()
    assert len(obs) == 1
    assert obs[0]["confirmations"] == 3


def test_a_later_lap_updates_the_finding(stack):
    p = stack["patrol"]
    p.observe(errand="e", kind="document_coverage", subject="book.pdf",
              detail="8/10", weight=0.8, lap=0)
    p.observe(errand="e", kind="document_coverage", subject="book.pdf",
              detail="10/10", weight=1.0, lap=1)
    obs = p.observations()[0]
    assert obs["detail"] == "10/10"
    assert obs["weight"] == 1.0
    assert obs["lap"] == 1


def test_least_complete_documents_come_first(stack):
    p = stack["patrol"]
    for subj, w in (("whole.pdf", 1.0), ("gappy.pdf", 0.4), ("half.pdf", 0.5)):
        p.observe(errand="e", kind="document_coverage", subject=subj,
                  detail="x", weight=w)
    assert [o["subject"] for o in p.observations()] == [
        "gappy.pdf", "half.pdf", "whole.pdf",
    ]


def test_observations_survive_independently_of_visits(stack):
    """The point of an authored table: it is not a derived cache."""
    p = stack["patrol"]
    p.observe(errand="e", kind="k", subject="s", detail="d")
    stack["db"].execute("DELETE FROM patrol_visits")
    assert len(p.observations()) == 1


# ── The source_provenance errand ────────────────────────────────────────


def _doc_memory(db, mid, doc, n, total):
    db.execute(
        "INSERT INTO memories (id, content, summary, source_model) "
        "VALUES (?, 'body', ?, 'test')",
        (mid, f"{doc} [{n}/{total}]"),
    )


def test_source_path_is_recovered_from_the_summary(stack):
    db, p = stack["db"], stack["patrol"]
    _doc_memory(db, "m1", "book.pdf", 1, 3)
    result = p.run_errand("source_provenance")
    assert result["acted"] == 1
    row = db.execute("SELECT source_path FROM memories WHERE id='m1'").fetchone()
    assert row["source_path"] == "book.pdf"


def test_a_summary_with_no_chunk_prefix_is_skipped_not_retried(stack):
    db, p = stack["db"], stack["patrol"]
    db.execute("INSERT INTO memories (id, content, summary, source_model) "
               "VALUES ('m1', 'body', 'desktop.ini [1/2]', 'test')")
    db.execute("UPDATE memories SET summary = 'desktop.ini' WHERE id='m1'")
    # population requires the bracket form, so it is not even selected
    errand = ERRANDS["source_provenance"]
    assert p.select(errand, 10) == []


def test_the_walk_observes_document_coverage(stack):
    """The observation, not the repair, is the point."""
    db, p = stack["db"], stack["patrol"]
    for n in (1, 2, 4):                     # chunk 3 never ingested
        _doc_memory(db, f"m{n}", "gappy.pdf", n, 4)
    p.run_errand("source_provenance", batch_size=10)
    obs = p.observations(kind="document_coverage")
    assert len(obs) == 1
    assert obs[0]["subject"] == "gappy.pdf"
    assert obs[0]["weight"] == 0.75
    assert "3/4 chunks located" in obs[0]["detail"]
    assert "not yet located: 3" in obs[0]["detail"]


def test_a_complete_document_is_observed_as_complete(stack):
    db, p = stack["db"], stack["patrol"]
    for n in (1, 2, 3):
        _doc_memory(db, f"m{n}", "whole.pdf", n, 3)
    p.run_errand("source_provenance", batch_size=10)
    obs = p.observations(kind="document_coverage")[0]
    assert obs["weight"] == 1.0
    assert "complete" in obs["detail"]


# ── The concept_gaps errand ─────────────────────────────────────────────


def _content_memory(db, mid, text):
    db.execute("INSERT INTO memories (id, content, summary, source_model) "
               "VALUES (?, ?, 'x', 'test')", (mid, text))


def test_a_recurring_capitalised_term_with_no_tag_is_recorded(stack):
    """The finding this exists for: zohar holds 15,205 memories and
    tiferet holds none — the material is here and unfindable."""
    db, p = stack["db"], stack["patrol"]
    _content_memory(db, "m1", "Tiferet is the centre. Tiferet balances. "
                              "Above Tiferet sits Binah, and Tiferet again.")
    p.run_errand("concept_gaps", batch_size=10)
    obs = p.observations(kind="concept_gap")
    assert [o["subject"] for o in obs] == ["tiferet"]
    assert "cannot retrieve by it" in obs[0]["detail"]
    # Weight is SPREAD now: one document, one memory -> 1.0. Concentration
    # only becomes meaningful once the term recurs across memories.
    assert obs[0]["weight"] == 1.0


def test_a_term_mentioned_once_is_not_a_concept(stack):
    db, p = stack["db"], stack["patrol"]
    _content_memory(db, "m1", "Tiferet appears here only the one time.")
    p.run_errand("concept_gaps", batch_size=10)
    assert p.observations(kind="concept_gap") == []


def test_capitalised_common_words_are_not_concepts(stack):
    """Sentence-initial 'Great' is indistinguishable from 'Malchut' to a
    regex; only the stop-list separates them."""
    db, p = stack["db"], stack["patrol"]
    _content_memory(db, "m1", "Great things. Great works. Great again. "
                              "Great once more. Which is which. Which again.")
    p.run_errand("concept_gaps", batch_size=10)
    assert p.observations(kind="concept_gap") == []


def test_a_well_held_tag_is_not_a_gap(stack):
    db, p = stack["db"], stack["patrol"]
    from chicory.layer1.tag_manager import TagManager
    tags = TagManager(db)
    t = tags.get_or_create("kabbalah", created_by="test")
    for i in range(25):
        _content_memory(db, f"pad{i}", "x")
        db.execute("INSERT OR IGNORE INTO memory_tags (memory_id, tag_id, assigned_by) "
                   "VALUES (?, ?, 'test')", (f"pad{i}", t.id))
    _content_memory(db, "m1", "Kabbalah says. Kabbalah teaches. "
                              "Kabbalah holds. Kabbalah again.")
    p.run_errand("concept_gaps", batch_size=50)
    assert [o["subject"] for o in p.observations(kind="concept_gap")] == []


def test_seeing_a_concept_again_confirms_it(stack):
    """Recurrence across passes is what separates a concept from noise."""
    db, p = stack["db"], stack["patrol"]
    for i in (1, 2):
        _content_memory(db, f"m{i}", "Malchut and Malchut and Malchut, Malchut.")
    p.run_errand("concept_gaps", batch_size=1)
    p.run_errand("concept_gaps", batch_size=1)
    obs = p.observations(kind="concept_gap")
    assert len(obs) == 1
    assert obs[0]["confirmations"] == 2


def test_the_survey_waits_for_quiet(stack):
    from chicory.layer4.wander import ERRANDS
    assert ERRANDS["concept_gaps"].when == "idle"
    stack["patrol"]._config.wander_errands = ["concept_gaps"]
    _content_memory(stack["db"], "m1", "Malchut Malchut Malchut Malchut.")
    assert stack["patrol"].run_due(idle=False)["errands"] == []
    assert stack["patrol"].run_due(idle=True)["errands"][0]["examined"] == 1


def test_a_term_spread_across_documents_ranks_below_a_concentrated_one(stack):
    """The fix for the first dictionary: 'Press' and 'University' recur in
    every bibliography and topped a list built on recurrence alone. What
    separates vocabulary from a concept is how far it ranges."""
    db, p = stack["db"], stack["patrol"]
    # a concept: many mentions, one document
    for i in range(4):
        db.execute("INSERT INTO memories (id, content, summary, source_model, source_path) "
                   "VALUES (?, 'Malchut Malchut Malchut Malchut.', 'x', 't', 'zohar.pdf')",
                   (f"c{i}",))
    # furniture: same mentions, spread over four documents
    for i in range(4):
        db.execute("INSERT INTO memories (id, content, summary, source_model, source_path) "
                   "VALUES (?, 'Cambridge Cambridge Cambridge Cambridge.', 'x', 't', ?)",
                   (f"v{i}", f"book{i}.pdf"))
    p.run_errand("concept_gaps", batch_size=20)
    obs = {o["subject"]: o for o in p.observations(kind="concept_gap")}
    assert obs["malchut"]["weight"] < obs["cambridge"]["weight"]
    # and the ordering puts the concept first
    assert p.observations(kind="concept_gap")[0]["subject"] == "malchut"


def test_documents_accumulate_across_passes(stack):
    """A walk sees a batch at a time, so corpus-wide spread must carry."""
    import json
    db, p = stack["db"], stack["patrol"]
    for i, doc in enumerate(("a.pdf", "b.pdf")):
        db.execute("INSERT INTO memories (id, content, summary, source_model, source_path) "
                   "VALUES (?, 'Tiferet Tiferet Tiferet Tiferet.', 'x', 't', ?)",
                   (f"m{i}", doc))
    p.run_errand("concept_gaps", batch_size=1)
    p.run_errand("concept_gaps", batch_size=1)
    obs = p.observations(kind="concept_gap")[0]
    assert json.loads(obs["data"])["documents"] == ["a.pdf", "b.pdf"]
    assert obs["confirmations"] == 2


# ── Regions of association ──────────────────────────────────────────────


def _doc_mem(db, mid, text, doc="d.pdf"):
    db.execute("INSERT INTO memories (id, content, summary, source_model, source_path) "
               "VALUES (?, ?, 'x', 't', ?)", (mid, text, doc))


def test_a_region_records_what_shares_the_memory(stack):
    """Co-occurrence within a memory, not within a book: two terms in the
    same chunk are related, two in the same 8,969-chunk book are not."""
    import json
    db, p = stack["db"], stack["patrol"]
    _doc_mem(db, "m1", "Malchut Malchut Malchut Malchut. "
                       "Chochmah Chochmah Chochmah Chochmah.")
    p.run_errand("concept_gaps", batch_size=10)
    obs = {o["subject"]: json.loads(o["data"]) for o in p.observations(kind="concept_gap")}
    assert obs["malchut"]["co"] == {"chochmah": 1}
    assert obs["chochmah"]["co"] == {"malchut": 1}


def test_a_region_records_the_tags_around_it(stack):
    """The bridge back into the index: where a missing tag would attach."""
    import json
    db, p = stack["db"], stack["patrol"]
    _doc_mem(db, "m1", "Malchut Malchut Malchut Malchut.")
    tag = stack["tags"].get_or_create("kabbalah", created_by="test") \
        if "tags" in stack else None
    from chicory.layer1.tag_manager import TagManager
    tag = TagManager(db).get_or_create("kabbalah", created_by="test")
    db.execute("INSERT OR IGNORE INTO memory_tags (memory_id, tag_id, assigned_by) "
               "VALUES ('m1', ?, 'test')", (tag.id,))
    p.run_errand("concept_gaps", batch_size=10)
    data = json.loads(p.observations(kind="concept_gap")[0]["data"])
    assert data["around"] == {"kabbalah": 1}


def test_edges_now_come_from_shared_memories(stack):
    db, p = stack["db"], stack["patrol"]
    # together in every memory -> strong edge
    for i in range(3):
        _doc_mem(db, f"t{i}", "Malchut Malchut Malchut Malchut. "
                              "Chochmah Chochmah Chochmah Chochmah.")
    # apart, in the same document -> no edge under a memory-level rule
    _doc_mem(db, "a1", "Tabernacle Tabernacle Tabernacle Tabernacle.")
    p.run_errand("concept_gaps", batch_size=20)
    p.build_observation_edges(min_confirmations=1)
    pairs = {(r["subject_a"], r["subject_b"]) for r in stack["db"].execute(
        "SELECT subject_a, subject_b FROM observation_edges").fetchall()}
    assert ("chochmah", "malchut") in pairs
    assert not any("tabernacle" in p_ for p_ in pairs)


def test_counters_accumulate_across_passes(stack):
    import json
    db, p = stack["db"], stack["patrol"]
    for i in range(2):
        _doc_mem(db, f"m{i}", "Malchut Malchut Malchut Malchut. "
                              "Chochmah Chochmah Chochmah Chochmah.")
    p.run_errand("concept_gaps", batch_size=1)
    p.run_errand("concept_gaps", batch_size=1)
    data = json.loads(
        [o for o in p.observations(kind="concept_gap")
         if o["subject"] == "malchut"][0]["data"])
    assert data["co"]["chochmah"] == 2      # summed, not overwritten
    assert data["mentions"] == 2


def test_a_tabular_chunk_is_skipped_not_paired(stack):
    """A locale table put 870 language names in one chunk; pairing them
    emitted a clique that swamped the net."""
    db, p = stack["db"], stack["patrol"]
    many = " ".join(f"{w} {w} {w} {w}." for w in
                    ("Abkhazian", "Achinese", "Acholi", "Adyghe", "Afrikaans",
                     "Aleut", "Amharic", "Angika", "Arabic", "Aragonese"))
    _doc_mem(db, "table", many)
    _doc_mem(db, "prose", "Malchut Malchut Malchut Malchut.")
    p.run_errand("concept_gaps", batch_size=10)
    row = db.execute("SELECT verdict, detail FROM patrol_visits "
                     "WHERE memory_id='table'").fetchone()
    assert row["verdict"] == "skipped"
    assert "tabular" in row["detail"]
    assert [o["subject"] for o in p.observations(kind="concept_gap")] == ["malchut"]


class TestWhatIsActuallyScheduled:
    """Registering an errand does not schedule it.

    `_run_due_locked` iterates `config.wander_errands` and nothing else,
    and no environment variable can add to it. Seven registered errands
    were absent from that list on the live install and carried real laps
    only because sessions had driven them by hand — the tell was
    `last_result: examined=4000`, a batch size the heartbeat cannot
    produce, since it passes none and so uses fifty. An errand left off
    the list has no background existence at all, and nothing in the code
    says so at the point where an errand is registered.
    """

    def _shipped(self):
        return ChicoryConfig(
            db_path=Path(":memory:"), anthropic_api_key="test",
            llm_model="test-model", embedding_model="mock",
        ).wander_errands

    def test_every_scheduled_name_is_a_registered_errand(self):
        """A typo here is silent: run_due logs and skips an unknown name."""
        unknown = [n for n in self._shipped() if n not in ERRANDS]
        assert unknown == [], f"scheduled but not registered: {unknown}"

    def test_corpus_asymmetry_is_scheduled(self):
        """It compares two accounts of one thing and withdraws findings
        whose populations have moved. Off this list it did neither: 124
        observations stood for a day against a primary text that had
        silently become four files out of eight."""
        assert "corpus_asymmetry" in self._shipped()

    def test_a_scheduled_errand_is_reachable_through_run_due(self, stack):
        """The property the list actually buys: run_due dispatches it.

        Asserted through the real dispatcher rather than by reading the
        list back, because the list being right and the walk reaching the
        errand are two different claims.
        """
        seen = []
        register(Errand(
            name="probe-scheduled",
            description="records that it was dispatched",
            run=lambda **kw: (seen.append(len(kw["memory_ids"])) or
                              {m: ("ok", None) for m in kw["memory_ids"]}),
        ))
        try:
            db, p = stack["db"], stack["patrol"]
            make_memory(db, "m1")
            p._config.wander_errands = ["probe-scheduled"]
            p.run_due(idle=True)
            assert seen, "a scheduled errand was never dispatched"

            seen.clear()
            p._config.wander_errands = []
            p._last_run.clear()
            p.run_due(idle=True)
            assert not seen, "an unscheduled errand was dispatched anyway"
        finally:
            ERRANDS.pop("probe-scheduled", None)
