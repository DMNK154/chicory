"""Episodic edge decay and archival measure inactivity, not status alone.

The age comparison used to write ``strftime('%%Y...')``; the sqlite3
driver does no %-formatting, so SQLite saw ``%%`` and produced the
literal text '%Y-%m-%dT...'.  No timestamp sorts below '%', so the age
test was always false: never-activated edges went candidate → decaying →
archived in one prune() regardless of age, and activated edges never
decayed at all.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer4.episodic_tensor import EpisodicTensor

WINDOW_HOURS = 100.0


@pytest.fixture
def tensor():
    config = ChicoryConfig(
        db_path=Path(":memory:"),
        embedding_model="mock",
        embedding_dimension=32,
        episodic_decay_inactive_hours=WINDOW_HOURS,
    )
    db = DatabaseEngine(config)
    db.connect()
    apply_schema(db)
    yield db, EpisodicTensor(db, config)
    db.close()


def _edge(db, status: str, *, created_hours_ago: float,
          activated_hours_ago: float | None) -> tuple[str, str]:
    mids = sorted(str(uuid.uuid4()) for _ in range(2))
    for mid in mids:
        db.execute(
            "INSERT INTO memories (id, content, summary, source_model) "
            "VALUES (?, 'c', 's', 'test')",
            (mid,),
        )
    activated = (
        None if activated_hours_ago is None
        else db.execute(
            "SELECT strftime('%Y-%m-%dT%H:%M:%f', 'now', ?)",
            (f"-{activated_hours_ago} hours",),
        ).fetchone()[0]
    )
    db.execute(
        """INSERT INTO memory_relational_tensor
           (memory_a_id, memory_b_id, semantic_strength, edge_status,
            created_at, last_activated_at)
           VALUES (?, ?, 1.0, ?,
                   strftime('%Y-%m-%dT%H:%M:%f', 'now', ?), ?)""",
        (mids[0], mids[1], status, f"-{created_hours_ago} hours", activated),
    )
    db.connection.commit()
    return mids[0], mids[1]


def _status(db, key) -> str:
    return db.execute(
        "SELECT edge_status FROM memory_relational_tensor "
        "WHERE memory_a_id = ? AND memory_b_id = ?",
        key,
    ).fetchone()["edge_status"]


def test_recently_activated_edge_keeps_its_status(tensor):
    db, et = tensor
    key = _edge(db, "warm", created_hours_ago=500, activated_hours_ago=1)
    et.prune()
    assert _status(db, key) == "warm"


def test_edge_inactive_past_the_window_starts_decaying(tensor):
    db, et = tensor
    key = _edge(db, "warm", created_hours_ago=500,
                activated_hours_ago=WINDOW_HOURS * 1.5)
    et.prune()
    assert _status(db, key) == "decaying"


def test_archival_needs_twice_the_window(tensor):
    db, et = tensor
    young = _edge(db, "decaying", created_hours_ago=500,
                  activated_hours_ago=WINDOW_HOURS * 1.5)
    old = _edge(db, "decaying", created_hours_ago=500,
                activated_hours_ago=WINDOW_HOURS * 2.5)
    et.prune()
    assert _status(db, young) == "decaying"
    assert _status(db, old) == "archived"


def test_never_activated_edge_is_aged_from_creation(tensor):
    db, et = tensor
    fresh = _edge(db, "candidate", created_hours_ago=1, activated_hours_ago=None)
    stale = _edge(db, "candidate", created_hours_ago=WINDOW_HOURS * 1.5,
                  activated_hours_ago=None)
    et.prune()
    assert _status(db, fresh) == "candidate"
    assert _status(db, stale) == "decaying"


def test_one_prune_does_not_archive_a_fresh_edge(tensor):
    db, et = tensor
    key = _edge(db, "candidate", created_hours_ago=0, activated_hours_ago=None)
    et.prune()
    et.prune()
    assert _status(db, key) == "candidate"


def test_mature_edges_never_decay(tensor):
    db, et = tensor
    key = _edge(db, "mature", created_hours_ago=10_000,
                activated_hours_ago=10_000)
    et.prune()
    assert _status(db, key) == "mature"
