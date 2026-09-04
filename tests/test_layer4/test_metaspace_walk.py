"""Tests for walking the metadata rather than the material.

The distributed cursor was written for memories, but nothing about it is
specific to them: it needs a table, a string id, and a tiebreak.  These
pin the generalisation, and the two errands that use it — both of which
RECORD and neither of which acts, because the survey that motivated them
also showed why acting automatically would be wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer3.centroid_subgraph import _array_to_blob
from chicory.layer4.wander import DOMAINS, Errand, Patrol
from tests.conftest import MockEmbeddingEngine

DIM = 32


@pytest.fixture
def stack():
    config = ChicoryConfig(
        db_path=Path(":memory:"),
        anthropic_api_key="test",
        llm_model="test-model",
        embedding_model="mock",
        embedding_dimension=DIM,
        wander_batch_size=50,
        wander_interval_minutes=0.0,
        wander_errands=["tag_seams", "tag_twins"],
        # Disabled; the hub rule is tested explicitly below. The errand
        # applies the roster's own cap with no floor, so on a fixture
        # library of a handful of memories every tag looks like a hub.
        subject_hub_tag_max_fraction=999.0,
    )
    db = DatabaseEngine(config)
    db.connect()
    apply_schema(db)
    emb = MockEmbeddingEngine(dimension=DIM)
    emb.set_db(db)
    yield {"config": config, "db": db, "patrol": Patrol(config, db, emb)}
    db.close()


def _tag(db, name: str) -> int:
    db.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (name,))
    return db.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()["id"]


def _memory(db, mid: str, content: str = "text") -> None:
    db.execute(
        "INSERT OR IGNORE INTO memories (id, content, source_model, ingestion_tier) "
        "VALUES (?, ?, 'test', 'critical')", (mid, content))


def _attach(db, mid: str, tag_id: int) -> None:
    _memory(db, mid)
    db.execute(
        "INSERT OR IGNORE INTO memory_tags (memory_id, tag_id, assigned_by) "
        "VALUES (?, ?, 'test')", (mid, tag_id))


def _centroid(db, tag_id: int, n: int = 3) -> None:
    v = np.zeros(DIM, dtype=np.float32)
    v[tag_id % DIM] = 1.0
    db.execute(
        "INSERT OR REPLACE INTO tag_centroids (tag_id, centroid, memory_count) "
        "VALUES (?, ?, ?)", (tag_id, _array_to_blob(v), n))


def _edge(db, a: int, b: int) -> None:
    db.execute(
        "INSERT OR REPLACE INTO centroid_edges "
        "(tag_a_id, tag_b_id, edge_strength, co_retrieval_count) VALUES (?,?,0.5,5)",
        (min(a, b), max(a, b)))


def _shelf(db, name: str, tag_ids: list[int]) -> None:
    db.execute(
        "INSERT INTO subjects (name, member_tag_ids, memory_count, status, depth, path) "
        "VALUES (?, ?, 1, 'active', 0, ?)", (name, json.dumps(tag_ids), name))


def _obs(db, kind: str) -> dict[str, dict]:
    return {r["subject"]: dict(r) for r in db.execute(
        "SELECT subject, detail, weight, data FROM patrol_observations "
        "WHERE kind = ? AND retracted = 0", (kind,))}


class TestTheCursorGeneralises:
    def test_the_tags_domain_walks_tags_not_memories(self, stack):
        db = stack["db"]
        _memory(db, "m1")
        ids = [_tag(db, n) for n in ("alpha", "beta", "gamma")]
        db.connection.commit()
        errand = Errand(name="probe", description="", run=lambda **k: {},
                        domain="tags")
        got = stack["patrol"].select_with_laps(errand, limit=10)
        assert {o for o, _ in got} == {str(i) for i in ids}
        assert "m1" not in {o for o, _ in got}

    def test_a_tag_walk_and_a_memory_walk_do_not_collide(self, stack):
        """patrol_visits is keyed (errand, id); the errand name keeps them apart."""
        db = stack["db"]
        _memory(db, "1")                 # a memory whose id looks like a tag id
        tid = _tag(db, "alpha")
        db.connection.commit()
        p = stack["patrol"]
        p.record_visits("mem_errand", 0, {"1": ("ok", None)})
        p.record_visits("tag_errand", 0, {str(tid): ("ok", None)})
        db.connection.commit()
        rows = db.execute(
            "SELECT errand, memory_id FROM patrol_visits ORDER BY errand").fetchall()
        assert len(rows) == 2
        assert {(r["errand"], r["memory_id"]) for r in rows} == {
            ("mem_errand", "1"), ("tag_errand", str(tid))}

    def test_unvisited_tags_lead_the_queue(self, stack):
        db = stack["db"]
        first, second = _tag(db, "alpha"), _tag(db, "beta")
        db.connection.commit()
        errand = Errand(name="probe", description="", run=lambda **k: {},
                        domain="tags")
        stack["patrol"].record_visits("probe", 0, {str(first): ("ok", None)})
        db.connection.commit()
        got = [o for o, _ in stack["patrol"].select_with_laps(errand, limit=10)]
        assert got[0] == str(second), "an unvisited tag must lead"

    def test_the_memories_domain_is_still_the_default(self):
        e = Errand(name="x", description="", run=lambda **k: {})
        assert e.domain == "memories"
        assert DOMAINS["memories"].table.startswith("memories")


class TestTagSeams:
    def test_a_placed_tag_raises_nothing(self, stack):
        db = stack["db"]
        a, b = _tag(db, "kabbalah"), _tag(db, "zohar")
        for t in (a, b):
            _attach(db, f"m{t}", t)
            _centroid(db, t)
        _edge(db, a, b)
        _shelf(db, "mysticism", [a, b])
        db.connection.commit()
        stack["patrol"].run_errand("tag_seams")
        assert _obs(db, "tag_seam") == {}

    def test_it_names_every_way_a_tag_is_unreachable(self, stack):
        db = stack["db"]
        lonely = _tag(db, "malchut")
        _attach(db, "m1", lonely)
        db.connection.commit()                       # no centroid, no edge, no shelf
        stack["patrol"].run_errand("tag_seams")
        seen = _obs(db, "tag_seam")
        assert "malchut" in seen
        faults = json.loads(seen["malchut"]["data"])["faults"]
        assert "no centroid" in faults
        assert "no edge" in faults
        assert "on no shelf" in faults

    def test_a_clock_stamp_is_not_a_seam(self, stack):
        db = stack["db"]
        t = _tag(db, "day-2026-07-18")
        _attach(db, "m1", t)
        db.connection.commit()
        stack["patrol"].run_errand("tag_seams")
        assert _obs(db, "tag_seam") == {}


class TestRedundantParent:
    def test_a_contained_fragment_is_flagged(self, stack):
        db = stack["db"]
        whole, frag = _tag(db, "machine-learning"), _tag(db, "learning")
        for mid in ("m1", "m2"):
            _attach(db, mid, whole)
            _attach(db, mid, frag)
        _centroid(db, whole); _centroid(db, frag)
        _edge(db, whole, frag)
        _shelf(db, "s", [whole, frag])
        db.connection.commit()
        stack["patrol"].run_errand("tag_seams")
        seen = _obs(db, "tag_seam")
        assert "learning" in seen
        assert any("duplicates machine-learning" in f
                   for f in json.loads(seen["learning"]["data"])["faults"])

    def test_a_fragment_with_its_own_memories_is_left_alone(self, stack):
        """`data` holds 4,103 memories outside `lab-data`; it is a real tag."""
        db = stack["db"]
        whole, frag = _tag(db, "lab-data"), _tag(db, "data")
        _attach(db, "m1", whole); _attach(db, "m1", frag)
        _attach(db, "m2", frag)                  # outside the parent
        _centroid(db, whole); _centroid(db, frag)
        _edge(db, whole, frag)
        _shelf(db, "s", [whole, frag])
        db.connection.commit()
        stack["patrol"].run_errand("tag_seams")
        assert _obs(db, "tag_seam") == {}, "containment, not name shape"


class TestTagTwins:
    def _pair(self, db, a: str, b: str, shared: int, only_a: int, only_b: int):
        ta, tb = _tag(db, a), _tag(db, b)
        for i in range(shared):
            _attach(db, f"s{i}", ta); _attach(db, f"s{i}", tb)
        for i in range(only_a):
            _attach(db, f"a{i}", ta)
        for i in range(only_b):
            _attach(db, f"b{i}", tb)
        _centroid(db, ta); _centroid(db, tb)
        db.connection.commit()
        return ta, tb

    def test_the_same_tag_twice_is_reported_as_near_certain(self, stack):
        db = stack["db"]
        self._pair(db, "tefilin", "tfilin", shared=6, only_a=0, only_b=0)
        stack["patrol"].run_errand("tag_twins")
        seen = _obs(db, "tag_twin")
        assert "tefilin" in seen or "tfilin" in seen
        rec = seen.get("tefilin") or seen["tfilin"]
        assert "same tag twice" in rec["detail"]

    def test_lookalikes_on_different_memories_are_offered_not_asserted(self, stack):
        db = stack["db"]
        self._pair(db, "device-dependent", "device-independent",
                   shared=0, only_a=4, only_b=4)
        stack["patrol"].run_errand("tag_twins")
        rec = (_obs(db, "tag_twin").get("device-dependent")
               or _obs(db, "tag_twin").get("device-independent"))
        assert rec is not None
        # names alike, memories disjoint -> must NOT read as a typo
        assert "same tag twice" not in rec["detail"]
        assert "vocabulary call" in rec["detail"]
        twins = json.loads(rec["data"])["twins"]
        assert twins[0]["overlap"] == 0.0

    def test_it_only_records(self, stack):
        """No merge, no edge, no tag deleted — the survey showed why."""
        db = stack["db"]
        ta, tb = self._pair(db, "tefilin", "tfilin", shared=6, only_a=0, only_b=0)
        before = db.execute("SELECT COUNT(*) n FROM memory_tags").fetchone()["n"]
        stack["patrol"].run_errand("tag_twins")
        assert db.execute(
            "SELECT COUNT(*) n FROM memory_tags").fetchone()["n"] == before
        assert db.execute(
            "SELECT COUNT(*) n FROM tags WHERE is_active=1").fetchone()["n"] == 2


class TestHubTags:
    """A tag on a fifth of the library is off the tree BY DESIGN."""

    def _stack_with_cap(self, tmp_fraction: float):
        config = ChicoryConfig(
            db_path=Path(":memory:"), anthropic_api_key="test",
            llm_model="test-model", embedding_model="mock",
            embedding_dimension=DIM, wander_batch_size=50,
            wander_interval_minutes=0.0, wander_errands=["tag_seams"],
            subject_hub_tag_max_fraction=tmp_fraction,
        )
        db = DatabaseEngine(config)
        db.connect()
        apply_schema(db)
        emb = MockEmbeddingEngine(dimension=DIM)
        emb.set_db(db)
        return {"db": db, "patrol": Patrol(config, db, emb)}

    def test_a_hub_tag_is_not_reported_as_a_seam(self):
        s = self._stack_with_cap(0.2)
        db = s["db"]
        hub = _tag(db, "document")
        for i in range(10):                    # on every memory in the library
            _attach(db, f"m{i}", hub)
        db.connection.commit()
        s["patrol"].run_errand("tag_seams")
        assert _obs(db, "tag_seam") == {}, (
            "subject_hub_tag_max_fraction exists to keep a tag like this off "
            "the tree; that is the rule working, not a seam"
        )
        db.close()

    def test_a_tag_under_the_cap_is_still_reported(self):
        s = self._stack_with_cap(0.9)
        db = s["db"]
        small = _tag(db, "malchut")
        for i in range(10):
            _memory(db, f"m{i}")
        _attach(db, "m0", small)               # 1 of 10 -> under the cap
        db.connection.commit()
        s["patrol"].run_errand("tag_seams")
        assert "malchut" in _obs(db, "tag_seam")
        db.close()


class TestObserveMerging:
    def test_a_list_of_records_is_replaced_not_unioned(self, stack):
        """`observe` merges lists as a set union, which needs hashables.

        `documents` is a list of names and accumulates correctly. A list
        of measurement records is not unionable, and merging two readings
        would invent a state neither pass observed — so the newer one
        stands.
        """
        p, db = stack["patrol"], stack["db"]
        for overlap in (0.2, 0.9):
            p.observe(errand="e", kind="k", subject="s", detail="d", weight=0.0,
                      data={"twins": [{"other": "x", "overlap": overlap}]},
                      lap=0)
        db.connection.commit()
        rec = _obs(db, "k")["s"]
        twins = json.loads(rec["data"])["twins"]
        assert len(twins) == 1
        assert twins[0]["overlap"] == 0.9, "the newer reading should stand"

    def test_a_list_of_names_still_accumulates(self, stack):
        p, db = stack["patrol"], stack["db"]
        p.observe(errand="e", kind="k2", subject="s", detail="d", weight=0.0,
                  data={"documents": ["a.pdf"]}, lap=0)
        p.observe(errand="e", kind="k2", subject="s", detail="d", weight=0.0,
                  data={"documents": ["b.pdf"]}, lap=0)
        db.connection.commit()
        docs = json.loads(_obs(db, "k2")["s"]["data"])["documents"]
        assert docs == ["a.pdf", "b.pdf"]


class TestEveryExaminedObjectIsStamped:
    """The invariant that keeps a walk advancing.

    An object the cursor offers and the errand does not stamp comes back
    at the head of the queue on the next batch, forever. Measured on the
    live library: tag_twins skipped short-named tags silently and was
    still at 94% coverage after 72 runs.
    """

    def test_short_named_tags_do_not_block_the_queue(self, stack):
        db, p = stack["db"], stack["patrol"]
        short = [_tag(db, n) for n in ("ai", "ml", "os")]      # <= 3 chars
        long_ = [_tag(db, n) for n in ("kabbalah", "vedanta")]
        for t in short + long_:
            _attach(db, f"m{t}", t)
        db.connection.commit()

        p.run_errand("tag_twins", batch_size=50)
        db.connection.commit()
        stamped = {r["memory_id"] for r in db.execute(
            "SELECT memory_id FROM patrol_visits WHERE errand='tag_twins'")}
        assert stamped == {str(t) for t in short + long_}, (
            "every id the cursor offered must carry a visit row"
        )

    def test_a_second_batch_advances_past_them(self, stack):
        db, p = stack["db"], stack["patrol"]
        for n in ("ai", "ml", "os", "kabbalah"):
            t = _tag(db, n)
            _attach(db, f"m{t}", t)
        db.connection.commit()
        p.run_errand("tag_twins", batch_size=2)
        db.connection.commit()
        first = {r["memory_id"] for r in db.execute(
            "SELECT memory_id FROM patrol_visits WHERE errand='tag_twins'")}
        p.run_errand("tag_twins", batch_size=2)
        db.connection.commit()
        second = {r["memory_id"] for r in db.execute(
            "SELECT memory_id FROM patrol_visits WHERE errand='tag_twins'")}
        assert len(second) > len(first), "the walk must reach new tags"
