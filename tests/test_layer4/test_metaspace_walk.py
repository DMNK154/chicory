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
    yield {"config": config, "db": db, "emb": emb,
           "patrol": Patrol(config, db, emb)}
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
        assert "no qualifying edge" in faults
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


class TestQualifyingEdges:
    """An edge the roster will not walk is not an edge, for this purpose.

    _connected_components keeps an edge below subject_edge_merge_min_count
    only when both ends sat in the same prior root. A tag that has never
    been shelved therefore needs a QUALIFYING edge to enter the tree at
    all. Measured on the live library: `archaeology`, 1,659 memories,
    carries seven edges all at co_retrieval_count 1 or 2 and is invisible
    to the roster while looking perfectly connected.
    """

    def _weak_edge(self, db, a, b, count):
        db.execute(
            "INSERT OR REPLACE INTO centroid_edges "
            "(tag_a_id, tag_b_id, edge_strength, co_retrieval_count) "
            "VALUES (?,?,0.5,?)", (min(a, b), max(a, b), count))

    def test_an_edge_below_the_merge_floor_does_not_count(self, stack):
        db = stack["db"]
        a, b = _tag(db, "archaeology"), _tag(db, "knowledge")
        for t in (a, b):
            _attach(db, f"m{t}", t)
            _centroid(db, t)
        self._weak_edge(db, a, b, count=2)      # below merge_min of 3
        _shelf(db, "s", [a, b])
        db.connection.commit()
        stack["patrol"].run_errand("tag_seams")
        seen = _obs(db, "tag_seam")
        assert "archaeology" in seen
        assert "no qualifying edge" in json.loads(
            seen["archaeology"]["data"])["faults"]

    def test_an_edge_at_the_floor_counts(self, stack):
        db = stack["db"]
        a, b = _tag(db, "archaeology"), _tag(db, "knowledge")
        for t in (a, b):
            _attach(db, f"m{t}", t)
            _centroid(db, t)
        self._weak_edge(db, a, b, count=3)
        _shelf(db, "s", [a, b])
        db.connection.commit()
        stack["patrol"].run_errand("tag_seams")
        assert _obs(db, "tag_seam") == {}


class TestSeamAudit:
    """The walk auditing its own record.

    An observation is a claim made at a moment, and the library moves
    underneath it. Within an hour of merging seven duplicate tags, 22
    standing seams pointed at tags that no longer existed — the same rot
    that left `malchut` claiming 3,948 mentions while its tag held 993.
    """

    def _seam(self, db, tag_id: int, name: str, faults: list[str], held: int = 5):
        db.execute(
            "INSERT INTO patrol_observations (errand, kind, subject, detail, "
            "weight, data) VALUES ('tag_seams','tag_seam',?,?,?,?)",
            (name, "; ".join(faults), 0.1,
             json.dumps({"tag_id": tag_id, "held": held, "faults": faults})))
        db.connection.commit()

    def _live(self, db, n=20):
        for i in range(n):
            _memory(db, f"bulk{i}")
        db.connection.commit()

    def test_a_seam_for_a_merged_tag_is_retracted(self, stack):
        db = stack["db"]
        gone, keep = _tag(db, "gvurah"), _tag(db, "gevurah")
        _attach(db, "m1", keep)
        self._seam(db, gone, "gvurah", ["no qualifying edge"])
        db.execute("UPDATE tags SET is_active=0, merged_into=? WHERE id=?",
                   (keep, gone))
        db.connection.commit()
        stack["patrol"].run_errand("seam_audit")
        assert _obs(db, "tag_seam") == {}, "a seam for a merged tag must retract"

    def test_a_seam_whose_faults_cleared_is_retracted(self, stack):
        db = stack["db"]
        a, b = _tag(db, "aramaic"), _tag(db, "hebrew")
        for t in (a, b):
            _attach(db, f"m{t}", t)
            _centroid(db, t)
        _edge(db, a, b)                       # count 5, clears the merge floor
        _shelf(db, "s", [a, b])
        self._seam(db, a, "aramaic", ["no centroid", "no qualifying edge"])
        stack["patrol"].run_errand("seam_audit")
        assert _obs(db, "tag_seam") == {}

    def test_a_still_true_seam_is_restated_not_dropped(self, stack):
        db = stack["db"]
        lonely = _tag(db, "thesis")
        _attach(db, "m1", lonely)
        _centroid(db, lonely)
        self._seam(db, lonely, "thesis", ["no centroid"])
        stack["patrol"].run_errand("seam_audit")
        seen = _obs(db, "tag_seam")
        assert "thesis" in seen
        faults = json.loads(seen["thesis"]["data"])["faults"]
        assert "no centroid" not in faults          # re-measured: it has one
        assert "no qualifying edge" in faults       # and this is still true

    def test_it_repairs_a_missing_centroid(self, stack):
        db, emb = stack["db"], stack["emb"]
        t = _tag(db, "archaeology")
        _attach(db, "m1", t)
        emb.store_cached("m1", emb.embed("some archaeology"))
        db.connection.commit()
        self._seam(db, t, "archaeology", ["no centroid"])
        assert db.execute("SELECT COUNT(*) n FROM tag_centroids").fetchone()["n"] == 0
        stack["patrol"].run_errand("seam_audit")
        assert db.execute(
            "SELECT COUNT(*) n FROM tag_centroids WHERE tag_id=?", (t,)
        ).fetchone()["n"] == 1, "a centroid is a pure function of what the tag holds"

    def test_it_never_seeds_an_edge_or_merges_a_tag(self, stack):
        """Those change what the roster DERIVES; an unattended errand must not."""
        db = stack["db"]
        a, b = _tag(db, "plato"), _tag(db, "greek")
        for t in (a, b):
            _attach(db, "shared", t)
            _centroid(db, t)
        self._seam(db, a, "plato", ["no qualifying edge", "on no shelf"])
        edges = db.execute("SELECT COUNT(*) n FROM centroid_edges").fetchone()["n"]
        tags = db.execute("SELECT COUNT(*) n FROM tags WHERE is_active=1").fetchone()["n"]
        stack["patrol"].run_errand("seam_audit")
        assert db.execute("SELECT COUNT(*) n FROM centroid_edges").fetchone()["n"] == edges
        assert db.execute(
            "SELECT COUNT(*) n FROM tags WHERE is_active=1").fetchone()["n"] == tags
        assert "plato" in _obs(db, "tag_seam"), "and the seam still stands"

    def test_a_tag_that_became_a_hub_is_retracted(self, stack):
        db = stack["db"]
        hub = _tag(db, "document")
        for i in range(10):
            _attach(db, f"h{i}", hub)
        self._seam(db, hub, "document", ["on no shelf"], held=10)
        stack["config"].subject_hub_tag_max_fraction = 0.2
        stack["patrol"].run_errand("seam_audit")
        assert _obs(db, "tag_seam") == {}


class TestTagGapVariants:
    """The blind spot between two bridges.

    meta_shelves compares gap<->gap; tag_twins compares tag<->tag; nothing
    compared tag<->gap. And a spelling leaves the gap net the moment it
    becomes a tag, so fixing one variant made its twin invisible to both.
    Measured: three Zohar editions, one indexed, `chokhmah` / `malkhut` /
    `sefirot` / `hesed` recorded as gaps and never joined to anything.
    """

    def _gap(self, db, subject: str, around: dict, mentions: int = 200):
        db.execute(
            "INSERT INTO patrol_observations (errand, kind, subject, detail, "
            "weight, data, confirmations) "
            "VALUES ('concept_gaps','concept_gap',?,'',0.1,?,4)",
            (subject, json.dumps({"around": around, "mentions": mentions,
                                  "documents": ["ed_b.pdf"], "co": {"binah": 3}})))
        db.connection.commit()

    def _indexed(self, db, tag: str, context: list[str], n: int = 6):
        t = _tag(db, tag)
        others = [_tag(db, c) for c in context]
        for i in range(n):
            mid = f"{tag}{i}"
            _attach(db, mid, t)
            for o in others:
                _attach(db, mid, o)
        _centroid(db, t)
        db.connection.commit()
        return t

    def test_it_finds_the_unindexed_spelling(self, stack):
        db = stack["db"]
        self._indexed(db, "chochmah", ["kabbalah", "zohar", "hebrew"])
        self._gap(db, "chokhmah", {"kabbalah": 40, "zohar": 30, "hebrew": 20})
        stack["patrol"].run_errand("tag_gap_variants")
        seen = _obs(db, "tag_gap_variant")
        assert "chochmah" in seen
        v = json.loads(seen["chochmah"]["data"])["variants"][0]
        assert v["gap"] == "chokhmah"
        assert v["ratio"] >= 0.8 and v["arc"] >= 0.9

    def test_spelling_alone_is_not_enough(self, stack):
        """The material has to agree, or `device-dependent` matches its opposite."""
        db = stack["db"]
        self._indexed(db, "chochmah", ["kabbalah", "zohar", "hebrew"])
        self._gap(db, "chokhmah", {"polymer": 40, "cellulose": 30})
        stack["patrol"].run_errand("tag_gap_variants")
        assert _obs(db, "tag_gap_variant") == {}

    def test_a_gap_that_is_already_a_tag_is_not_a_finding(self, stack):
        db = stack["db"]
        self._indexed(db, "chochmah", ["kabbalah", "zohar"])
        self._indexed(db, "chokhmah", ["kabbalah", "zohar"])
        self._gap(db, "chokhmah", {"kabbalah": 40, "zohar": 30})
        stack["patrol"].run_errand("tag_gap_variants")
        assert _obs(db, "tag_gap_variant") == {}, "it is indexed already"

    def test_it_only_records(self, stack):
        db = stack["db"]
        self._indexed(db, "chochmah", ["kabbalah", "zohar", "hebrew"])
        self._gap(db, "chokhmah", {"kabbalah": 40, "zohar": 30, "hebrew": 20})
        tags = db.execute("SELECT COUNT(*) n FROM tags WHERE is_active=1").fetchone()["n"]
        links = db.execute("SELECT COUNT(*) n FROM memory_tags").fetchone()["n"]
        stack["patrol"].run_errand("tag_gap_variants")
        assert db.execute(
            "SELECT COUNT(*) n FROM tags WHERE is_active=1").fetchone()["n"] == tags
        assert db.execute("SELECT COUNT(*) n FROM memory_tags").fetchone()["n"] == links


class TestObserveDoesNotSumIdentifiers:
    """An identifier is not a quantity, and a measurement is not a tally.

    observe() summed ANY numeric field across passes. `tag_id` was added
    to itself on every re-deposit, so a seam recorded against tag 69 came
    to point at tag 207 — an unrelated tag sitting at the running total —
    and `held` became a sum of every reading ever taken, which is how
    `example` claimed 16,130 memories while holding 2. Forty of forty
    seams and 406 of 406 twins were wrong.
    """

    def test_an_identifier_is_not_added_up(self, stack):
        p, db = stack["patrol"], stack["db"]
        for _ in range(3):
            p.observe(errand="e", kind="k", subject="s", detail="d", weight=0.0,
                      data={"tag_id": 69, "held": 2}, lap=0)
        db.connection.commit()
        d = json.loads(_obs(db, "k")["s"]["data"])
        assert d["tag_id"] == 69, "an id must survive being re-observed"
        assert d["held"] == 2, "a measurement is the newer reading, not a sum"

    def test_a_declared_counter_still_accumulates(self, stack):
        """`mentions` is counted a batch at a time and has no other way
        to reach a corpus-wide total."""
        p, db = stack["patrol"], stack["db"]
        for _ in range(3):
            p.observe(errand="e", kind="k2", subject="s", detail="d", weight=0.0,
                      data={"mentions": 10}, lap=0)
        db.connection.commit()
        assert json.loads(_obs(db, "k2")["s"]["data"])["mentions"] == 30
