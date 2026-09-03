"""Tests for meta-shelf charters.

A charter asserts that a set of concepts is one subject.  The roster then
derives a shelf from it on every rebuild — so the thing worth testing is
not that a shelf appears once, but that it appears the SAME on the second
and third rebuild, and that everything else keeps deriving around it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer1.tag_manager import TagManager
from chicory.layer3.centroid_subgraph import CentroidSubgraph, _array_to_blob
from chicory.layer4.librarian import SubjectRoster
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
        subject_min_edge_strength=0.05,
        subject_min_cluster_size=2,
        subject_hub_tag_max_fraction=999.0,
        meta_shelf_min_size=2,
    )
    db = DatabaseEngine(config)
    db.connect()
    apply_schema(db)
    emb = MockEmbeddingEngine(dimension=DIM)
    emb.set_db(db)
    yield {
        "config": config,
        "db": db,
        "tags": TagManager(db),
        "roster": SubjectRoster(config, db, CentroidSubgraph(config, db, emb)),
    }
    db.close()


def _vec(i: int, tilt: float = 0.0) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[i % DIM] = 1.0
    if tilt:
        v[(i + 1) % DIM] = tilt
    return v / np.linalg.norm(v)


def _tag(db, name: str, vec: np.ndarray, memory_count: int = 5) -> int:
    db.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (name,))
    tid = db.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()["id"]
    db.execute(
        "INSERT OR REPLACE INTO tag_centroids (tag_id, centroid, memory_count) "
        "VALUES (?, ?, ?)", (tid, _array_to_blob(vec), memory_count),
    )
    return tid


def _edge(db, a: int, b: int, strength: float = 0.5, count: int = 5) -> None:
    lo, hi = min(a, b), max(a, b)
    db.execute(
        "INSERT OR REPLACE INTO centroid_edges "
        "(tag_a_id, tag_b_id, edge_strength, co_retrieval_count) VALUES (?, ?, ?, ?)",
        (lo, hi, strength, count),
    )


def _charter(db, slug: str, tag_ids: list[int], display: str | None = None,
             status: str = "materialized") -> int:
    db.execute(
        "INSERT INTO meta_shelves (slug, display_name, status) VALUES (?, ?, ?)",
        (slug, display, status),
    )
    sid = db.execute(
        "SELECT id FROM meta_shelves WHERE slug = ?", (slug,),
    ).fetchone()["id"]
    for i, tid in enumerate(tag_ids):
        db.execute(
            "INSERT INTO meta_shelf_members (shelf_id, concept, tag_id) "
            "VALUES (?, ?, ?)", (sid, f"{slug}-c{i}", tid),
        )
    db.connection.commit()
    return sid


def _scatter(db, names: list[str], base: int, link: bool = True) -> list[int]:
    """A clique of tags that would otherwise form one derived component."""
    ids = [_tag(db, n, _vec(base + i, tilt=0.3)) for i, n in enumerate(names)]
    if link:
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                _edge(db, ids[i], ids[j])
    db.connection.commit()
    return ids


def _paths(db) -> set[str]:
    return {r["path"] for r in db.execute(
        "SELECT path FROM subjects WHERE status = 'active'")}


def _members(db, name: str) -> set[int]:
    row = db.execute(
        "SELECT member_tag_ids FROM subjects WHERE name = ? AND status = 'active'",
        (name,),
    ).fetchone()
    return set(json.loads(row["member_tag_ids"])) if row else set()


class TestCharterDerivation:
    def test_a_charter_becomes_a_root_shelf_under_its_own_name(self, stack):
        db = stack["db"]
        sef = _scatter(db, ["malchut", "binah", "gvurah"], 1)
        _scatter(db, ["rust", "python", "golang"], 10)
        _charter(db, "sefirot", sef, display="The Sefirot")

        stack["roster"].rebuild()

        row = db.execute(
            "SELECT name, path, depth, parent_id, display_name, memory_count "
            "FROM subjects WHERE name = 'sefirot' AND status = 'active'"
        ).fetchone()
        assert row is not None, "chartered shelf did not derive"
        assert row["path"] == "sefirot"           # the address is the slug
        assert row["depth"] == 0                  # its own root, not a child
        assert row["parent_id"] is None
        assert row["display_name"] == "The Sefirot"
        assert row["memory_count"] is not None    # still DERIVED, not authored

    def test_chartered_tags_are_withheld_from_the_derived_walk(self, stack):
        """The whole point: a charter's tags cannot be absorbed by the giant.

        Every sefirot tag is linked to `mysticism` at full strength, which
        on the live library is exactly how malchut ended up on the Vedanta
        branch.  Reserving happens before component-finding, so the edge
        still exists and the roster simply declines to walk through it.
        """
        db = stack["db"]
        giant = _scatter(db, ["mysticism", "vedanta", "hinduism"], 20)
        sef = _scatter(db, ["malchut", "binah", "gvurah"], 1)
        for t in sef:                    # the absorbing edges, at full strength
            _edge(db, t, giant[0], strength=1.0, count=99)
        db.connection.commit()
        edges_before = db.execute(
            "SELECT COUNT(*) n FROM centroid_edges").fetchone()["n"]

        _charter(db, "sefirot", sef)
        stack["roster"].rebuild()

        assert _members(db, "sefirot") == set(sef)
        for r in db.execute("SELECT name, member_tag_ids FROM subjects "
                            "WHERE status = 'active' AND name != 'sefirot'"):
            got = set(json.loads(r["member_tag_ids"]))
            assert not (got & set(sef)), f"{r['name']} absorbed chartered tags"
        # Reserving withholds; it must never delete.  centroid_edges is read
        # by retrieval, the canopy and centroid reweighting too.
        assert db.execute(
            "SELECT COUNT(*) n FROM centroid_edges").fetchone()["n"] == edges_before


class TestIdempotency:
    def test_three_rebuilds_move_nothing(self, stack):
        """The check that would have caught both previous churn bugs."""
        db = stack["db"]
        sef = _scatter(db, ["malchut", "binah", "gvurah"], 1)
        _scatter(db, ["rust", "python", "golang"], 10)
        _scatter(db, ["mysticism", "vedanta", "hinduism"], 20)
        _charter(db, "sefirot", sef, display="The Sefirot")

        stack["roster"].rebuild()
        first = _paths(db)

        second = stack["roster"].rebuild()
        after_second = _paths(db)
        third = stack["roster"].rebuild()
        after_third = _paths(db)

        assert after_second == first, "run 2 moved a path"
        assert after_third == after_second, "run 3 moved a path"
        assert second["subjects_retired"] == 0
        assert third["subjects_retired"] == 0
        assert "sefirot" in after_third

    def test_a_rename_survives_rebuilds(self, stack):
        """display_name is authored; the DO UPDATE SET must keep omitting it."""
        db = stack["db"]
        sef = _scatter(db, ["malchut", "binah", "gvurah"], 1)
        _charter(db, "sefirot", sef, display="The Sefirot")
        stack["roster"].rebuild()

        stack["roster"].rename("sefirot", "The Ten Emanations")
        stack["roster"].rebuild()

        assert db.execute(
            "SELECT display_name FROM subjects WHERE name = 'sefirot'"
        ).fetchone()["display_name"] == "The Ten Emanations"


class TestGuards:
    def test_a_thin_charter_warns_rather_than_vanishing(self, stack):
        db = stack["db"]
        sef = _scatter(db, ["malchut", "binah", "gvurah"], 1)
        _charter(db, "sefirot", sef)
        # Two of three members lose their centroid, so they stop being
        # topical and the shelf can no longer derive.
        db.execute("DELETE FROM tag_centroids WHERE tag_id IN (?, ?)",
                   (sef[0], sef[1]))
        _scatter(db, ["rust", "python", "golang"], 10)
        db.connection.commit()

        result = stack["roster"].rebuild()

        assert "sefirot" not in _paths(db)
        assert any("sefirot" in w for w in result.get("charter_warnings", [])), \
            "a shelf that stopped deriving vanished without saying so"

    def test_overlapping_charters_resolve_by_age_not_by_exploding(self, stack):
        """An overlap must not abort the maintain it is discovered in."""
        db = stack["db"]
        shared = _scatter(db, ["malchut", "binah", "gvurah"], 1)
        other = _scatter(db, ["turya", "taijasa", "vishva"], 30)
        _charter(db, "sefirot", shared)
        _charter(db, "advaita", other + [shared[0]])    # claims malchut too

        result = stack["roster"].rebuild()             # must not raise

        first, second = _members(db, "sefirot"), _members(db, "advaita")
        assert shared[0] in first, "the earlier charter should keep the tag"
        assert shared[0] not in second
        assert not (first & second), "charters must partition"
        assert any("advaita" in w for w in result.get("charter_warnings", []))

    def test_only_materialized_charters_derive(self, stack):
        db = stack["db"]
        sef = _scatter(db, ["malchut", "binah", "gvurah"], 1)
        _scatter(db, ["rust", "python", "golang"], 10)
        _charter(db, "sefirot", sef, status="confirmed")   # not yet materialized

        stack["roster"].rebuild()

        assert "sefirot" not in _paths(db)
        # and its tags stayed available to the ordinary derivation
        seen: set[int] = set()
        for r in db.execute(
            "SELECT member_tag_ids FROM subjects WHERE status = 'active'"
        ):
            seen |= set(json.loads(r["member_tag_ids"]))
        assert set(sef) & seen


# ── The proposer ────────────────────────────────────────────────────────


def _proposer(cfg=None):
    from chicory.layer4.meta_shelves import MetaShelfProposer
    cfg = cfg or ChicoryConfig(
        db_path=Path(":memory:"), anthropic_api_key="test",
        llm_model="test-model", embedding_model="mock",
        embedding_dimension=DIM)
    p = MetaShelfProposer.__new__(MetaShelfProposer)
    p._config, p._db = cfg, None
    return p


class TestOrthographicBridge:
    def _nodes(self, names, around=None):
        around = around or {"kabbalah": 10, "zohar": 8, "hebrew": 5}
        return {n: {"around": dict(around), "co": {}, "documents": set(),
                    "mentions": 1} for n in names}

    def test_it_finds_a_transliteration_variant(self):
        nodes = self._nodes(["gevurah", "gvurah", "malchut", "binah"])
        assert ("gevurah", "gvurah") in _proposer()._ortho_bridges(nodes, set())

    def test_a_variant_that_is_already_a_weak_edge_is_still_bridged(self):
        """The bug that silently lost the motivating pair.

        `gevurah` and `gvurah` share zero memories but score sc 0.326,
        which is rank 40 of 48 in gvurah's neighbour list — so mutual-kNN
        discards the edge.  Skipping the pair here as "already connected"
        means it is dropped at the next step instead, with no error.
        """
        nodes = self._nodes(["gevurah", "gvurah"])
        already = {("gevurah", "gvurah")}
        assert ("gevurah", "gvurah") in _proposer()._ortho_bridges(nodes, already)

    def test_lexicographic_order_does_not_decide_membership(self):
        """'gevurah' < 'gvurah' is True; a `b <= a` guard eats exactly this."""
        assert "gevurah" < "gvurah"
        for pair in (["gevurah", "gvurah"], ["gvurah", "gevurah"]):
            got = _proposer()._ortho_bridges(self._nodes(pair), set())
            assert ("gevurah", "gvurah") in got, f"lost when ordered {pair}"

    def test_unrelated_material_is_not_bridged_on_spelling_alone(self):
        nodes = self._nodes(["gevurah"])
        nodes.update(self._nodes(["gevurat"], around={"polymer": 9, "cellulose": 7}))
        assert not _proposer()._ortho_bridges(nodes, set())
