"""Tests for diffing two accounts of one thing.

The rule is narrower than "find an asymmetry". A tag pair scoring 0.997
on centroid similarity is two DIFFERENT things whose vectors sit close;
3,724 of those produced nothing worth asking. Both sides have to be
describing the same material, which is what makes a divergence readable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer4.asymmetry import CorpusAsymmetry
from chicory.layer4.wander import Patrol
from tests.conftest import MockEmbeddingEngine

PAIR = {
    "name": "editions",
    "left_label": "edition A",
    "left": "source_path = 'a.pdf'",
    "right_label": "edition B",
    "right": "source_path = 'b.pdf'",
}


@pytest.fixture
def stack():
    config = ChicoryConfig(
        db_path=Path(":memory:"), anthropic_api_key="test",
        llm_model="test-model", embedding_model="mock",
        embedding_dimension=32, wander_batch_size=10,
        wander_interval_minutes=0.0, wander_errands=["corpus_asymmetry"],
        asymmetry_pairs=[PAIR], asymmetry_min_rate=0.2,
        asymmetry_max_other=0.05, asymmetry_min_population=4,
        tag_content_baseline_sample=200, tag_content_sample=100,
        tag_content_min_memories=10,
    )
    db = DatabaseEngine(config)
    db.connect()
    apply_schema(db)
    emb = MockEmbeddingEngine(dimension=32)
    emb.set_db(db)
    yield {"config": config, "db": db, "patrol": Patrol(config, db, emb)}
    db.close()


def _mem(db, mid, src, content):
    db.execute(
        "INSERT OR IGNORE INTO memories (id, content, source_model, "
        "ingestion_tier, source_path) VALUES (?, ?, 'test', 'critical', ?)",
        (mid, content, src))
    db.connection.commit()


def _sides(db, a_lines, b_lines):
    for i, line in enumerate(a_lines):
        _mem(db, f"a{i}", "a.pdf", line)
    for i, line in enumerate(b_lines):
        _mem(db, f"b{i}", "b.pdf", line)


class TestComparison:
    def test_it_finds_a_word_one_side_uses_and_the_other_does_not(self, stack):
        db = stack["db"]
        _sides(db,
               [f"Kudsha reigns here {i}" for i in range(6)],
               [f"Sovereign reigns here {i}" for i in range(6)])
        out = CorpusAsymmetry(stack["config"], db).compare(PAIR)
        terms = {f["term"]: f for f in out["findings"]}
        assert "kudsha" in terms
        assert terms["kudsha"]["present_in"] == "edition A"
        assert terms["kudsha"]["absent_from"] == "edition B"
        assert "sovereign" in terms
        assert terms["sovereign"]["present_in"] == "edition B"

    def test_a_shared_word_is_not_an_asymmetry(self, stack):
        db = stack["db"]
        _sides(db,
               [f"Malchut appears here {i}" for i in range(6)],
               [f"Malchut appears here too {i}" for i in range(6)])
        out = CorpusAsymmetry(stack["config"], db).compare(PAIR)
        assert "malchut" not in {f["term"] for f in out["findings"]}

    def test_both_rates_are_recorded(self, stack):
        """Every good question came from a PAIR of numbers; a single rate
        with nothing to compare it against is what the noise looked like."""
        db = stack["db"]
        _sides(db, [f"Kudsha here {i}" for i in range(6)],
               [f"Nothing here {i}" for i in range(6)])
        f = CorpusAsymmetry(stack["config"], db).compare(PAIR)["findings"][0]
        assert "rate" in f and "other_rate" in f
        assert f["rate"] > f["other_rate"]

    def test_rates_are_used_not_raw_counts(self, stack):
        """The sides are usually very different sizes — 546 chunks of
        scholarship against 15,508 of primary text — so a count would only
        ever say which side is bigger."""
        db = stack["db"]
        _sides(db,
               [f"Rashbi teaches {i}" for i in range(5)],
               [f"Shimon teaches {i}" for i in range(40)])
        out = CorpusAsymmetry(stack["config"], db).compare(PAIR)
        terms = {f["term"] for f in out["findings"]}
        assert "rashbi" in terms, "the smaller side must still register"

    def test_a_side_too_small_to_rate_is_skipped(self, stack):
        db = stack["db"]
        _sides(db, ["Kudsha here"], [f"Nothing here {i}" for i in range(6)])
        out = CorpusAsymmetry(stack["config"], db).compare(PAIR)
        assert "skipped" in out

    def test_a_broken_predicate_does_not_stop_the_walk(self, stack):
        db = stack["db"]
        _sides(db, [f"Kudsha here {i}" for i in range(6)],
               [f"Nothing here {i}" for i in range(6)])
        cfg = stack["config"]
        cfg.asymmetry_pairs = [
            {"name": "broken", "left_label": "x", "left": "no_such_column = 1",
             "right_label": "y", "right": "source_path = 'b.pdf'"},
            PAIR,
        ]
        out = CorpusAsymmetry(cfg, db).run()
        assert [p["pair"] for p in out["pairs"]] == ["editions"], (
            "a bad predicate is the author's mistake, not a reason to stop"
        )


class TestErrand:
    def test_it_records_both_rates(self, stack):
        db = stack["db"]
        _sides(db, [f"Kudsha here {i}" for i in range(6)],
               [f"Sovereign here {i}" for i in range(6)])
        stack["patrol"].run_errand("corpus_asymmetry")
        rows = {r["subject"]: json.loads(r["data"]) for r in db.execute(
            "SELECT subject, data FROM patrol_observations "
            "WHERE kind = 'corpus_asymmetry' AND retracted = 0")}
        assert "editions:kudsha" in rows
        d = rows["editions:kudsha"]
        assert d["rate"] > 0 and d["other_rate"] == 0.0
        assert d["present_in"] == "edition A"

    def test_it_only_records(self, stack):
        db = stack["db"]
        _sides(db, [f"Kudsha here {i}" for i in range(6)],
               [f"Sovereign here {i}" for i in range(6)])
        tags = db.execute("SELECT COUNT(*) n FROM tags").fetchone()["n"]
        links = db.execute("SELECT COUNT(*) n FROM memory_tags").fetchone()["n"]
        stack["patrol"].run_errand("corpus_asymmetry")
        assert db.execute("SELECT COUNT(*) n FROM tags").fetchone()["n"] == tags
        assert db.execute(
            "SELECT COUNT(*) n FROM memory_tags").fetchone()["n"] == links


class TestTagContent:
    """A tag's material against the label it was filed under.

    Comparing content to the tag NAME says nothing — `mysticism` never
    appears in the material it labels. What separates a subject from a
    label is whether its memories share vocabulary the library at large
    does not: `plato` yields socrates, glaucon, republic; `whirly-art`,
    on 1,552 memories, yields nothing at all.
    """

    def _library(self, db, n=300):
        """Big enough that the tag under test is a small share of it.

        The baseline is a sample of the whole library, so a tag's own
        memories are in it. On the real corpus a 400-memory tag is 0.3% of
        124,363 and contributes nothing; in a fixture of ninety it would
        dominate the baseline it is being compared against.
        """
        for i in range(n):
            _mem(db, f"lib{i}", "misc.pdf",
                 f"Ordinary material about Nothing Particular here {i}")

    def test_a_real_subject_names_what_it_is_about(self, stack):
        db = stack["db"]
        self._library(db)
        tid = db.execute("INSERT INTO tags (name) VALUES ('plato')").lastrowid
        for i in range(30):
            _mem(db, f"p{i}", "rep.pdf", f"Socrates asked Glaucon in Republic {i}")
            db.execute("INSERT INTO memory_tags (memory_id, tag_id, assigned_by) "
                       "VALUES (?, ?, 'test')", (f"p{i}", tid))
        db.connection.commit()
        from chicory.layer4.asymmetry import TagContentAsymmetry
        got = TagContentAsymmetry(stack["config"], db).examine(tid, "plato")
        terms = {d["term"] for d in got["distinctive"]}
        assert "socrates" in terms and "glaucon" in terms

    def test_a_label_with_no_subject_is_reported(self, stack):
        db = stack["db"]
        self._library(db)
        tid = db.execute("INSERT INTO tags (name) VALUES ('whirly-art')").lastrowid
        for i in range(30):
            _mem(db, f"w{i}", "misc.pdf",
                 f"Ordinary material about Nothing Particular here {i+900}")
            db.execute("INSERT INTO memory_tags (memory_id, tag_id, assigned_by) "
                       "VALUES (?, ?, 'test')", (f"w{i}", tid))
        db.connection.commit()
        from chicory.layer4.asymmetry import TagContentAsymmetry
        got = TagContentAsymmetry(stack["config"], db).examine(tid, "whirly-art")
        assert got["distinctive"] == [], (
            "material indistinguishable from the library carries no subject"
        )

    def test_an_abstraction_is_not_penalised_for_not_naming_itself(self, stack):
        """`mysticism` is a real subject nobody writes the word for."""
        db = stack["db"]
        self._library(db)
        tid = db.execute("INSERT INTO tags (name) VALUES ('mysticism')").lastrowid
        for i in range(30):
            _mem(db, f"m{i}", "z.pdf", f"Rabbi Shimon and Hashem and Malchut {i}")
            db.execute("INSERT INTO memory_tags (memory_id, tag_id, assigned_by) "
                       "VALUES (?, ?, 'test')", (f"m{i}", tid))
        db.connection.commit()
        from chicory.layer4.asymmetry import TagContentAsymmetry
        got = TagContentAsymmetry(stack["config"], db).examine(tid, "mysticism")
        assert got["distinctive"], "it has a subject even without its own name"
        assert got["names_itself"] is False

    def test_too_few_memories_is_no_finding(self, stack):
        db = stack["db"]
        self._library(db)
        tid = db.execute("INSERT INTO tags (name) VALUES ('tiny')").lastrowid
        _mem(db, "t1", "misc.pdf", "One Solitary Chunk")
        db.execute("INSERT INTO memory_tags (memory_id, tag_id, assigned_by) "
                   "VALUES ('t1', ?, 'test')", (tid,))
        db.connection.commit()
        from chicory.layer4.asymmetry import TagContentAsymmetry
        assert TagContentAsymmetry(stack["config"], db).examine(tid, "tiny") is None
