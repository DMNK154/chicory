"""Tests for diffing two accounts of one thing.

The rule is narrower than "find an asymmetry". A tag pair scoring 0.997
on centroid similarity is two DIFFERENT things whose vectors sit close;
3,724 of those produced nothing worth asking. Both sides have to be
describing the same material, which is what makes a divergence readable.

And both sides have to be the material a reader thinks they are. A side
used to be a SQL LIKE over `source_path`, which meant membership was a
property of a filename: twenty-nine works arrived on one day, two of them
the Pritzker Zohar, and "the primary text" silently became four files out
of eight while `companions` stayed on the record at 0.03% of a primary
text holding it at 5.79%.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer4.asymmetry import (
    CorpusAsymmetry,
    CorpusRegistry,
    document_digest,
)
from chicory.layer4.wander import Patrol, _asymmetry_definition
from tests.conftest import MockEmbeddingEngine

PAIR = {
    "name": "editions",
    "left_label": "edition A",
    "left_roles": ["edition-a"],
    "right_label": "edition B",
    "right_roles": ["edition-b"],
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
        corpus_documents=[],
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


def _mem(db, mid, src, content, created="2026-01-01T00:00:00.000"):
    db.execute(
        "INSERT OR IGNORE INTO memories (id, content, source_model, "
        "ingestion_tier, source_path, created_at) VALUES (?, ?, 'test', "
        "'critical', ?, ?)",
        (mid, content, src, created))
    db.connection.commit()


def _doc(stack, path, lines, role=None, title=None, created="2026-01-01T00:00:00.000"):
    """Store one document and, if it has a role, register it.

    The digest is computed the way the registry computes it, which is the
    point: a test that hard-coded one would not notice the registry and
    the scan disagreeing about what a document is.
    """
    for i, line in enumerate(lines):
        _mem(stack["db"], f"{path}:{i}", path, line, created)
    digest = CorpusRegistry(stack["config"], stack["db"]).digest_for(path)
    if role is not None:
        stack["config"].corpus_documents.append(
            {"role": role, "digest": digest, "title": title or path})
    return digest


def _sides(stack, a_lines, b_lines):
    _doc(stack, "a.pdf", a_lines, role="edition-a", title="edition A")
    _doc(stack, "b.pdf", b_lines, role="edition-b", title="edition B")


class TestComparison:
    def test_it_finds_a_word_one_side_uses_and_the_other_does_not(self, stack):
        _sides(stack,
               [f"Kudsha reigns here {i}" for i in range(6)],
               [f"Sovereign reigns here {i}" for i in range(6)])
        out = CorpusAsymmetry(stack["config"], stack["db"]).compare(PAIR)
        terms = {f["term"]: f for f in out["findings"]}
        assert "kudsha" in terms
        assert terms["kudsha"]["present_in"] == "edition A"
        assert terms["kudsha"]["absent_from"] == "edition B"
        assert "sovereign" in terms
        assert terms["sovereign"]["present_in"] == "edition B"

    def test_a_shared_word_is_not_an_asymmetry(self, stack):
        _sides(stack,
               [f"Malchut appears here {i}" for i in range(6)],
               [f"Malchut appears here too {i}" for i in range(6)])
        out = CorpusAsymmetry(stack["config"], stack["db"]).compare(PAIR)
        assert "malchut" not in {f["term"] for f in out["findings"]}

    def test_both_rates_are_recorded(self, stack):
        """Every good question came from a PAIR of numbers; a single rate
        with nothing to compare it against is what the noise looked like."""
        _sides(stack, [f"Kudsha here {i}" for i in range(6)],
               [f"Nothing here {i}" for i in range(6)])
        f = CorpusAsymmetry(stack["config"], stack["db"]).compare(PAIR)["findings"][0]
        assert "rate" in f and "other_rate" in f
        assert f["rate"] > f["other_rate"]

    def test_rates_are_used_not_raw_counts(self, stack):
        """The sides are usually very different sizes — 546 chunks of
        scholarship against 17,010 of primary text — so a count would only
        ever say which side is bigger."""
        _sides(stack,
               [f"Rashbi teaches {i}" for i in range(5)],
               [f"Shimon teaches {i}" for i in range(40)])
        out = CorpusAsymmetry(stack["config"], stack["db"]).compare(PAIR)
        terms = {f["term"] for f in out["findings"]}
        assert "rashbi" in terms, "the smaller side must still register"

    def test_a_side_too_small_to_rate_is_skipped(self, stack):
        _sides(stack, ["Kudsha here"], [f"Nothing here {i}" for i in range(6)])
        out = CorpusAsymmetry(stack["config"], stack["db"]).compare(PAIR)
        assert "skipped" in out

    def test_a_pair_that_resolves_to_nothing_does_not_stop_the_walk(self, stack):
        """A role no document carries is the author's mistake. It is
        reported rather than dropped: the whole failure being fixed here
        is a side quietly meaning less than it says."""
        _sides(stack, [f"Kudsha here {i}" for i in range(6)],
               [f"Nothing here {i}" for i in range(6)])
        cfg = stack["config"]
        cfg.asymmetry_pairs = [
            {"name": "broken", "left_label": "x", "left_roles": ["no-such-role"],
             "right_label": "y", "right_roles": ["edition-b"]},
            PAIR,
        ]
        out = CorpusAsymmetry(cfg, stack["db"]).run()
        by_name = {p["pair"]: p for p in out["pairs"]}
        assert set(by_name) == {"broken", "editions"}
        assert "skipped" in by_name["broken"]
        assert by_name["editions"]["found"] > 0


class TestDocumentIdentity:
    """Membership is a property of the work, not of its filename.

    `A River Flows from Eden` arrived a second time as
    `dokumen.pub_a-river-flows-from-eden-...`, 546 chunks byte-identical
    to the copy already shelved. Only the LIKE pattern's failure to match
    the new name kept the scholarship side from doubling.
    """

    LINES = [f"Kudsha reigns here {i}" for i in range(6)]

    def test_the_digest_does_not_depend_on_chunk_order(self):
        """A re-ingest that renumbers chunks must land on the same work."""
        chunks = ["aa11", "bb22", "cc33"]
        assert document_digest(chunks) == document_digest(reversed(chunks))
        assert document_digest(chunks) != document_digest(chunks[:2])

    def test_two_byte_identical_copies_are_one_document(self, stack):
        _doc(stack, "a.pdf", self.LINES, role="edition-a")
        _doc(stack, "a-copy.pdf", self.LINES, created="2026-02-01T00:00:00.000")
        docs = CorpusRegistry(stack["config"], stack["db"]).documents()
        holding = [d for d in docs.values() if "a.pdf" in d.copies]
        assert len(holding) == 1
        assert holding[0].copies == ("a.pdf", "a-copy.pdf")
        assert holding[0].duplicates == ("a-copy.pdf",)

    def test_the_copy_that_arrived_first_is_the_one_counted(self, stack):
        _doc(stack, "a.pdf", self.LINES, role="edition-a")
        _doc(stack, "a-copy.pdf", self.LINES, created="2026-02-01T00:00:00.000")
        _doc(stack, "b.pdf", [f"Sovereign here {i}" for i in range(6)],
             role="edition-b")
        out = CorpusAsymmetry(stack["config"], stack["db"]).compare(PAIR)
        assert out["left"].documents == ("a.pdf",)
        assert out["left"].duplicates == ("a-copy.pdf",)
        assert out["left_chunks"] == 6, "the duplicate must not be counted twice"

    def test_a_duplicate_does_not_dilute_a_heterogeneous_side(self, stack):
        """This is the one that bit. In a homogeneous side a duplicate is
        a rescale that cancels — 150 findings before and after, identical
        term sets. In a union it is dilution: the primary text went 15,508
        chunks to 24,476, every rate against it fell by 1.578, and 11 of
        the 60 recorded findings were manufactured by the drop.
        """
        cfg = stack["config"]
        _doc(stack, "sch.pdf", [f"Companions gather {i}" for i in range(6)],
             role="scholarship")
        _doc(stack, "p1.pdf", [f"Companions walking {i}" for i in range(2)]
             + [f"Nothing here {i}" for i in range(4)], role="primary")
        pair = {"name": "p", "left_label": "the scholarship",
                "left_roles": ["scholarship"], "right_label": "the primary text",
                "right_roles": ["primary"]}
        cfg.asymmetry_max_other = 0.2
        before = CorpusAsymmetry(cfg, stack["db"]).compare(pair)
        assert "companions" not in {f["term"] for f in before["findings"]}, (
            "companions is in a third of the primary text; it is no asymmetry"
        )
        # The same primary text arrives again under a new name.
        _doc(stack, "p1-copy.pdf", [f"Companions walking {i}" for i in range(2)]
             + [f"Nothing here {i}" for i in range(4)],
             created="2026-02-01T00:00:00.000")
        after = CorpusAsymmetry(cfg, stack["db"]).compare(pair)
        assert after["right_chunks"] == before["right_chunks"]
        assert "companions" not in {f["term"] for f in after["findings"]}

    def test_a_renamed_work_keeps_its_role(self, stack):
        """The digest is over content, so the filename may change freely.
        A LIKE over source_path is what lost Pritzker."""
        digest = _doc(stack, "old-name.pdf", self.LINES, role="edition-a")
        stack["db"].execute(
            "UPDATE memories SET source_path = 'new-name.pdf' "
            "WHERE source_path = 'old-name.pdf'")
        stack["db"].connection.commit()
        got = CorpusRegistry(stack["config"], stack["db"]).resolve(["edition-a"])
        assert got.documents == ("new-name.pdf",)
        assert got.digests == (digest,)

    def test_an_unregistered_work_joins_no_side(self, stack):
        _sides(stack, [f"Kudsha here {i}" for i in range(6)],
               [f"Sovereign here {i}" for i in range(6)])
        _doc(stack, "stranger.pdf", [f"Pritzker speaks {i}" for i in range(6)])
        out = CorpusAsymmetry(stack["config"], stack["db"]).compare(PAIR)
        assert "stranger.pdf" not in out["left"].documents
        assert "stranger.pdf" not in out["right"].documents
        assert out["right_chunks"] == 6

    def test_a_registry_entry_matching_nothing_is_reported(self, stack):
        """The one failure this scheme has that filenames did not: a work
        re-chunked or removed drops out silently unless it is named."""
        _sides(stack, [f"Kudsha here {i}" for i in range(6)],
               [f"Sovereign here {i}" for i in range(6)])
        stack["config"].corpus_documents.append(
            {"role": "edition-a", "digest": "0" * 16, "title": "a lost volume"})
        got = CorpusRegistry(stack["config"], stack["db"]).resolve(["edition-a"])
        assert got.missing == ("a lost volume",)


class TestBearingChunks:
    """A chunk that yields no term is denominator and nothing else.

    Two of the works ingested that day are Hebrew-script Zohar editions:
    1,571 chunks between them and not one capitalised Latin term. They can
    never supply a numerator while lowering every rate measured against
    their side, which is the shape of the commentary chunks that cost the
    bilingual lexicon 41 entries. No threshold is needed — the populations
    separate absolutely, 0 of 1,571 against 20,842 of 21,269.
    """

    def test_a_chunk_with_no_usable_term_is_not_in_the_population(self, stack):
        _doc(stack, "a.pdf", [f"Kudsha here {i}" for i in range(6)],
             role="edition-a")
        _doc(stack, "b.pdf", [f"Sovereign here {i}" for i in range(6)]
             + ["בראשית ברא אלהים"] * 20, role="edition-b")
        out = CorpusAsymmetry(stack["config"], stack["db"]).compare(PAIR)
        assert out["right_chunks_all"] == 26
        assert out["right_chunks"] == 6, "the Hebrew chunks bear no term"

    def test_dead_chunks_do_not_depress_the_other_side_rate(self, stack):
        """The gate `other_rate <= asymmetry_max_other` is absolute, so a
        one-sided dilution does not cancel: it pushes terms across the gate
        and invents findings."""
        cfg = stack["config"]
        cfg.asymmetry_max_other = 0.1
        _doc(stack, "a.pdf", [f"Kudsha rules {i}" for i in range(10)],
             role="edition-a")
        _doc(stack, "b.pdf", [f"Kudsha appears {i}" for i in range(2)]
             + [f"Nothing here {i}" for i in range(8)], role="edition-b")
        clean = CorpusAsymmetry(cfg, stack["db"]).compare(PAIR)
        assert "kudsha" not in {f["term"] for f in clean["findings"]}, (
            "kudsha is in 20% of side B; that is not absence"
        )
        # A Hebrew-script volume joins side B. It IS that edition, so it
        # carries the role; it simply bears no term the diff can read.
        _doc(stack, "heb.pdf", ["זהר על התורה"] * 40, role="edition-b")
        diluted = CorpusAsymmetry(cfg, stack["db"]).compare(PAIR)
        assert diluted["right_chunks_all"] == 50
        assert "kudsha" not in {f["term"] for f in diluted["findings"]}, (
            "40 term-free chunks must not turn a 20% presence into an absence"
        )
        assert diluted["right_chunks"] == clean["right_chunks"]


class TestErrand:
    def test_it_records_both_rates(self, stack):
        db = stack["db"]
        _sides(stack, [f"Kudsha here {i}" for i in range(6)],
               [f"Sovereign here {i}" for i in range(6)])
        stack["patrol"].run_errand("corpus_asymmetry")
        rows = {r["subject"]: json.loads(r["data"]) for r in db.execute(
            "SELECT subject, data FROM patrol_observations "
            "WHERE kind = 'corpus_asymmetry' AND retracted = 0")}
        assert "editions:kudsha" in rows
        d = rows["editions:kudsha"]
        assert d["rate"] > 0 and d["other_rate"] == 0.0
        assert d["present_in"] == "edition A"

    def test_every_observation_names_the_documents_it_compared(self, stack):
        """A rate means nothing except against a named population, and
        these populations move. Not one of the 120 standing observations
        said which files were in them."""
        db = stack["db"]
        _sides(stack, [f"Kudsha here {i}" for i in range(6)],
               [f"Sovereign here {i}" for i in range(6)])
        stack["patrol"].run_errand("corpus_asymmetry")
        d = json.loads(db.execute(
            "SELECT data FROM patrol_observations WHERE subject = "
            "'editions:kudsha' AND retracted = 0").fetchone()["data"])
        assert d["left_documents"] == ["a.pdf"]
        assert d["right_documents"] == ["b.pdf"]
        assert d["left_chunks"] == 6 and d["right_chunks"] == 6
        assert d["basis"]

    def test_a_duplicate_copy_is_named_as_excluded(self, stack):
        db = stack["db"]
        _sides(stack, [f"Kudsha here {i}" for i in range(6)],
               [f"Sovereign here {i}" for i in range(6)])
        _doc(stack, "a-copy.pdf", [f"Kudsha here {i}" for i in range(6)],
             created="2026-02-01T00:00:00.000")
        stack["patrol"].run_errand("corpus_asymmetry")
        d = json.loads(db.execute(
            "SELECT data FROM patrol_observations WHERE subject = "
            "'editions:kudsha' AND retracted = 0").fetchone()["data"])
        assert d["duplicates_excluded"] == ["a-copy.pdf"]

    def test_it_only_records(self, stack):
        db = stack["db"]
        _sides(stack, [f"Kudsha here {i}" for i in range(6)],
               [f"Sovereign here {i}" for i in range(6)])
        tags = db.execute("SELECT COUNT(*) n FROM tags").fetchone()["n"]
        links = db.execute("SELECT COUNT(*) n FROM memory_tags").fetchone()["n"]
        stack["patrol"].run_errand("corpus_asymmetry")
        assert db.execute("SELECT COUNT(*) n FROM tags").fetchone()["n"] == tags
        assert db.execute(
            "SELECT COUNT(*) n FROM memory_tags").fetchone()["n"] == links


class TestWithdrawal:
    """A finding is conditional on the populations it compared.

    When those change the finding is not stale in the way the lap audit
    catches — the walk would report it again, with different numbers. It
    has to be withdrawn on the ground that its definition moved.
    """

    def _first_lap(self, stack):
        _sides(stack, [f"Kudsha here {i}" for i in range(6)],
               [f"Sovereign here {i}" for i in range(6)])
        stack["patrol"].run_errand("corpus_asymmetry")

    def test_a_finding_survives_a_lap_that_changes_nothing(self, stack):
        self._first_lap(stack)
        before = stack["db"].execute(
            "SELECT data FROM patrol_observations WHERE subject = "
            "'editions:kudsha'").fetchone()["data"]
        CorpusAsymmetry(stack["config"], stack["db"]).run(
            patrol=stack["patrol"], lap=1)
        row = stack["db"].execute(
            "SELECT retracted, data FROM patrol_observations WHERE subject = "
            "'editions:kudsha'").fetchone()
        assert row["retracted"] == 0
        assert json.loads(row["data"])["basis"] == json.loads(before)["basis"]

    def test_a_document_joining_a_side_withdraws_the_old_findings(self, stack):
        """The Pritzker case: primary text arrives, every rate against the
        primary text changes, and every standing finding was measured
        against a population that no longer exists."""
        self._first_lap(stack)
        _doc(stack, "b2.pdf", [f"Sovereign again {i}" for i in range(6)],
             role="edition-b")
        out = CorpusAsymmetry(stack["config"], stack["db"]).run(
            patrol=stack["patrol"], lap=1)
        assert out["pairs"][0]["withdrawn"] > 0
        assert out["pairs"][0]["right_documents"] == ["b.pdf", "b2.pdf"]

    def test_a_finding_with_no_stamp_is_withdrawn(self, stack):
        """The 120 rows standing when this shipped carried no record of
        which documents were compared, so none of them can be trusted."""
        db = stack["db"]
        _sides(stack, [f"Kudsha here {i}" for i in range(6)],
               [f"Sovereign here {i}" for i in range(6)])
        db.execute(
            "INSERT INTO patrol_observations (errand, kind, subject, detail, "
            "weight, lap, data) VALUES ('corpus_asymmetry', "
            "'corpus_asymmetry', 'editions:antique', 'from before the "
            "registry', 0.01, 0, ?)",
            (json.dumps({"term": "antique", "rate": 0.4, "other_rate": 0.0}),))
        db.connection.commit()
        CorpusAsymmetry(stack["config"], db).run(patrol=stack["patrol"], lap=1)
        row = db.execute(
            "SELECT retracted, detail FROM patrol_observations "
            "WHERE subject = 'editions:antique'").fetchone()
        assert row["retracted"] == 1
        assert "predates" in row["detail"]

    def test_a_withdrawn_row_does_not_weld_two_memberships_together(self, stack):
        """`observe` unions list values across passes, and a row is reused
        across a retraction. A stamp that merged would describe a
        population neither pass ever looked at."""
        self._first_lap(stack)
        _doc(stack, "b2.pdf", [f"Sovereign again {i}" for i in range(6)],
             role="edition-b")
        CorpusAsymmetry(stack["config"], stack["db"]).run(
            patrol=stack["patrol"], lap=1)
        d = json.loads(stack["db"].execute(
            "SELECT data FROM patrol_observations WHERE subject = "
            "'editions:kudsha'").fetchone()["data"])
        assert d["right_documents"] == ["b.pdf", "b2.pdf"]
        assert d["left_documents"] == ["a.pdf"]


class TestLapGuard:
    """When the once-per-lap guard must stand aside.

    The guard exists so a whole-corpus diff runs once per lap rather than
    once per batch. But a lap here is the entire memories domain: 148,654
    stampings were outstanding when this was written, ~2,974 batches at
    the default fifty, and new material pushes the boundary out as it
    arrives. Holding a correction behind that is the same defect the
    registry exists to fix — an answer standing long after the thing it
    was an answer about has moved.
    """

    def _seed(self, stack):
        _sides(stack, [f"Kudsha here {i}" for i in range(6)],
               [f"Sovereign here {i}" for i in range(6)])

    def _run(self, stack):
        return stack["patrol"].run_errand("corpus_asymmetry")

    def test_a_second_pass_in_the_same_lap_is_skipped(self, stack):
        """The guard still does its job when nothing has changed."""
        self._seed(stack)
        self._run(stack)
        before = dict(stack["db"].execute(
            "SELECT updated_at, confirmations FROM patrol_observations "
            "WHERE subject = 'editions:kudsha'").fetchone())
        self._run(stack)
        after = dict(stack["db"].execute(
            "SELECT updated_at, confirmations FROM patrol_observations "
            "WHERE subject = 'editions:kudsha'").fetchone())
        assert after == before, (
            "a whole-corpus diff must not re-derive once per batch"
        )

    def test_a_finding_with_no_stamp_opens_the_guard(self, stack):
        """The 124 rows standing when the registry shipped. Each said a
        rate about a population nobody recorded, and the guard would have
        held them at retracted=0 until the lap turned."""
        db = stack["db"]
        self._seed(stack)
        db.execute(
            "INSERT INTO patrol_observations (errand, kind, subject, detail, "
            "weight, lap, data) VALUES ('corpus_asymmetry', "
            "'corpus_asymmetry', 'editions:antique', 'from before the "
            "registry', 0.01, 0, ?)",
            (json.dumps({"term": "antique", "rate": 0.4, "other_rate": 0.0}),))
        # Arm the guard exactly as a completed lap would.
        db.execute(
            "UPDATE wander_state SET cursor_hint = ? WHERE errand = ?",
            (f"lap0:{_asymmetry_definition(stack['config'])}",
             "corpus_asymmetry"))
        db.connection.commit()
        self._run(stack)
        assert db.execute(
            "SELECT retracted FROM patrol_observations WHERE subject = "
            "'editions:antique'").fetchone()["retracted"] == 1

    def test_editing_the_registry_opens_the_guard(self, stack):
        """A role or pair the author changed on purpose must not wait for
        a whole-corpus lap to take effect."""
        db = stack["db"]
        self._seed(stack)
        self._run(stack)
        armed = db.execute(
            "SELECT cursor_hint FROM wander_state WHERE errand = "
            "'corpus_asymmetry'").fetchone()["cursor_hint"]
        # A third volume is declared on the right side.
        _doc(stack, "b2.pdf", [f"Sovereign again {i}" for i in range(6)],
             role="edition-b")
        assert _asymmetry_definition(stack["config"]) not in armed
        self._run(stack)
        d = json.loads(db.execute(
            "SELECT data FROM patrol_observations WHERE subject = "
            "'editions:kudsha'").fetchone()["data"])
        assert d["right_documents"] == ["b.pdf", "b2.pdf"]

    def test_the_marker_is_stable_when_nothing_is_edited(self, stack):
        cfg = stack["config"]
        assert _asymmetry_definition(cfg) == _asymmetry_definition(cfg)


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
