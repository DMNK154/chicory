"""Tests for reading a facing-text corpus in both languages.

The Zohar volumes arrive with U+200B as the word separator, ordinary
spaces INSIDE words, and nikkud throughout. Nothing works until that is
handled: a scan that splits on whitespace finds גבורה three times in a
corpus containing it 213 times.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.db.schema import apply_schema
from chicory.layer4.bilingual import (
    BilingualLexicon,
    base_forms,
    dice,
    english_terms,
    hebrew_words,
    skeleton_counts,
)
from chicory.layer4.wander import Patrol
from tests.conftest import MockEmbeddingEngine

ZWSP = "​"
# "and the wise shall shine", as the extraction actually delivers it:
# points, spaces inside words, zero-width space between them.
BROKEN = "ְו ַהַּמ ְׂש ִּכ יִלים"


@pytest.fixture
def stack():
    config = ChicoryConfig(
        db_path=Path(":memory:"), anthropic_api_key="test",
        llm_model="test-model", embedding_model="mock",
        embedding_dimension=32, wander_batch_size=10,
        wander_interval_minutes=0.0, wander_errands=["hebrew_lexicon"],
        bilingual_min_chunks=2, bilingual_min_dice=0.45,
        bilingual_ubiquity_max=0.90,
    )
    db = DatabaseEngine(config)
    db.connect()
    apply_schema(db)
    emb = MockEmbeddingEngine(dimension=32)
    emb.set_db(db)
    yield {"config": config, "db": db, "patrol": Patrol(config, db, emb)}
    db.close()


def _mem(db, mid, content):
    db.execute(
        "INSERT OR IGNORE INTO memories (id, content, source_model, "
        "ingestion_tier, source_path) VALUES (?, ?, 'test', 'critical', 'z.pdf')",
        (mid, content))
    db.connection.commit()


class TestNormalisation:
    def test_a_word_broken_by_spaces_and_points_comes_back(self):
        assert hebrew_words(BROKEN) == ["והמשכילים"]

    def test_the_zero_width_space_is_the_word_boundary(self):
        assert hebrew_words("גבורה" + ZWSP + "חסד") == ["גבורה", "חסד"]

    def test_a_plain_space_is_not_a_word_boundary(self):
        """The failure mode: whitespace splitting shreds every word."""
        assert hebrew_words("גב ורה") == ["גבורה"]

    def test_nikkud_is_dropped(self):
        assert hebrew_words("גְבּורָה") \
            == ["גבורה"]

    def test_non_hebrew_is_ignored(self):
        assert hebrew_words("Rabbi Shimon said (Daniel 12:3)") == []


class TestPrefixes:
    def test_attached_prefixes_are_offered_as_alternatives(self):
        assert base_forms("בחכמה") == ["בחכמה", "חכמה"]

    def test_the_unstripped_form_comes_first(self):
        """Stripping is a guess; the caller that wants one answer takes
        the word as written."""
        assert base_forms("בריך")[0] == "בריך"

    def test_a_non_prefix_letter_stops_the_strip(self):
        assert base_forms("תפארת") == ["תפארת"]

    def test_it_will_not_strip_below_the_minimum(self):
        assert base_forms("והב") == ["והב"]

    def test_inflections_fold_onto_one_term(self):
        counts = skeleton_counts("בחכמה" + ZWSP + "מחכמה" + ZWSP + "חכמה")
        assert counts["חכמה"] == 3


class TestEnglishSide:
    def test_it_takes_capitalised_words(self):
        assert english_terms("Moshe went to Egypt with Aharon") == {
            "Moshe", "Egypt", "Aharon"}

    def test_it_drops_the_edition_furniture(self):
        assert english_terms("Rabbi Shimon said, Behold the Holy Zohar") == {
            "Shimon"}


class TestDice:
    def test_perfect_overlap_is_one(self):
        assert dice(10, 10, 10) == 1.0

    def test_no_overlap_is_zero(self):
        assert dice(0, 10, 10) == 0.0

    def test_a_ubiquitous_partner_is_penalised(self):
        """Sharing every chunk with a word that is on every page is weak
        evidence; the denominator has to say so."""
        assert dice(10, 10, 5000) < 0.1


class TestLexicon:
    def _corpus(self, db):
        # משה appears with Moshe; פרעה with Pharaoh; ספר is on every page
        # and must win nothing. The filler matters: a term present in
        # EVERY chunk is cut for ubiquity, which is the rule working.
        for i in range(6):
            _mem(db, f"a{i}", f"ספר{ZWSP}אמר{ZWSP}משה spoke. Moshe answered.")
        for i in range(6):
            _mem(db, f"b{i}", f"ספר{ZWSP}אמר{ZWSP}פרעה refused. Pharaoh hardened.")
        for i in range(8):
            _mem(db, f"f{i}", f"ספר{ZWSP}אמר{ZWSP}דבר plainly. Yehuda listened.")

    def test_it_learns_the_pairing_from_the_facing_text(self, stack):
        self._corpus(stack["db"])
        lex = BilingualLexicon(stack["config"], stack["db"])
        entries = {e["hebrew"]: e["english"] for e in lex.align(lex.scan())}
        assert entries.get("משה") == "Moshe"
        assert entries.get("פרעה") == "Pharaoh"

    def test_a_word_on_every_page_wins_nothing(self, stack):
        self._corpus(stack["db"])
        lex = BilingualLexicon(stack["config"], stack["db"])
        got = {e["hebrew"] for e in lex.align(lex.scan())}
        assert "ספר" not in got, (
            "a term in every chunk distinguishes nothing, however high "
            "its raw co-occurrence"
        )

    def test_prefix_debris_loses_to_the_longer_form(self, stack):
        db = stack["db"]
        for i in range(6):
            _mem(db, f"c{i}", f"אמר{ZWSP}מטטרון spoke. Metatron answered.")
        for i in range(8):
            _mem(db, f"d{i}", f"אמר{ZWSP}דבר plainly. Yehuda listened.")
        lex = BilingualLexicon(stack["config"], stack["db"])
        got = {e["hebrew"] for e in lex.align(lex.scan())
               if e["english"] == "Metatron"}
        assert "מטטרון" in got
        assert "טטרון" not in got, "the stripped guess must lose"

    def test_the_errand_records_the_lexicon(self, stack):
        self._corpus(stack["db"])
        stack["patrol"].run_errand("hebrew_lexicon")
        rows = {r["subject"]: r for r in stack["db"].execute(
            "SELECT subject, data FROM patrol_observations "
            "WHERE kind = 'bilingual_term' AND retracted = 0")}
        assert "משה" in rows
        assert json.loads(rows["משה"]["data"])["english"] == "Moshe"

    def test_it_only_records(self, stack):
        self._corpus(stack["db"])
        db = stack["db"]
        tags = db.execute("SELECT COUNT(*) n FROM tags").fetchone()["n"]
        links = db.execute("SELECT COUNT(*) n FROM memory_tags").fetchone()["n"]
        stack["patrol"].run_errand("hebrew_lexicon")
        assert db.execute("SELECT COUNT(*) n FROM tags").fetchone()["n"] == tags
        assert db.execute(
            "SELECT COUNT(*) n FROM memory_tags").fetchone()["n"] == links


class TestBoilerplate:
    """A page that always travels together aligns perfectly and means nothing.

    Measured: a liturgical preface occupies exactly 12 chunks beside the
    English word `Singer` in the same 12, so all 21 of its Hebrew words
    score dice 1.000 — above בינה/Binah at 0.903. Co-occurrence cannot
    tell a translation from a repeated block; arity can.
    """

    def test_a_repeated_block_is_dropped_entirely(self, stack):
        db = stack["db"]
        block = ZWSP.join(["אמר", "חטאותינו", "ותקשיב", "לתורתך",
                           "שאנחנו", "נשמתנו"])
        for i in range(5):
            _mem(db, f"blk{i}", f"{block} Printed by Singer and Sons.")
        for i in range(9):
            _mem(db, f"oth{i}", f"אמר{ZWSP}דבר plainly. Yehuda listened.")
        lex = BilingualLexicon(stack["config"], stack["db"])
        got = [e for e in lex.align(lex.scan()) if e["english"] == "Singer"]
        assert got == [], "a block gives no reason to prefer any member"

    def test_a_genuine_pair_survives_the_guard(self, stack):
        db = stack["db"]
        for i in range(5):
            _mem(db, f"g{i}", f"אמר{ZWSP}מטטרון spoke. Metatron answered.")
        for i in range(9):
            _mem(db, f"h{i}", f"אמר{ZWSP}דבר plainly. Yehuda listened.")
        lex = BilingualLexicon(stack["config"], stack["db"])
        got = {e["hebrew"] for e in lex.align(lex.scan())
               if e["english"] == "Metatron"}
        assert "מטטרון" in got


class TestAramaicPrefix:
    """The Zohar is Aramaic, where ד (of/that) is the commonest prefix.

    Omitting it left construct forms standing as words, and a construct
    form never appears without what it governs — so דזעיר aligned to
    `Anpin` and והוד to `Netzach`, each mapping to its partner rather
    than itself.
    """

    def test_the_genitive_is_stripped(self):
        assert base_forms("דעתיקא") == ["דעתיקא", "עתיקא"]

    def test_construct_forms_fold_onto_the_bare_word(self):
        counts = skeleton_counts("דזעיר" + ZWSP + "זעיר" + ZWSP + "וזעיר")
        assert counts["זעיר"] == 3

    def test_a_hebrew_prefix_still_works(self):
        assert base_forms("בחכמה") == ["בחכמה", "חכמה"]


class TestSuffixAmbiguity:
    """One skeleton being a suffix of another has two opposite causes.

    `ריך` is debris from stripping `בריך`, so the LONGER form is real.
    `דזעיר` is a construct form whose bare word `זעיר` is real, so the
    SHORTER one is. The shapes are identical; the counts are not.
    """

    def test_the_bare_word_beats_its_construct_form(self, stack):
        db = stack["db"]
        # זעיר appears bare and prefixed; both emit the bare skeleton, so
        # it outnumbers דזעיר.
        for i in range(4):
            _mem(db, f"p{i}", f"אמר{ZWSP}דזעיר spoke. Anpin answered.")
        for i in range(4):
            _mem(db, f"q{i}", f"אמר{ZWSP}זעיר spoke. Anpin answered.")
        for i in range(9):
            _mem(db, f"r{i}", f"אמר{ZWSP}דבר plainly. Yehuda listened.")
        lex = BilingualLexicon(stack["config"], stack["db"])
        got = {e["hebrew"] for e in lex.align(lex.scan())
               if e["english"] == "Anpin"}
        assert "זעיר" in got
        assert "דזעיר" not in got

    def test_stripping_debris_still_loses_to_its_parent(self, stack):
        db = stack["db"]
        # בריך never appears bare, so ריך only ever ties with it.
        for i in range(5):
            _mem(db, f"s{i}", f"אמר{ZWSP}בריך is said. Berikh follows.")
        for i in range(9):
            _mem(db, f"t{i}", f"אמר{ZWSP}דבר plainly. Yehuda listened.")
        lex = BilingualLexicon(stack["config"], stack["db"])
        got = {e["hebrew"] for e in lex.align(lex.scan())
               if e["english"] == "Berikh"}
        assert "בריך" in got
        assert "ריך" not in got


class TestAttestation:
    """A prefix letter and a root letter look identical.

    ב opens both בחכמה (in-wisdom) and בינה (Binah); מ opens both מחכמה
    and מלכות. Stripping by shape turns Binah into ינה and Malchut into
    לכות. What separates them is that a real prefix is OPTIONAL, so the
    bare form appears somewhere on its own.
    """

    def test_a_root_letter_is_not_stripped(self):
        attested = {"בינה", "חכמה", "בחכמה"}
        counts = skeleton_counts("בינה", attested)
        assert "בינה" in counts
        assert "ינה" not in counts, "ינה is not a word this corpus uses"

    def test_a_real_prefix_is_stripped(self):
        attested = {"בחכמה", "חכמה"}
        counts = skeleton_counts("בחכמה", attested)
        assert counts["חכמה"] == 1, "חכמה stands on its own, so ב is a prefix"

    def test_without_a_corpus_it_falls_back_to_shape(self):
        assert "ינה" in skeleton_counts("בינה")

    def test_binah_survives_a_full_build(self, stack):
        db = stack["db"]
        for i in range(5):
            _mem(db, f"bi{i}", f"אמר{ZWSP}בינה is third. Binah receives.")
        for i in range(9):
            _mem(db, f"bj{i}", f"אמר{ZWSP}דבר plainly. Yehuda listened.")
        lex = BilingualLexicon(stack["config"], stack["db"])
        got = {e["hebrew"] for e in lex.align(lex.scan()) if e["english"] == "Binah"}
        assert "בינה" in got
        assert "ינה" not in got
