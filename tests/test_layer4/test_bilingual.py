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
    hebrew_share,
    hebrew_words,
    skeleton_counts,
)
from chicory.layer4.wander import Patrol, _lexicon_definition
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
        bilingual_ubiquity_max=0.90, bilingual_min_hebrew_share=0.10,
        # Scaled to the fixtures the way min_chunks is: these corpora are
        # a handful of chunks, so the shipped joint-evidence floor of 6
        # would reject every pair in this file.
        bilingual_min_shared=2,
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


class TestCommentaryDilution:
    """A volume can carry the English and be unable to carry the Hebrew.

    Twenty-nine works were ingested and this pass LOST 41 of its best
    entries. סנהדרין fell from dice 0.865 to 0.260 and סוטה from 0.952
    to 0.317 — not because their alignment weakened but because the
    English side got heavier. For סנהדרין, `n` stayed 16 and `shared`
    stayed 16; `m` went 21 to 107, and 86 of the 107 are the two
    Pritzker volumes: English commentary with inline Hebrew citations,
    median 2 distinct Hebrew words per chunk against the bilingual
    edition's 191. They carry the English term, their Hebrew cannot
    align, and every one of those 722 chunks is pure Dice denominator.

    39 of the 41 come back under the gate. The other two, מדרש and
    הנעלם, recover their dice — 0.638 and 0.633, well clear of the 0.45
    floor — and were then folded onto דרש and נעלם by the fold's
    shortest-form rule. That was a fold artifact, not a gate failure, and
    both are back now that the fold asks for evidence; see
    TestFoldResidual. A test that
    demanded them back would be testing the fold.
    """

    # Three Hebrew words against nine English: share 0.250, deliberately
    # near the low end of the real facing band (p10 0.385, median 0.475).
    FACING = (ZWSP.join(["אמר", "סנהדרין", "קרבן"])
              + " the offering of the court, if they erred. Sanhedrin.")
    # One bare citation in a page of English: share 0.033, inside the
    # measured commentary band (p90 0.019, max 0.273).
    COMMENTARY = (
        "בבלי See the discussion of this passage in Sanhedrin, where the "
        "matter of the court and its offering is treated at length by the "
        "later authorities and again in the printed commentaries."
    )
    FILLER = "אמר" + ZWSP + "דבר plainly. Yehuda listened."

    def _lex(self, stack, share=None):
        cfg = stack["config"]
        if share is not None:
            cfg.bilingual_min_hebrew_share = share
        return BilingualLexicon(cfg, stack["db"])

    def test_a_dense_facing_chunk_is_counted(self, stack):
        for i in range(4):
            _mem(stack["db"], f"f{i}", self.FACING)
        assert self._lex(stack).scan()["chunks"] == 4

    def test_a_commentary_chunk_with_one_citation_is_not_counted(self, stack):
        """One Hebrew word in a page of English is a citation, not a
        translation. Measured, those chunks sit at median share 0.008 and
        722 of them were being counted."""
        for i in range(4):
            _mem(stack["db"], f"c{i}", self.COMMENTARY)
        assert self._lex(stack).scan()["chunks"] == 0

    def test_a_term_lost_to_the_denominator_comes_back(self, stack):
        """The signature of this failure is that only `m` moves.

        סנהדרין held n=16 and shared=16 through the whole ingest; `m`
        went 21 to 107 and dice went 0.865 to 0.260. Reproduced here at
        four and four: twelve commentary chunks carry `Sanhedrin` without
        the Hebrew it names, `m` goes 4 to 16, and dice 1.000 to 0.400
        drops the entry under the 0.45 floor. Excluding them restores it
        untouched.
        """
        db = stack["db"]
        for i in range(4):
            _mem(db, f"f{i}", self.FACING)
        for i in range(12):
            _mem(db, f"c{i}", self.COMMENTARY)
        for i in range(9):
            _mem(db, f"z{i}", self.FILLER)

        loose = self._lex(stack, share=0.0)
        assert "סנהדרין" not in {e["hebrew"] for e in loose.align(loose.scan())}

        lex = self._lex(stack, share=0.10)
        rec = {e["hebrew"]: e for e in lex.align(lex.scan())}["סנהדרין"]
        assert rec["english"] == "Sanhedrin"
        assert (rec["chunks"], rec["shared"], rec["english_chunks"]) == (4, 4, 4)

    def test_the_commentary_does_not_name_its_own_citation(self, stack):
        """Dilution is only half of it.

        The commentary carries `Sanhedrin` beside one unrelated inline
        citation, so that citation co-occurs with the term perfectly:
        ungated, בבלי takes `Sanhedrin` while the word the English is
        actually translating is cut. The losing entry and the winning one
        are the same event.
        """
        db = stack["db"]
        for i in range(4):
            _mem(db, f"f{i}", self.FACING)
        for i in range(12):
            _mem(db, f"c{i}", self.COMMENTARY)
        for i in range(9):
            _mem(db, f"z{i}", self.FILLER)
        loose = self._lex(stack, share=0.0)
        assert "בבלי" in {e["hebrew"] for e in loose.align(loose.scan())
                          if e["english"] == "Sanhedrin"}
        lex = self._lex(stack, share=0.10)
        got = {e["hebrew"] for e in lex.align(lex.scan())
               if e["english"] == "Sanhedrin"}
        assert "סנהדרין" in got
        assert "בבלי" not in got, "a citation is not a translation"

    def test_the_share_counts_english_words_not_english_terms(self):
        """The denominator is every distinct English word, not the terms
        the extractor keeps.

        Both readings separate the editions, at very different scales.
        Over the whole vocabulary the commentary sits near 0.008 against
        the facing text's 0.475. Over `english_terms` — capitalised,
        stop-listed, deduplicated — it rises to 0.062 against 0.903, and
        the same 0.10 gate that keeps 2 commentary chunks of 722 would
        keep 200. The knob would read the same and do almost nothing.
        """
        text = self.COMMENTARY
        assert hebrew_share(text) < 0.10
        by_terms = 1 / (1 + len(english_terms(text)))
        assert by_terms > 0.10, (
            "the wrong denominator puts this chunk on the wrong side"
        )

    def test_a_page_with_no_english_is_still_counted(self, stack):
        """The gate asks how much of the chunk is Hebrew, not whether an
        English side exists. Gating on the presence of English was
        measured and is identical to no gate at all — every commentary
        chunk has English — while gating it the other way would drop
        genuine facing chunks that carry very little.
        """
        pure = ZWSP.join(["אמר", "סנהדרין", "קרבן"])
        _mem(stack["db"], "pure", pure)
        assert hebrew_share(pure) == 1.0
        assert self._lex(stack).scan()["chunks"] == 1

    def test_a_chunk_exactly_at_the_threshold_is_kept(self):
        """The comparison is `>=`.

        It never bites in production — the two populations' bulk sits two
        orders of magnitude apart — but their tails DO overlap, facing
        text down to 0.041 and commentary up to 0.273, so the boundary is
        a real edge and not a formality.
        """
        text = "אמר one two three four five six seven eight nine"
        assert hebrew_share(text) == pytest.approx(0.1)
        assert hebrew_share(text) >= 0.10

    def test_a_gate_of_zero_counts_everything(self, stack):
        """The knob needs an off position or the measurement that
        produced it cannot be repeated: 365 aligned ungated against 380
        at 0.10 is the comparison, and the ungated half has to stay
        reachable from config.
        """
        db = stack["db"]
        for i in range(4):
            _mem(db, f"c{i}", self.COMMENTARY)
        assert self._lex(stack, share=0.0).scan()["chunks"] == 4

    def test_raising_the_gate_excludes_a_thinner_chunk(self, stack):
        """Not just on and off. The facing fixture scores 0.250 and the
        filler 0.400, so a gate of 0.35 keeps only the filler."""
        db = stack["db"]
        for i in range(3):
            _mem(db, f"f{i}", self.FACING)
        for i in range(3):
            _mem(db, f"z{i}", self.FILLER)
        assert self._lex(stack, share=0.10).scan()["chunks"] == 6
        assert self._lex(stack, share=0.35).scan()["chunks"] == 3

    def test_the_scan_says_what_it_left_out(self, stack):
        """This failure was invisible because the sweep reports what it
        counted and not what it refused: the number fell 6,436 to 5,667
        with nothing saying why."""
        db = stack["db"]
        for i in range(4):
            _mem(db, f"f{i}", self.FACING)
        for i in range(12):
            _mem(db, f"c{i}", self.COMMENTARY)
        out = self._lex(stack).scan()
        assert out["chunks"] == 4
        assert out["skipped"]["below_share"] == 12

    def test_a_commentary_fragment_is_not_a_word_the_corpus_uses(self, stack):
        """Attestation is only as good as the population it reads.

        The rule that separates a prefix from a root letter is that a
        real prefix is OPTIONAL, so its bare form turns up on its own.
        Commentary Hebrew breaks that: with no zero-width separator whole
        phrases arrive glued into one token, and 463 fragments entered
        the attested set as if they were words — then licensed strips on
        nine words of real facing text, כותיה to ותיה across 21 chunks.
        That is `ינה` again, through a different door.

        The aligned count is the same whether the first pass is gated or
        not, so nothing about the headline number tells you which
        population attestation read.
        """
        db = stack["db"]
        for i in range(4):
            _mem(db, f"f{i}", ZWSP.join(["כותיה", "אמר", "קרבן"])
                 + " the offering, he said. Sanhedrin.")
        # Glued commentary Hebrew: no separator, so this arrives as one
        # standalone token and would otherwise be attested as a word.
        for i in range(6):
            _mem(db, f"c{i}", "ותיה and the later authorities discuss this "
                              "passage of the court at considerable length "
                              "in their printed commentaries again.")
        assert "ותיה" not in set(self._lex(stack).scan()["hebrew"])


class TestFoldResidual:
    """The shorter form has to earn its seat, not win by arriving first.

    Prefix-stripping is a guess, so a skeleton and its debris both align
    to the same English word and only one can be kept. Which one is the
    artifact flips with the cause and the shapes are identical, so the
    fold used position as a stand-in for evidence — and on the live
    library that put 35 fragments on the record: מצרים as צרים, שכינתא as
    ינתא, המנונא as נונא, and בריך as the very ריך the docstring cites as
    the thing to avoid.

    The evidence is already in both records. `shared` counts the chunks
    that name the English term and carry the form, so the difference is
    how often the short form was seen there WITHOUT the long one.
    """

    def _counts(self, hebrew, english, together, chunks=40):
        return {"chunks": chunks, "hebrew": hebrew,
                "english": english, "together": together}

    def _lex(self, stack):
        return BilingualLexicon(stack["config"], stack["db"])

    def test_debris_loses_when_it_is_never_seen_alone(self, stack):
        """ריך appears only where בריך does, so it is debris and the
        long form takes the entry."""
        counts = self._counts(
            hebrew={"בריך": 20, "ריך": 20},
            english={"Berikh": 20},
            together={"בריך": {"Berikh": 20}, "ריך": {"Berikh": 20}})
        got = {e["hebrew"] for e in self._lex(stack).align(counts)}
        assert got == {"בריך"}

    def test_a_bare_root_keeps_its_entry_when_it_stands_alone(self, stack):
        """זעיר occurs in chunks where דזעיר does not, so it is a word in
        its own right and the construct form folds into it."""
        counts = self._counts(
            hebrew={"זעיר": 30, "דזעיר": 20},
            english={"Anpin": 30},
            together={"זעיר": {"Anpin": 30}, "דזעיר": {"Anpin": 20}})
        got = {e["hebrew"] for e in self._lex(stack).align(counts)}
        assert got == {"זעיר"}

    def test_the_shekhinta_case(self, stack):
        """The rule's old premise, stated in the code, was that a bare
        form 'always outnumbers any single prefixed variant'. On the live
        corpus ינתא at 501 chunks is RARER than both שכינתא (634) and
        כינתא (633) that were folded onto it."""
        counts = self._counts(
            hebrew={"ינתא": 501, "כינתא": 633, "שכינתא": 634},
            english={"Shekhinta": 640},
            together={"ינתא": {"Shekhinta": 501},
                      "כינתא": {"Shekhinta": 633},
                      "שכינתא": {"Shekhinta": 634}},
            chunks=1400)
        got = {e["hebrew"] for e in self._lex(stack).align(counts)}
        assert "ינתא" not in got, "the rarer fragment kept the entry"
        assert "שכינתא" in got

    def test_one_chunk_of_its_own_is_enough(self, stack):
        """The threshold is evidence existing, not evidence being common
        — a bare form seen once without its parent is still a word."""
        counts = self._counts(
            hebrew={"דרש": 21, "מדרש": 20},
            english={"Midrash": 21},
            together={"דרש": {"Midrash": 21}, "מדרש": {"Midrash": 20}})
        got = {e["hebrew"] for e in self._lex(stack).align(counts)}
        assert got == {"דרש"}


class TestClusterCap:
    """A word translates a word; a quoted verse translates nothing.

    A verse quoted N times makes every content word in it co-occur
    perfectly, so Dice cannot tell the translation from its neighbours.
    Ps 119:18 gives ואביטה→'Uncover' a flat 1.000 while גל, the word that
    actually means uncover, is not in the cluster at all.
    """

    def _verse(self, n=12):
        """THREE words that always travel together, as a verse does.

        Three and not four on purpose: the nine clusters this cap change
        removes are all size three, so a fixture of four would pass at a
        cap of 3 as well and prove nothing.
        """
        words = ["ואביטה", "מתורתך", "נפלאות"]
        return {"chunks": 40,
                "hebrew": {w: n for w in words},
                "english": {"Uncover": n},
                "together": {w: {"Uncover": n} for w in words}}

    def test_a_verse_block_is_dropped_whole(self, stack):
        lex = BilingualLexicon(stack["config"], stack["db"])
        assert lex.align(self._verse()) == [], (
            "four skeletons for one gloss is a block, not four translations"
        )

    def test_a_doublet_pair_survives(self, stack):
        """This corpus really does carry Hebrew/Aramaic doublets — פסח
        and פסחא, שבת and שבתא — and a cap of 1 would take them."""
        counts = {"chunks": 40,
                  "hebrew": {"שבת": 20, "שבתא": 16},
                  "english": {"Shabbat": 22},
                  "together": {"שבת": {"Shabbat": 20},
                               "שבתא": {"Shabbat": 16}}}
        got = {e["hebrew"] for e in
               BilingualLexicon(stack["config"], stack["db"]).align(counts)}
        assert got == {"שבת", "שבתא"}


class TestBuildGuardTracksItsDefinition:
    """Changing a cut must not wait for the cursor to wrap.

    The build is guarded to once per lap, and a lap here is the whole
    Hebrew-bearing population — ~170 batches at one per five idle
    minutes. When the fold, the floors and the cap were changed, the
    guard had no way to know, so the library went on serving a lexicon
    built under the old cuts, fragments included, until the cursor
    happened to come round. Same defect the corpus-asymmetry marker
    already carries a fingerprint for.
    """

    def test_the_marker_is_stable_when_nothing_is_edited(self, stack):
        cfg = stack["config"]
        assert _lexicon_definition(cfg) == _lexicon_definition(cfg)

    @pytest.mark.parametrize("knob,value", [
        ("bilingual_max_forms_per_term", 1),
        ("bilingual_min_dice", 0.6),
        ("bilingual_min_shared", 9),
        ("bilingual_min_chunks", 5),
        ("bilingual_ubiquity_max", 0.7),
        ("bilingual_min_hebrew_share", 0.2),
    ])
    def test_every_cut_changes_the_marker(self, stack, knob, value):
        """Each of these decides what comes out, so each must re-derive."""
        cfg = stack["config"]
        before = _lexicon_definition(cfg)
        setattr(cfg, knob, value)
        assert _lexicon_definition(cfg) != before, f"{knob} left the marker alone"

    def test_an_edited_cut_reopens_the_guard(self, stack):
        db = stack["db"]
        for i in range(4):
            _mem(db, f"g{i}", ZWSP.join(["אמר", "בינה"]) + " Rabbi said Binah.")
        p = stack["patrol"]
        p.run_errand("hebrew_lexicon")
        armed = db.execute("SELECT cursor_hint FROM wander_state "
                           "WHERE errand='hebrew_lexicon'").fetchone()["cursor_hint"]
        assert armed and armed.endswith(_lexicon_definition(stack["config"]))
        stack["config"].bilingual_min_dice = 0.99
        assert not armed.endswith(_lexicon_definition(stack["config"])), (
            "a changed cut must not match the armed marker"
        )
