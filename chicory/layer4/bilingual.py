"""Reading a facing-text corpus in both of its languages.

Twelve volumes of the Zohar in this library are bilingual — Aramaic and
Hebrew beside an English translation — and the English channel could
reach almost none of them.  Measured: the concept extractor tagged 92 of
their 5,719 chunks, 1.6%, because it looks for a capitalised word used
four or more times in one chunk and this edition uses its vocabulary
sparsely across many chunks instead.

The Hebrew is a better key for three reasons, and none of them is subtle.

It is INVARIANT.  ``gvurah`` and ``gevurah`` are two editions' romanisation
of one word; גבורה is גבורה in both.  The whole edition-variant problem
that produced tag_gap_variants exists only in the transliteration.

It needs NO HEURISTIC.  The English extractor guesses termhood from
capitalisation, which is why it needs a long stop-list and a four-mention
floor, and why it still admits `Called` and `Namely`.  A Hebrew string in
an English sentence is a term by construction.

It has NO FLOOR PROBLEM.  ``_CONCEPT_RE`` requires five letters, so הוד
(hod) and דעת (daat) — two of the ten sefirot — could never be extracted
at all.  In Hebrew they are found 90 and 153 times.

Measured against the English channel on the same volumes: chochmah 35
chunks to 311, chesed 32 to 271, malchut 2 to 117, and seven sefirot that
went from zero to found.  17.1% of the volumes rather than 1.6%.

The extraction has a signature worth stating, because nothing works until
it is handled: the PDF text uses U+200B ZERO WIDTH SPACE as the word
separator and ordinary spaces INSIDE words, with nikkud as combining
marks throughout.  So ``המשכילים`` arrives as ``ְו ַהַּמ ְׂש ִּכ יִלים``.
Split on the zero-width space, keep only the 22 consonants, and the word
comes back.

Then English names what Hebrew found.  Because the translation faces the
source, the mapping can be LEARNED rather than supplied: a skeleton and
its English word appear in the same chunks, and Dice over that
co-occurrence recovers משה→Moshe, מטטרון→Metatron, המנונא→Hamnuna and
אדהכי→Meanwhile without a dictionary.

FACING is the load-bearing word, and it is not the same as bilingual.
Two more Zohar editions arrived and this pass lost 41 of its best
entries — סנהדרין/Sanhedrin fell from dice 0.865 to 0.260, סוטה/Sotah
from 0.952 to 0.317.  Nothing about their alignment had weakened: for
סנהדרין the chunk count `n` stayed at 16 and the shared count stayed at
16.  What moved was `m`, from 21 to 107, and 86 of the 107 were the
Pritzker volumes — English commentary carrying inline Hebrew citations,
median 2 Hebrew words per chunk against this edition's 191.  Such a
chunk clears "has a skeleton" on its one citation and then contributes
its entire English vocabulary to a denominator it can never help.

So the population is chunks where the Hebrew IS the text, measured by
`hebrew_share`, and not chunks that merely contain Hebrew.  Two other
new volumes turned out to be inert rather than harmful: lacking the
zero-width separator, they glue each chunk into one over-long skeleton
and were already being dropped.
"""

from __future__ import annotations

import re
from collections import Counter

# The 22 consonants, final forms included.  Nikkud (U+0591-U+05C7) and the
# cantillation marks are combining characters and are simply not in this
# range, so filtering to it removes them without a separate pass.
_CONSONANT_MIN, _CONSONANT_MAX = "א", "ת"

# The word separator the extraction actually uses.  Ordinary spaces appear
# INSIDE words and must not split them.
_WORD_SEP = "​"

# Letters that attach to the front of a word: vav (and), he (the), bet
# (in), lamed (to), mem (from), shin (that), kaf (like) — and dalet, which
# is the point.  The Zohar is largely ARAMAIC, where ד is the genitive and
# relative particle and by far the commonest prefix of all.
#
# Leaving it out was not a small omission.  It left דעתיקא, דזעיר, דכפורי
# and דאתי in the lexicon as separate words, and because a construct form
# never appears without the thing it governs, they aligned to their
# PARTNER rather than themselves: והוד ("and Hod") mapped to `Netzach`,
# since the pair Netzach-and-Hod is never split.
PREFIX_LETTERS = "והבלמשכד"

# Long enough to be a word, short enough not to be a phrase glued together
# by a missing separator.
MIN_SKELETON, MAX_SKELETON = 3, 9

_ENGLISH_RE = re.compile(r"\b([A-Z][a-zA-Z]{3,})\b")

# Every English word, not just the ones that could be terms.  Used only
# to measure how much of a chunk is Hebrew — see `hebrew_share`.
_ENGLISH_WORD_RE = re.compile(r"[A-Za-z]{2,}")

# Capitalised words that carry no subject in this material: sentence
# openers, the apparatus of the edition, and the names that appear on
# nearly every page and would therefore align with everything.
_ENGLISH_STOP = frozenset("""
the and but for that this with from come behold said holy one blessed
book zohar torah what when then there they their have been which because
therefore however indeed thus here now all not you your his her its him
she was were will would should could shall may might must can rabbi
verse portion chapter section page volume english aramaic hebrew
""".split())


def hebrew_words(text: str) -> list[str]:
    """Consonantal skeletons, in order.

    Splitting on the zero-width space is the whole trick.  Splitting on
    whitespace instead shreds every word into letters, which is why a
    naive scan finds גבורה three times in a corpus that contains it 213
    times.
    """
    out: list[str] = []
    for piece in text.split(_WORD_SEP):
        skeleton = "".join(
            ch for ch in piece if _CONSONANT_MIN <= ch <= _CONSONANT_MAX
        )
        if skeleton:
            out.append(skeleton)
    return out


def base_forms(word: str) -> list[str]:
    """The word, plus what it looks like with attached prefixes removed.

    Returned longest-first so a caller that wants one answer takes the
    unstripped form: stripping is a guess, and ``ריך`` from ``בריך`` is
    the shape of that guess going wrong.
    """
    forms = [word]
    for k in (1, 2):
        if len(word) - k < MIN_SKELETON:
            break
        if not all(ch in PREFIX_LETTERS for ch in word[:k]):
            break
        forms.append(word[k:])
    return forms


def english_terms(text: str) -> set[str]:
    """Capitalised words in the facing translation, minus the furniture."""
    return {
        w for w in _ENGLISH_RE.findall(text)
        if w.lower() not in _ENGLISH_STOP
    }


def hebrew_share(text: str) -> float:
    """How much of this chunk is Hebrew, as a share of its vocabulary.

    The test for facing text.  ``english_terms`` is the wrong denominator
    here — capitalised, stop-listed and deduplicated, it is small enough
    that a page of commentary still scores 0.062, and a gate that admits
    that admits 200 of 722 commentary chunks.  Every distinct word on
    both sides is the comparison the separation was measured on.

    TYPES rather than tokens on the English side, so the numerator and
    the denominator are counted the same way — and because an apparatus
    grows its token count linearly and its vocabulary sublinearly, a
    token denominator would put the whole of a footnote-heavy edition
    under the gate at roughly five times this edition's English per
    chunk, where types survive to about eighteen.

    Deliberately independent of ``attested``: the attestation pass is
    what BUILDS that set, and it needs gating too — ungated it admits
    463 fragments glued out of unseparated commentary Hebrew, which then
    license prefix strips on real facing-text words.
    """
    heb = {w for w in hebrew_words(text)
           if MIN_SKELETON <= len(w) <= MAX_SKELETON}
    eng = {w.lower() for w in _ENGLISH_WORD_RE.findall(text)}
    total = len(heb) + len(eng)
    return (len(heb) / total) if total else 0.0


def skeleton_counts(text: str, attested: set[str] | None = None) -> Counter:
    """Every usable skeleton in a chunk, counted, prefixes folded in.

    ``attested`` is the set of words the corpus actually uses on their own,
    and without it this cannot work.  A prefix letter and a root letter
    look identical: ב opens both בחכמה (in-wisdom) and בינה (Binah), and מ
    opens both מחכמה and מלכות.  Strip by shape and Binah becomes ``ינה``
    and Malchut becomes ``לכות``, neither of which is a word.

    A real prefix is OPTIONAL, so its bare form turns up somewhere on its
    own — זעיר appears beside דזעיר — while a root fragment never does.
    Passing the attested set makes that the test.  Passing nothing keeps
    the shape-only behaviour, which is right for a single-chunk caller
    that has no corpus to consult.
    """
    counts: Counter = Counter()
    for word in hebrew_words(text):
        if not (MIN_SKELETON <= len(word) <= MAX_SKELETON):
            continue
        for form in base_forms(word):
            if not (MIN_SKELETON <= len(form) <= MAX_SKELETON):
                continue
            if attested is not None and form != word and form not in attested:
                continue
            counts[form] += 1
    return counts


def dice(shared: int, a_total: int, b_total: int) -> float:
    """Overlap of two chunk-sets, normalised by both sizes.

    Chosen over raw co-occurrence because the corpus is dominated by a few
    ubiquitous words: ``ספר`` and ``זהר`` appear in 5,090 of 5,719 chunks,
    so anything else common co-occurs with them constantly while meaning
    nothing.  Dice punishes that on the denominator — but not enough on
    its own, which is why callers also apply a ubiquity cut.
    """
    total = a_total + b_total
    return (2.0 * shared / total) if total else 0.0


class BilingualLexicon:
    """Builds a Hebrew-to-English lexicon from the facing translation.

    A whole-corpus sweep rather than a per-chunk deposit.  The alignment
    is a ratio over chunk-sets, so it means nothing until both sides are
    fully counted, and 5,719 chunks is small enough to hold at once — the
    per-term deposit that ``concept_gaps`` uses would write roughly three
    hundred observations per chunk to compute the same thing.
    """

    def __init__(self, config, db) -> None:
        self._config = config
        self._db = db

    def scan(self, limit: int | None = None) -> dict:
        """Count skeletons, English terms, and their co-occurrence."""
        heb_chunks: Counter = Counter()
        eng_chunks: Counter = Counter()
        together: dict[str, Counter] = {}
        seen = 0
        # The Unicode range, not the aleph shortcut the errand's population
        # uses.  That shortcut is a cursor optimisation \u2014 0.4s against 3.4s
        # over the same 5,7xx memories \u2014 and it is an approximation: `\u05DE\u05E9\u05D4`
        # contains no aleph at all.  The sweep that does the counting
        # cannot afford to miss a chunk silently, so it pays the full scan
        # once per lap.
        sql = ("SELECT content FROM memories WHERE is_archived = 0 "
               "AND content GLOB '*[\u05D0-\u05EA]*'")
        if limit:
            sql += f" LIMIT {int(limit)}"
        # Containing Hebrew is not the same as BEING facing text, and the
        # difference is what broke this pass.  A chunk below the share is
        # commentary with a citation in it: it can carry the English term
        # but not the Hebrew that names it, so it is denominator and
        # nothing else.  Gated on BOTH passes, because attestation reading
        # a wider corpus than the count is how ``\u05D5\u05EA\u05D9\u05D4`` becomes a word.
        floor = self._config.bilingual_min_hebrew_share
        skipped = {"below_share": 0, "no_skeletons": 0}
        kept: list[str] = []
        for row in self._db.execute(sql):
            text = row["content"] or ""
            if hebrew_share(text) < floor:
                skipped["below_share"] += 1
                continue
            kept.append(text)

        # First pass: which words does the corpus use on their own?  That
        # is the only thing that separates a prefix from a root letter.
        attested: set[str] = set()
        for text in kept:
            for w in hebrew_words(text):
                if MIN_SKELETON <= len(w) <= MAX_SKELETON:
                    attested.add(w)

        for text in kept:
            skeletons = set(skeleton_counts(text, attested))
            if not skeletons:
                skipped["no_skeletons"] += 1
                continue
            english = english_terms(text)
            seen += 1
            for s in skeletons:
                heb_chunks[s] += 1
            for e in english:
                eng_chunks[e] += 1
            for s in skeletons:
                pair = together.setdefault(s, Counter())
                for e in english:
                    pair[e] += 1
        return {"chunks": seen, "hebrew": heb_chunks,
                "english": eng_chunks, "together": together,
                # What it refused, because a sweep that reports only what
                # it counted is how 722 diluting chunks went unnoticed for
                # a month: the number fell and nothing said why.
                "skipped": skipped}

    def align(self, counts: dict) -> list[dict]:
        """Pair each skeleton with the English word that tracks it.

        Two cuts that the raw ratio needs.  UBIQUITY: `ספר` and `זהר` sit
        in 5,090 of 5,719 chunks, so they co-occur with every common word
        and score 0.985 against `Parashat` while meaning nothing — a term
        on nearly every page distinguishes nothing, the same reason the
        roster caps hub tags.  And LONGEST FORM WINS: prefix-stripping is
        a guess, so `בריך` and its debris `ריך` both align to `Berikh`,
        and only the longer one is kept.
        """
        chunks = max(counts["chunks"], 1)
        ceiling = self._config.bilingual_ubiquity_max * chunks
        floor = self._config.bilingual_min_chunks
        min_dice = self._config.bilingual_min_dice

        best: dict[str, dict] = {}
        for skeleton, n in counts["hebrew"].items():
            if n < floor or n > ceiling:
                continue
            top = None
            for term, shared in counts["together"].get(skeleton, {}).items():
                m = counts["english"].get(term, 0)
                if m < floor or m > ceiling:
                    continue
                d = dice(shared, n, m)
                if top is None or d > top["dice"]:
                    top = {"hebrew": skeleton, "english": term,
                           "dice": round(d, 4), "chunks": n,
                           "english_chunks": m, "shared": shared}
            if top and top["dice"] >= min_dice:
                best[skeleton] = top

        # Longest form wins: drop a skeleton that is a suffix of a longer
        # one aligned to the same English word.
        out = []
        by_english: dict[str, list[dict]] = {}
        for rec in best.values():
            by_english.setdefault(rec["english"], []).append(rec)
        max_forms = self._config.bilingual_max_forms_per_term
        for term, recs in by_english.items():
            # When one skeleton is a suffix of another, one of them is an
            # artifact — but WHICH one flips depending on the cause, and
            # the shapes are identical.  `ריך` is debris from stripping
            # `בריך`; `דזעיר` is a real construct form whose bare word
            # `זעיר` is the one worth keeping.
            #
            # Count settles it without needing to know.  A bare form is
            # emitted for every prefixed occurrence as well as its own, so
            # it always outnumbers any single prefixed variant.  Debris,
            # emitted only alongside its parent, ties — and the longer form
            # takes the tie.
            # Shortest first, now that attestation has done the harder
            # job.  Debris like ריך and ינה never reaches this point,
            # because the corpus never uses them alone — so if two forms
            # are BOTH here and one is a suffix of the other, the shorter
            # is the bare root and the longer carries a prefix.
            recs.sort(key=lambda r: (len(r["hebrew"]), -r["chunks"]))
            kept: list[dict] = []
            for r in recs:
                if any(k["hebrew"].endswith(r["hebrew"]) or
                       r["hebrew"].endswith(k["hebrew"])
                       for k in kept if k["hebrew"] != r["hebrew"]):
                    continue
                kept.append(r)
            # A word translates a word.  If one English term still draws
            # many unrelated skeletons after the suffix fold, they arrived
            # together in a repeated block rather than as translations,
            # and nothing in the co-occurrence distinguishes them — so the
            # cluster goes, rather than an arbitrary member of it.
            if len(kept) > max_forms:
                continue
            out.extend(kept)
        out.sort(key=lambda r: -r["dice"])
        return out

    def build(self, patrol=None, lap: int = 0, limit: int | None = None) -> dict:
        """Scan, align, and record the lexicon as observations."""
        counts = self.scan(limit)
        entries = self.align(counts)
        if patrol is not None:
            for rec in entries:
                patrol.observe(
                    errand="hebrew_lexicon",
                    kind="bilingual_term",
                    subject=rec["hebrew"],
                    detail=(
                        f"{rec['hebrew']} tracks {rec['english']!r} across "
                        f"{rec['shared']} of {rec['chunks']} chunks "
                        f"(dice {rec['dice']})"
                    ),
                    weight=round(1.0 - rec["dice"], 6),
                    data={k: v for k, v in rec.items() if k != "hebrew"},
                    lap=lap,
                )
            self._db.connection.commit()
        return {
            "chunks_scanned": counts["chunks"],
            "skipped": counts.get("skipped", {}),
            "skeletons": len(counts["hebrew"]),
            "english_terms": len(counts["english"]),
            "aligned": len(entries),
            "top": entries[:15],
        }
