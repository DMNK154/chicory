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


# A word ending in ח or ע carrying a FURTIVE PATACH — the vowel Hebrew
# writes under a final guttural — comes out of this edition's PDF with
# that guttural first, the line break behind it, and the rest of the word
# after.  See `_unrotate`.
_FURTIVE = re.compile(r"^([\u05d7\u05e2]\u05b7)\s*\n(.+)$", re.S)


def _unrotate(piece: str) -> str:
    """Undo the facing edition's line-broken furtive-patach words.

    `יְהוֹשֻׁעַ` is stored as ``'עַ\nְַיהֹוֻׁש'`` — ע plus its patach, a
    newline, then the rest — and reads out as `עיהוש`.  Measured on the
    live library: 3,766 occurrences, every one in the twelve HOLY ZOHAR
    volumes, recovering רוח 703 times, נח 167, רקיע 140, מזבח 116,
    יהושע 97, ריח 90, כח 72, משיח 70, שומע 40, יודע 37.

    Only ח and ע, because only they carry it here.  Judged against a
    96,955-token control built from the two Hebrew editions that have no
    zero-width separator (split those on whitespace; splitting them on
    the separator yields one token per chunk and no control at all): ח
    repairs 156 words against 122 it would break, ע 226 against 106.  ה
    is EXCLUDED on the same evidence — 27 occurrences, 0 repaired and 3
    broken — so the class is measured rather than derived from the
    grammar, which would have admitted it.

    Deliberately narrow.  "Swap whenever a piece holds a newline" is
    false: over 17,552 such pieces the stored form is the attested one by
    1,889 to 414.  The furtive patach is what separates a word the
    extractor rotated from a line break that did no harm.
    """
    m = _FURTIVE.match(piece)
    if m is None:
        return piece
    lead, rest = m.group(1), m.group(2)
    if not any(_CONSONANT_MIN <= ch <= _CONSONANT_MAX for ch in rest):
        return piece                      # nothing to put in front of it
    return rest + lead


def hebrew_words(text: str) -> list[str]:
    """Consonantal skeletons, in order.

    Splitting on the zero-width space is the whole trick.  Splitting on
    whitespace instead shreds every word into letters, which is why a
    naive scan finds גבורה three times in a corpus that contains it 213
    times.

    Repairs the facing edition's rotated furtive-patach words on the way
    through — see `_unrotate`.  This is the only place anything in the
    package tokenises Hebrew, so it is the only place the repair has to
    live.
    """
    out: list[str] = []
    for piece in text.split(_WORD_SEP):
        skeleton = "".join(
            ch for ch in _unrotate(piece)
            if _CONSONANT_MIN <= ch <= _CONSONANT_MAX
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
        in 5,097 of the 5,667 counted chunks, so they co-occur with every
        common word and score 0.985 against `Parashat` while meaning
        nothing — a term on nearly every page distinguishes nothing, the
        same reason the roster caps hub tags.

        And THE SHORTER FORM MUST EARN ITS SEAT.  Prefix-stripping is a
        guess, so `בריך` and its debris `ריך` both align to `Berikh` and
        only one can be kept — but which one is the artifact flips with
        the cause, and the two shapes are identical.  This once read
        "longest form wins", and said so in four places after the sort had
        been changed to keep the SHORTEST: the record then carried 35
        entries naming a fragment — `מצרים` as `צרים`, `שכינתא` as
        `ינתא`, `המנונא` as `נונא`, and `בריך` as the very `ריך` this
        paragraph cites as the thing to avoid.

        What decides it is already in both records.  `shared` counts the
        chunks that name the English term and carry the form, so
        `shared(short) - shared(long)` is how often the short form was
        seen there WITHOUT the long one.  Zero means it was never seen on
        its own and is debris.
        """
        chunks = max(counts["chunks"], 1)
        ceiling = self._config.bilingual_ubiquity_max * chunks
        floor = self._config.bilingual_min_chunks
        shared_floor = self._config.bilingual_min_shared
        min_dice = self._config.bilingual_min_dice

        best: dict[str, dict] = {}
        for skeleton, n in counts["hebrew"].items():
            if n < floor or n > ceiling:
                continue
            top = None
            for term, shared in counts["together"].get(skeleton, {}).items():
                m = counts["english"].get(term, 0)
                # The floor belongs on the JOINT evidence, not on how often
                # the English word happens to occur.  `m` is a term's total
                # frequency, and the facing-text gate deliberately shrank
                # the population it is counted over — the ceiling rescales
                # itself against `chunks` and the floor did not, so it
                # silently became ~12% stricter and took 17 entries whose
                # dice the same gate had just IMPROVED: ישראלית/Israelite
                # 0.48 to 0.80, אוריאל/Uriel 0.483 to 0.56.  What says
                # whether a pairing is a convention rather than a
                # coincidence is how often the two appear TOGETHER, and
                # the gate does not move that.
                if m > ceiling or shared < shared_floor:
                    continue
                d = dice(shared, n, m)
                if top is None or d > top["dice"]:
                    top = {"hebrew": skeleton, "english": term,
                           "dice": round(d, 4), "chunks": n,
                           "english_chunks": m, "shared": shared}
            if top and top["dice"] >= min_dice:
                best[skeleton] = top

        # Two skeletons aligned to the same English word, one a suffix of
        # the other: keep the one the corpus gives independent evidence
        # for.  See the residual test below.
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
            # Shortest first, but only to fix an order — the residual
            # test below decides, not the position.
            #
            # Two arguments stood here and both were wrong.  "A bare form
            # is emitted for every prefixed occurrence as well as its own,
            # so it always outnumbers any single prefixed variant": on the
            # live corpus that fails outright for `ינתא`, which at 501
            # chunks is RARER than both `שכינתא` (634) and `כינתא` (633)
            # that were folded onto it.  And "debris never reaches this
            # point, because the corpus never uses it alone": measured
            # over 121 fold pairs, all 121 dropped forms and all 121 kept
            # forms are attested standing alone, so attestation separates
            # nothing here.
            recs.sort(key=lambda r: (len(r["hebrew"]), -r["chunks"]))
            kept: list[dict] = []
            for r in recs:
                clash = next((k for k in kept
                              if k["hebrew"] != r["hebrew"]
                              and (k["hebrew"].endswith(r["hebrew"])
                                   or r["hebrew"].endswith(k["hebrew"]))), None)
                if clash is None:
                    kept.append(r)
                    continue
                # The shorter form has to EARN its seat rather than win by
                # arriving first.  Position was standing in for evidence,
                # and the evidence is already in both records: `shared` is
                # how many chunks name the English term and carry the
                # form, so `shared(short) - shared(long)` is the count of
                # chunks that name the term and carry the short form
                # WITHOUT the long one.  Zero of those means the short
                # form was never seen on its own here, and is debris.
                #
                # Measured over 121 fold pairs on the live library: 54
                # have a residual of exactly 0 and the old rule kept the
                # short form in every one of them, which is how 35
                # standing entries came to name a fragment — מצרים as
                # צרים, שמעון as עון, שכינתא as ינתא, המנונא as נונא, and
                # בריך as ריך, the very case the docstring cites as the
                # reason for the rule.  45 pairs have a residual of 8 or
                # more and are the genuine clitics: דישראל/ישראל 1305,
                # דקודשא/קודשא 1507.
                #
                # Attestation cannot do this job: all 121 dropped forms
                # AND all 121 kept forms are attested standing alone, so
                # it separates nothing here.
                if clash["shared"] - r["shared"] >= 1:
                    continue                      # the short form earned it
                kept.remove(clash)                # it did not; take the long
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
