"""One thing represented twice, and where the representations disagree.

Every question this library generated that was worth asking came from an
asymmetry with TWO sides.  ``Kudsha Berikh`` appears 2,539 times in one
Zohar edition and zero times in the other two.  ``namely`` is in 36% of
one edition's chunks and 0.0% of another's.  ``companions`` is in 39.2%
of the scholarship and 0.03% of the primary texts it describes.

The ones that were worthless had only one side.  A tag pair scoring 0.997
on centroid similarity — ``purpose ~ potential``, ``creativity ~
responsibility`` — is two DIFFERENT things whose vectors happen to sit
close, so there is nothing for a disagreement to be about.  3,724 of
those, and not one made a question.

So the rule is narrower than "find an asymmetry": find the SAME thing
represented twice, and measure where the two accounts diverge.  A concept
in two languages.  A text in three witnesses.  A subject described from
inside and from outside.  What each pair has in common is that both sides
are supposed to be talking about the same material, which is what makes a
divergence mean something.

What this cannot do is decide which divergences matter.  ``namely``
topped one comparison because an edition fuses its commentary into the
running text — genuinely the most interesting thing in that pair — while
``pages`` topped another because one side has an index.  Nothing in the
numbers separates those, which is the same wall every filter in this
system has hit.  So both rates go on the record and a reader decides.

A pair used to be a pair of SQL LIKE predicates naming files, and that is
how this module's own flagship example became false.  Twenty-nine works
arrived on one day; two of them were the Pritzker Zohar, which IS primary
text and matched no pattern, so "the primary text" silently meant four
files out of eight.  ``companions`` stayed on the record as being in 39.2%
of the scholarship and 0.03% of the primary text while sitting in 5.79% of
Pritzker — 95 chunks of the very thing it was said to be absent from.

So membership is a property of the DOCUMENT now, not of its name.  Each
work is identified by a digest over its chunk hashes and given a role in a
registry a person maintains; a side of a comparison names roles.  A
filename cannot drift out of a population it belongs to, and two copies of
one work cannot both be counted: the same book arrived that day as
``dokumen.pub_a-river-flows-from-eden-...`` beside the
``A_River_Flows_from_Eden_-_Melila_Hellner-Eshed`` already shelved, 546
chunks each and byte-identical, and only the LIKE pattern's failure to
match the new name kept the scholarship side from silently doubling.

What a duplicate does depends on where it lands.  In a homogeneous side it
is a rescale that cancels — the Kabbalah Centre edition counted twice gave
150 findings and the identical term set.  In a heterogeneous union it is
dilution: that same duplicate took the primary text from 15,508 chunks to
24,476, every scholarship rate measured against it fell by 1.578, and 11
of the 60 findings on record were manufactured by the drop.

And a chunk that yields no term at all is denominator and nothing else.
Two of the new works are Hebrew-script Zohar editions: 1,571 chunks
between them and not one capitalised Latin term, so they can never supply
a numerator while lowering every rate measured against their side.  That
is the shape of the commentary chunks that cost the bilingual lexicon 41
entries, and here it needs no threshold at all — a chunk counts when it
bears a usable term, and the populations separate absolutely: 0 of 1,571
against 20,842 of 21,269 everywhere else.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

_log = logging.getLogger(__name__)

# The same capitalisation heuristic the concept walk uses, so a term found
# here is a term found there and the two are comparable.
_TERM_RE = re.compile(r"\b([A-Z][a-zA-Z]{3,})\b")

# Documents are digested in batches so a library of any size costs one
# scan, and profiled in batches because SQLite caps host parameters.
_PATH_BATCH = 400


def document_digest(fingerprints) -> str:
    """Identify a work by what it holds, not by what it was called.

    Order-independent, so a re-ingest that renumbers chunks lands on the
    same digest, and truncated to 16 hex characters so a registry entry
    stays readable beside the title it names.
    """
    h = hashlib.sha256()
    for fp in sorted(fingerprints):
        h.update(fp.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class Document:
    """One work, however many filenames the library has for it."""

    digest: str
    canonical: str                  # the copy that arrived first
    copies: tuple[str, ...]         # every path carrying this content
    chunks: int                     # chunks in ONE copy
    role: str | None = None
    title: str | None = None

    @property
    def duplicates(self) -> tuple[str, ...]:
        return tuple(p for p in self.copies if p != self.canonical)


@dataclass(frozen=True)
class Membership:
    """Which documents a side resolved to, and which it could not.

    Recorded on every observation the comparison produces.  An asymmetry
    is a claim about two populations, and a reader who cannot see which
    files were in them cannot tell a finding from an artefact of that
    day's ingest — which is exactly how ``companions`` stayed on record
    for a day claiming 0.03% of a primary text holding 5.79%.
    """

    roles: tuple[str, ...] = ()
    documents: tuple[str, ...] = ()     # canonical paths, sorted
    digests: tuple[str, ...] = ()
    duplicates: tuple[str, ...] = ()    # paths dropped as identical copies
    missing: tuple[str, ...] = ()       # registry entries matching nothing

    @property
    def basis(self) -> str:
        """A fingerprint of the definition this side was measured under.

        Changes when a document joins or leaves the side, which is the
        signal that every finding computed under the old definition has to
        be withdrawn rather than left standing over new material.
        """
        h = hashlib.sha256()
        for d in sorted(self.digests):
            h.update(d.encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()[:12]


class CorpusRegistry:
    """Which work is which, and what role it plays.

    The registry is authored, not inferred.  Nothing in the numbers can
    decide that the Pritzker Zohar is primary text and Hellner-Eshed is
    scholarship about it — that is an editorial judgement, and the only
    part worth automating is making its absence visible.  So a document
    outside the registry has no role and joins no comparison, and a
    registry entry resolving to nothing is reported rather than dropped.
    """

    def __init__(self, config, db) -> None:
        self._config = config
        self._db = db
        self._docs: dict[str, Document] | None = None

    def _scan(self) -> dict[str, Document]:
        """Digest every document in the library, once.

        Reads ``content_hash`` rather than content — an indexed column of
        short strings — and falls back to hashing content only for rows
        that have none, which on the live library is 2,935 of 155,279.  A
        NULL hash folded to an empty string would give every such document
        the same digest as any other with the same chunk count, and that
        is a collision this cannot afford.
        """
        declared = {e["digest"]: e for e in self._config.corpus_documents
                    if e.get("digest")}

        fps: dict[str, list[str]] = defaultdict(list)
        first: dict[str, str] = {}
        for row in self._db.execute(
            "SELECT source_path, content_hash, created_at, "
            "CASE WHEN content_hash IS NULL THEN content ELSE NULL END AS raw "
            "FROM memories WHERE is_archived = 0 AND source_path IS NOT NULL"
        ):
            path = row["source_path"]
            fps[path].append(row["content_hash"] or hashlib.sha256(
                (row["raw"] or "").encode("utf-8")).hexdigest()[:16])
            created = row["created_at"] or ""
            if path not in first or created < first[path]:
                first[path] = created

        by_digest: dict[str, list[str]] = defaultdict(list)
        for path, chunk_fps in fps.items():
            by_digest[document_digest(chunk_fps)].append(path)

        docs: dict[str, Document] = {}
        for digest, paths in by_digest.items():
            # The copy that was already here is the one that counts and
            # the later arrival is the duplicate, which is the reading a
            # person would give it.
            copies = tuple(sorted(paths, key=lambda p: (first.get(p, ""), p)))
            entry = declared.get(digest, {})
            docs[digest] = Document(
                digest=digest, canonical=copies[0], copies=copies,
                chunks=len(fps[copies[0]]),
                role=entry.get("role"), title=entry.get("title"),
            )
        return docs

    def documents(self) -> dict[str, Document]:
        """Every distinct work, keyed by digest.  Scanned once."""
        if self._docs is None:
            self._docs = self._scan()
        return self._docs

    def digest_for(self, path: str) -> str | None:
        """The digest of one document, for putting it in the registry.

        A person adding a work needs its digest, and deriving one by hand
        from the schema is exactly the step that does not get taken.
        """
        rows = self._db.execute(
            "SELECT content_hash, CASE WHEN content_hash IS NULL THEN content "
            "ELSE NULL END AS raw FROM memories "
            "WHERE is_archived = 0 AND source_path = ?", (path,)
        ).fetchall()
        if not rows:
            return None
        return document_digest(
            r["content_hash"] or hashlib.sha256(
                (r["raw"] or "").encode("utf-8")).hexdigest()[:16]
            for r in rows)

    def duplicated(self) -> list[Document]:
        """Works the library holds under more than one name."""
        return sorted((d for d in self.documents().values() if d.duplicates),
                      key=lambda d: -d.chunks)

    def resolve(self, roles) -> Membership:
        """Every document holding one of these roles, each counted once."""
        wanted = tuple(roles or ())
        docs = sorted(
            (d for d in self.documents().values()
             if d.role is not None and d.role in wanted),
            key=lambda d: d.canonical,
        )
        held = self.documents()
        # A registry entry naming a role on this side but matching no
        # document is a work that was renamed, re-chunked or removed.  It
        # is the one failure this scheme has and the old one did not, so
        # it goes on the record rather than being left to be noticed.
        missing = tuple(sorted(
            e.get("title") or e.get("digest", "?")
            for e in self._config.corpus_documents
            if e.get("role") in wanted and e.get("digest") not in held
        ))
        return Membership(
            roles=wanted,
            documents=tuple(d.canonical for d in docs),
            digests=tuple(sorted(d.digest for d in docs)),
            duplicates=tuple(sorted(p for d in docs for p in d.duplicates)),
            missing=missing,
        )


class CorpusAsymmetry:
    """Diffs the vocabulary of two populations that describe one thing."""

    def __init__(self, config, db, registry: CorpusRegistry | None = None) -> None:
        self._config = config
        self._db = db
        self._registry = registry if registry is not None else CorpusRegistry(config, db)

    @property
    def registry(self) -> CorpusRegistry:
        return self._registry

    def _profile(self, paths) -> tuple[Counter, int, int]:
        """How often each capitalised term appears, by CHUNK not by mention.

        Per chunk because the two sides are usually different sizes — the
        scholarship is 546 chunks against 17,010 of primary text — and a
        raw count would say nothing but "one side is bigger".

        The population is chunks that BEAR a term.  A chunk with none can
        never contribute to a numerator and contributes to every
        denominator, so counting it lowers a side's rates by a factor that
        has nothing to do with vocabulary.  Both gates here are absolute —
        clear ``asymmetry_min_rate`` on one side, fall under
        ``asymmetry_max_other`` on the other — so a one-sided dilution
        does not cancel out.  It pushes terms across the second gate and
        invents findings.

        No threshold, because none is needed: of the works in the Zohar
        comparison the two Hebrew-script editions bear a term in 0 of
        their 1,571 chunks and every other bears one in 20,842 of 21,269.
        """
        from chicory.layer4.wander import _CONCEPT_STOPWORDS

        counts: Counter = Counter()
        bearing = total = 0
        paths = tuple(paths)
        for start in range(0, len(paths), _PATH_BATCH):
            batch = paths[start:start + _PATH_BATCH]
            ph = ",".join("?" * len(batch))
            for row in self._db.execute(
                f"SELECT content FROM memories WHERE is_archived = 0 "
                f"AND source_path IN ({ph})", batch
            ):
                total += 1
                terms = {t.lower() for t in _TERM_RE.findall(row["content"] or "")}
                terms -= _CONCEPT_STOPWORDS
                if not terms:
                    continue
                bearing += 1
                for term in terms:
                    counts[term] += 1
        return counts, bearing, total

    def compare(self, pair: dict) -> dict:
        """One declared pair, diffed both ways."""
        cfg = self._config
        left_m = self._registry.resolve(pair.get("left_roles"))
        right_m = self._registry.resolve(pair.get("right_roles"))
        left, ln, l_all = self._profile(left_m.documents)
        right, rn, r_all = self._profile(right_m.documents)
        base = {"pair": pair["name"], "left": left_m, "right": right_m,
                "left_chunks": ln, "right_chunks": rn,
                "left_chunks_all": l_all, "right_chunks_all": r_all,
                "findings": []}
        floor = cfg.asymmetry_min_population
        if ln < floor or rn < floor:
            gone = left_m.missing + right_m.missing
            return dict(base, skipped=(
                f"a side is too small ({ln} / {rn}, need {floor})" + (
                    "; the registry names " + ", ".join(gone)
                    + " but the library holds no such document" if gone else "")))

        hi, lo = cfg.asymmetry_min_rate, cfg.asymmetry_max_other
        findings = []
        for terms, a_n, b, b_n, side, a_lab, b_lab in (
            (left, ln, right, rn, "left", pair["left_label"], pair["right_label"]),
            (right, rn, left, ln, "right", pair["right_label"], pair["left_label"]),
        ):
            for term, k in terms.items():
                rate = k / a_n
                if rate < hi:
                    continue
                other = b.get(term, 0) / b_n
                if other > lo:
                    continue
                findings.append({
                    "term": term, "side": side,
                    "present_in": a_lab, "absent_from": b_lab,
                    "rate": round(rate, 4), "other_rate": round(other, 4),
                    "chunks": k,
                })
        findings.sort(key=lambda f: -f["rate"])
        return dict(base, findings=findings)

    @staticmethod
    def stamp(result: dict) -> dict:
        """What the answer was conditional on, carried on every finding.

        A rate means nothing except against a named population, and these
        populations move.  The day this was written the primary text was
        four of the library's eight Zohar files, and not one of the 120
        standing observations said which.
        """
        left, right = result["left"], result["right"]
        return {
            "left_documents": list(left.documents),
            "right_documents": list(right.documents),
            "left_chunks": result["left_chunks"],
            "right_chunks": result["right_chunks"],
            "duplicates_excluded": sorted(
                set(left.duplicates) | set(right.duplicates)),
            "unresolved_registry_entries": sorted(
                set(left.missing) | set(right.missing)),
            "basis": f"{left.basis}:{right.basis}",
        }

    def _withdraw_stale(self, pair: str, basis: str) -> int:
        """Retract findings measured under a different definition.

        Not the lap test the audit already applies — that catches a
        finding the walk stopped reporting, and these are findings it
        would report again with different numbers.  A row carrying no
        stamp predates the registry and is withdrawn on the same ground:
        it was computed over a population nobody wrote down.
        """
        stale: list[tuple[str, int]] = []
        for row in self._db.execute(
            "SELECT id, data FROM patrol_observations "
            "WHERE kind = 'corpus_asymmetry' AND retracted = 0 "
            "AND subject LIKE ?", (f"{pair}:%",)
        ):
            try:
                data = json.loads(row["data"] or "{}")
            except (ValueError, TypeError):
                data = {}
            was = data.get("basis")
            if was != basis:
                stale.append((
                    f"the populations it compared have changed "
                    f"(measured under {was}, now {basis})" if was else
                    "it predates any record of which documents were compared",
                    row["id"]))
        for why, obs_id in stale:
            self._db.execute(
                "UPDATE patrol_observations SET retracted = 1, "
                "detail = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') "
                "WHERE id = ?", (f"withdrawn: {why}", obs_id))
        return len(stale)

    def run(self, patrol=None, lap: int = 0) -> dict:
        """Every declared pair, recorded as observations."""
        out = []
        for pair in self._config.asymmetry_pairs:
            try:
                result = self.compare(pair)
                stamp = self.stamp(result)
            except Exception as exc:            # a bad registry entry is the
                _log.warning("asymmetry pair %r failed: %s",  # author's, not
                             pair.get("name"), exc)           # a reason to
                continue                                      # stop the walk
            withdrawn = 0
            if patrol is not None:
                withdrawn = self._withdraw_stale(result["pair"], stamp["basis"])
            kept = result["findings"][: self._config.asymmetry_max_per_pair]
            if patrol is not None:
                for f in kept:
                    side_n = (stamp["left_chunks"] if f["side"] == "left"
                              else stamp["right_chunks"])
                    patrol.observe(
                        errand="corpus_asymmetry",
                        kind="corpus_asymmetry",
                        subject=f"{result['pair']}:{f['term']}",
                        detail=(
                            f"{f['term']!r} is in {f['rate'] * 100:.1f}% of "
                            f"{f['present_in']} ({side_n:,} chunks) and "
                            f"{f['other_rate'] * 100:.2f}% of {f['absent_from']}"
                            f" — two accounts of one thing that do not use the "
                            f"same word for it"
                        ),
                        weight=round(f["other_rate"] / f["rate"], 6),
                        data={**f, **stamp},
                        lap=lap,
                    )
                self._db.connection.commit()
            row = {"pair": result["pair"], "withdrawn": withdrawn, **stamp}
            if result.get("skipped"):
                out.append({**row, "skipped": result["skipped"]})
            else:
                out.append({**row, "recorded": len(kept),
                            "found": len(result["findings"]),
                            "top": kept[:6]})
        return {"pairs": out}


class TagContentAsymmetry:
    """What a tag's material is about, against what the tag claims.

    The other comparison here diffs two POPULATIONS.  This one diffs two
    REPRESENTATIONS of the same memories: the words they use, and the tag
    they were filed under.

    A lexical diff of content against tag NAME would be useless — an
    abstraction like `mysticism` never appears in the material it labels,
    and would look like a mismatch every time.  What separates a real
    subject from a label is whether its memories share vocabulary that the
    library at large does not.  `plato` yields socrates, glaucon,
    republic; `biosemiotics` yields pattee.  `whirly-art` and `essay`,
    1,552 memories each, yield nothing at all — the index claims a subject
    where the content has none.

    Walks tags a batch at a time rather than sweeping, because profiling
    every tag's memories at once is millions of chunk reads.  The baseline
    is sampled once per batch: a rate estimated from a few thousand chunks
    is precise enough to say whether a term is distinctive.
    """

    def __init__(self, config, db) -> None:
        self._config = config
        self._db = db
        self._baseline: tuple[Counter, int] | None = None

    def baseline(self) -> tuple[Counter, int]:
        """Term rates across the library, sampled.

        ORDER BY id rather than RANDOM(): a random sort over six figures
        of rows costs more than the profiling does, and the sample only
        needs to be representative of VOCABULARY, which a spread of ids is.
        """
        if self._baseline is None:
            from chicory.layer4.wander import _CONCEPT_STOPWORDS
            n = self._config.tag_content_baseline_sample
            counts: Counter = Counter()
            seen = 0
            for row in self._db.execute(
                "SELECT content FROM memories WHERE is_archived = 0 "
                "AND id IN (SELECT id FROM memories WHERE is_archived = 0 "
                f"ORDER BY id LIMIT {int(n)} OFFSET (SELECT MAX(0, "
                f"(COUNT(*) - {int(n)}) / 2) FROM memories))"
            ):
                seen += 1
                for t in {x.lower() for x in _TERM_RE.findall(row["content"] or "")}:
                    if t not in _CONCEPT_STOPWORDS:
                        counts[t] += 1
            self._baseline = (counts, max(seen, 1))
        return self._baseline

    def examine(self, tag_id: int, name: str) -> dict | None:
        """Whether this tag's memories share vocabulary the library lacks."""
        from chicory.layer4.wander import _CONCEPT_STOPWORDS

        cfg = self._config
        base, bn = self.baseline()
        rows = self._db.execute(
            "SELECT m.content FROM memories m "
            "JOIN memory_tags mt ON mt.memory_id = m.id "
            "WHERE mt.tag_id = ? AND m.is_archived = 0 LIMIT ?",
            (tag_id, cfg.tag_content_sample),
        ).fetchall()
        if len(rows) < cfg.tag_content_min_memories:
            return None
        counts: Counter = Counter()
        for r in rows:
            for t in {x.lower() for x in _TERM_RE.findall(r["content"] or "")}:
                if t not in _CONCEPT_STOPWORDS:
                    counts[t] += 1
        n = len(rows)
        distinctive = []
        for term, k in counts.items():
            rate = k / n
            if rate < cfg.tag_content_min_rate:
                continue
            b = base.get(term, 0) / bn
            if b * cfg.tag_content_lift <= rate:
                distinctive.append({"term": term, "rate": round(rate, 4),
                                    "baseline": round(b, 4)})
        distinctive.sort(key=lambda d: -d["rate"])
        return {"tag": name, "tag_id": tag_id, "sampled": n,
                "distinctive": distinctive[:12],
                # Does the tag name itself turn up in what its material is
                # about?  Absent is not a fault — `mysticism` is a real
                # subject nobody writes the word for — but present is a
                # strong sign the label was earned rather than assigned.
                "names_itself": any(d["term"] == name.lower()
                                    for d in distinctive)}
