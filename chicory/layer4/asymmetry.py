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
"""

from __future__ import annotations

import logging
import re
from collections import Counter

_log = logging.getLogger(__name__)

# The same capitalisation heuristic the concept walk uses, so a term found
# here is a term found there and the two are comparable.
_TERM_RE = re.compile(r"\b([A-Z][a-zA-Z]{3,})\b")


class CorpusAsymmetry:
    """Diffs the vocabulary of two populations that describe one thing."""

    def __init__(self, config, db) -> None:
        self._config = config
        self._db = db

    def _profile(self, predicate: str) -> tuple[Counter, int]:
        """How often each capitalised term appears, by CHUNK not by mention.

        Per chunk because the two sides are usually different sizes — the
        scholarship is 546 chunks against 15,508 of primary text — and a
        raw count would say nothing but "one side is bigger".
        """
        from chicory.layer4.wander import _CONCEPT_STOPWORDS

        counts: Counter = Counter()
        chunks = 0
        for row in self._db.execute(
            f"SELECT content FROM memories WHERE is_archived = 0 "
            f"AND ({predicate})"
        ):
            text = row["content"] or ""
            chunks += 1
            for term in {t.lower() for t in _TERM_RE.findall(text)}:
                if term not in _CONCEPT_STOPWORDS:
                    counts[term] += 1
        return counts, chunks

    def compare(self, pair: dict) -> dict:
        """One declared pair, diffed both ways."""
        cfg = self._config
        left, ln = self._profile(pair["left"])
        right, rn = self._profile(pair["right"])
        floor = cfg.asymmetry_min_population
        if ln < floor or rn < floor:
            return {"pair": pair["name"], "skipped":
                    f"a side is too small ({ln} / {rn}, need {floor})"}

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
        return {"pair": pair["name"], "left_chunks": ln, "right_chunks": rn,
                "findings": findings}

    def run(self, patrol=None, lap: int = 0) -> dict:
        """Every declared pair, recorded as observations."""
        out = []
        for pair in self._config.asymmetry_pairs:
            try:
                result = self.compare(pair)
            except Exception as exc:            # a bad predicate is the
                _log.warning("asymmetry pair %r failed: %s",  # author's, not
                             pair.get("name"), exc)           # a reason to
                continue                                      # stop the walk
            if result.get("skipped"):
                out.append(result)
                continue
            kept = result["findings"][: self._config.asymmetry_max_per_pair]
            if patrol is not None:
                for f in kept:
                    patrol.observe(
                        errand="corpus_asymmetry",
                        kind="corpus_asymmetry",
                        subject=f"{result['pair']}:{f['term']}",
                        detail=(
                            f"{f['term']!r} is in {f['rate'] * 100:.1f}% of "
                            f"{f['present_in']} and {f['other_rate'] * 100:.2f}% "
                            f"of {f['absent_from']} — two accounts of one thing "
                            f"that do not use the same word for it"
                        ),
                        weight=round(f["other_rate"] / f["rate"], 6),
                        data=f,
                        lap=lap,
                    )
                self._db.connection.commit()
            out.append({"pair": result["pair"], "recorded": len(kept),
                        "found": len(result["findings"]),
                        "left_chunks": result["left_chunks"],
                        "right_chunks": result["right_chunks"],
                        "top": kept[:6]})
        return {"pairs": out}
