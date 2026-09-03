"""Proposing meta-shelves: assembling a subject out of what cannot be found.

The gap net records concepts the library discusses and holds no tag for.
Singly each is a nuisance.  Grouped they are a SUBJECT — and grouping them
is the hard part, because the obvious measure cannot do it.

Direct co-occurrence Jaccard fails for an arithmetic reason rather than a
tuning one.  Jaccard is bounded above by ``min(m_a, m_b) / max(m_a, m_b)``,
and the sefirot span 66 to 3,948 mentions, so 17 of their 78 pairs cannot
reach the 0.10 edge threshold however perfectly they correlate.  Measured
on the live library, that shattered one subject into six fragments and four
isolates: ``chochmah``-``binah``, ``tiferet``-``chesed``-``gvurah``-
``netzach``, ``anpin``-``malchut``, and ``shechinah``, ``keter``, ``yesod``
and ``gevurah`` alone.

Two second-order signals do work, and one edge case needs a third:

  arc  cosine over the `around` tag counters — the tags carried by the
       memories that discuss a concept.  "Do these live in the same
       material."  Measured 7x separation against a null.
  sc   cosine over the `co` counters — which other concepts share a
       memory with this one.  Scale-free, so immune to the ceiling above.
       Sefirot mean 0.436 against a null of 0.0042.
  ortho  `gevurah` and `gvurah` share literally ZERO memories.  No
       co-occurrence signal of any strength reaches them; only the
       spelling does, at difflib ratio 0.923.

None of this decides anything.  Every group is a PROPOSAL, because the junk
filter provably cannot finish the job: a League client config scores docJ
0.948 and one lab's cellulose vocabulary 0.785.  Both really are a single
source's vocabulary — the filter is right and the shelf is still unwanted.
A person confirms; that is not ceremony, it is the only working
discriminator.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Iterable

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine

_log = logging.getLogger(__name__)

_STRUCTURAL_TAG_PREFIXES = ("month-", "day-", "minute-", "hour-",
                            "week-", "year-")

# Citation context: a surname followed by initials, an "et al", or a pair
# of years close together.  These are how a bibliography looks from inside
# a chunk, and they are what separates a subject's vocabulary from the
# names of the people who wrote about it.
_CITE_INITIALS = re.compile(r"^\s*,?\s*[A-Z]\.\s*[A-Z]?\.?")
_CITE_ETAL = re.compile(r"^\s*,?\s*et\s+al", re.IGNORECASE)
_YEAR = re.compile(r"\b(1[89]\d{2}|20[0-2]\d)\b")


def _cosine(a: dict, b: dict) -> float:
    """Cosine over two sparse counters."""
    if not a or not b:
        return 0.0
    shared = a.keys() & b.keys()
    if not shared:
        return 0.0
    num = sum(a[k] * b[k] for k in shared)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return num / (na * nb) if na and nb else 0.0


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


class MetaShelfProposer:
    """Assembles candidate subjects from the gap net and files them."""

    def __init__(self, config: ChicoryConfig, db: DatabaseEngine) -> None:
        self._config = config
        self._db = db

    # ── loading ─────────────────────────────────────────────────────────

    def _hub_cap(self) -> float:
        row = self._db.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE is_archived = 0"
        ).fetchone()
        live = max(row["n"] if row else 0, 1)
        return self._config.subject_hub_tag_max_fraction * live

    def _topical_tag_names(self) -> set[str]:
        """The roster's own eligibility rule, reused deliberately.

        Unfiltered, every component's top three `around` tags are
        `document`, `pdf` and a month stamp, so a similarity computed over
        them measures which import batch two concepts arrived in.  Reusing
        _topical_tag_ids' rule keeps the joining signal measuring the same
        thing the shelves are made of.
        """
        cap = self._hub_cap()
        names = set()
        for r in self._db.execute(
            "SELECT t.name, tc.memory_count FROM tag_centroids tc "
            "JOIN tags t ON t.id = tc.tag_id WHERE t.is_active = 1"
        ):
            name = r["name"]
            if len(name) > 1 and not name.startswith(_STRUCTURAL_TAG_PREFIXES) \
                    and (r["memory_count"] or 0) <= cap:
                names.add(name)
        return names

    def _load_nodes(self, min_confirmations: int = 2) -> dict[str, dict]:
        """Gap observations rich enough to be joined to something."""
        topical = self._topical_tag_names()
        nodes: dict[str, dict] = {}
        for r in self._db.execute(
            "SELECT subject, weight, data FROM patrol_observations "
            "WHERE kind = 'concept_gap' AND retracted = 0 "
            "AND confirmations >= ?", (min_confirmations,),
        ):
            try:
                data = json.loads(r["data"] or "{}")
            except (ValueError, TypeError):
                continue
            co = data.get("co") or {}
            if not co:
                continue
            around = {k: v for k, v in (data.get("around") or {}).items()
                      if k in topical}
            nodes[r["subject"]] = {
                "co": co,
                "around": around,
                "documents": set(data.get("documents") or []),
                "mentions": max(int(data.get("mentions") or 1), 1),
                "spread": r["weight"],
            }
        return nodes

    # ── candidate generation ────────────────────────────────────────────

    def _candidate_pairs(self, nodes: dict[str, dict]) -> set[tuple[str, str]]:
        """Pairs worth scoring, via an inverted index over `co` keys.

        6,159 concepts is 19 million unordered pairs; walking the index
        instead gives a few hundred thousand.  A `co` key held by very many
        concepts says nothing about any of them — it is a stopword of this
        corpus rather than of English — so it is skipped.
        """
        index: dict[str, list[str]] = {}
        for name, n in nodes.items():
            for key in n["co"]:
                index.setdefault(key, []).append(name)

        limit = self._config.meta_shelf_co_stopword_holders
        pairs: set[tuple[str, str]] = set()
        for key, holders in index.items():
            if len(holders) > limit or len(holders) < 2:
                continue
            holders = sorted(holders)
            for i in range(len(holders)):
                for j in range(i + 1, len(holders)):
                    pairs.add((holders[i], holders[j]))
        return pairs

    def _scored_edges(
        self, nodes: dict[str, dict], pairs: Iterable[tuple[str, str]],
    ) -> dict[tuple[str, str], float]:
        arc_min = self._config.meta_shelf_arc_min
        sc_min = self._config.meta_shelf_sc_min
        out: dict[tuple[str, str], float] = {}
        for a, b in pairs:
            na, nb = nodes[a], nodes[b]
            if _cosine(na["around"], nb["around"]) < arc_min:
                continue
            sc = _cosine(na["co"], nb["co"])
            if sc >= sc_min:
                out[(a, b)] = sc
        return out

    def _mutual_knn(
        self, edges: dict[tuple[str, str], float], k: int,
    ) -> set[tuple[str, str]]:
        """Keep an edge only when each end ranks the other in its top k.

        The anti-chaining device, and not optional.  Single linkage on any
        one of these signals blows up: document Jaccard at 0.2 yields a
        5,821-of-6,159 blob, and raw arc >= 0.97 yields 480 nodes, which is
        simply "the Zohar".
        """
        neighbours: dict[str, list[tuple[float, str]]] = {}
        for (a, b), sc in edges.items():
            neighbours.setdefault(a, []).append((sc, b))
            neighbours.setdefault(b, []).append((sc, a))
        top: dict[str, set[str]] = {}
        for name, lst in neighbours.items():
            lst.sort(key=lambda x: (-x[0], x[1]))
            top[name] = {n for _s, n in lst[:k]}
        return {
            (a, b) for (a, b) in edges
            if b in top.get(a, ()) and a in top.get(b, ())
        }

    def _ortho_bridges(
        self, nodes: dict[str, dict], existing: set[tuple[str, str]],
    ) -> set[tuple[str, str]]:
        """Spelling variants, which no co-occurrence signal can reach.

        `gevurah` and `gvurah` share ZERO memories — overlap coefficient
        0.000, direct Jaccard 0.000, and sc 0.326 which is rank 40 of 48 in
        gvurah's own neighbour list, so mutual-kNN drops it.  Only the
        orthography connects them.

        Enumeration is over explicit index pairs rather than a
        lexicographic skip inside length buckets.  That distinction is not
        stylistic: `'gevurah' < 'gvurah'` is true, so a `b <= a` guard
        drops exactly the pair this method exists for, silently and with no
        error to notice.
        """
        ratio_min = self._config.meta_shelf_variant_ratio
        arc_min = self._config.meta_shelf_arc_min
        names = sorted(nodes)
        by_len: dict[int, list[int]] = {}
        for idx, name in enumerate(names):
            by_len.setdefault(len(name), []).append(idx)

        out: set[tuple[str, str]] = set()
        matcher = SequenceMatcher(autojunk=False)
        for length, bucket in by_len.items():
            near = [i for delta in (0, 1, 2) for i in by_len.get(length + delta, ())]
            for i in bucket:
                a = names[i]
                if len(a) < 5:
                    continue
                matcher.set_seq2(a)
                for j in near:
                    if i == j:
                        continue
                    b = names[j]
                    if len(b) < 5:
                        continue
                    pair = (a, b) if a < b else (b, a)
                    if pair in out:
                        continue
                    # Deliberately NOT skipping pairs that are already scored
                    # edges.  A bridge's job is to survive mutual-kNN, and a
                    # variant pair with a weak-but-present co-occurrence score
                    # is the case that most needs it: gevurah~gvurah scores
                    # sc 0.326, which is rank 40 of 48 in gvurah's neighbour
                    # list, so kNN discards it.  Excluding it here as
                    # "already connected" loses it at the next step instead.
                    # difflib's own cheap upper bounds first: both are O(len)
                    # and exact, so nothing that would have matched is lost.
                    # Without them this is the slowest step in the pass by a
                    # wide margin — 101s against 6,159 concepts.
                    matcher.set_seq1(b)
                    if matcher.real_quick_ratio() < ratio_min \
                            or matcher.quick_ratio() < ratio_min \
                            or matcher.ratio() < ratio_min:
                        continue
                    if _cosine(nodes[a]["around"], nodes[b]["around"]) < arc_min:
                        continue
                    out.add(pair)
        return out

    # ── grouping ────────────────────────────────────────────────────────

    @staticmethod
    def _components(
        names: Iterable[str], edges: Iterable[tuple[str, str]],
    ) -> list[set[str]]:
        adj: dict[str, set[str]] = {n: set() for n in names}
        for a, b in edges:
            if a in adj and b in adj:
                adj[a].add(b)
                adj[b].add(a)
        seen: set[str] = set()
        out: list[set[str]] = []
        for start in adj:
            if start in seen:
                continue
            stack, comp = [start], set()
            seen.add(start)
            while stack:
                cur = stack.pop()
                comp.add(cur)
                for nxt in adj[cur]:
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            out.append(comp)
        return out

    def _escalating_components(
        self, nodes: dict[str, dict], edges: dict[tuple[str, str], float],
        bridges: set[tuple[str, str]],
    ) -> list[set[str]]:
        """Split a component too big to be one subject, on its own members.

        The 450-node Zohar component holds two real subjects — the Idra
        vocabulary and the Torah-portion vocabulary — and no single global
        threshold separates them, because the right cut differs per
        component.  Re-clustering inside the component at a stricter sc
        does separate them.
        """
        cap = self._config.meta_shelf_split_cap
        step = self._config.meta_shelf_sc_step
        ceiling = self._config.meta_shelf_sc_ceiling
        base = self._config.meta_shelf_sc_min

        kept = self._mutual_knn(edges, self._config.meta_shelf_mutual_k)
        out: list[set[str]] = []
        queue = [(c, base) for c in
                 self._components(nodes, kept | bridges)]
        while queue:
            comp, level = queue.pop()
            if len(comp) < cap or level >= ceiling:
                out.append(comp)
                continue
            level += step
            inner = {p: s for p, s in edges.items()
                     if s >= level and p[0] in comp and p[1] in comp}
            inner_kept = self._mutual_knn(inner, self._config.meta_shelf_mutual_k)
            inner_bridges = {p for p in bridges if p[0] in comp and p[1] in comp}
            parts = self._components(comp, inner_kept | inner_bridges)
            if len(parts) <= 1:
                # Cannot be separated at this level; try harder or accept.
                if level >= ceiling:
                    out.append(comp)
                else:
                    queue.append((comp, level))
                continue
            queue.extend((p, level) for p in parts)
        return out

    # ── junk discrimination ─────────────────────────────────────────────

    def _doc_jaccard(self, nodes: dict[str, dict], group: set[str]) -> float:
        """Mean pairwise overlap of members' document sets.

        A real vocabulary's terms share files; a citation convention's
        terms each sit in a different reference list.
        """
        members = sorted(group)
        if len(members) < 2:
            return 0.0
        total, n = 0.0, 0
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                total += _jaccard(nodes[members[i]]["documents"],
                                  nodes[members[j]]["documents"])
                n += 1
        return round(total / n, 4) if n else 0.0

    def _ref_hi(self, nodes: dict[str, dict], group: set[str]) -> tuple[float, int]:
        """Fraction of members that live mostly in citation context.

        The only rule that catches a bibliography which IS document
        concentrated — one lab where everyone cites everyone, which passes
        the document test honestly.

        The scan budget is fixed rather than proportional, because this
        measure is sampling-unstable: one component moved from 0.067 to
        0.15 on sample size alone.  A term with too few counted occurrences
        abstains instead of voting.
        """
        max_terms, max_docs, max_occ, min_occ = 25, 6, 120, 20
        members = sorted(group, key=lambda t: -nodes[t]["mentions"])[:max_terms]
        qualified, citational = 0, 0
        for term in members:
            docs = sorted(nodes[term]["documents"])[:max_docs]
            if not docs:
                continue
            placeholders = ",".join("?" * len(docs))
            rows = self._db.execute(
                f"SELECT content FROM memories WHERE is_archived = 0 "
                f"AND source_path IN ({placeholders}) LIMIT 400",
                tuple(docs),
            ).fetchall()
            occ, cite = 0, 0
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            for r in rows:
                text = r["content"] or ""
                for m in pattern.finditer(text):
                    occ += 1
                    right = text[m.end():m.end() + 24]
                    window = text[max(0, m.start() - 90):m.end() + 90]
                    if _CITE_INITIALS.match(right) or _CITE_ETAL.match(right) \
                            or len(_YEAR.findall(window)) >= 2:
                        cite += 1
                    if occ >= max_occ:
                        break
                if occ >= max_occ:
                    break
            if occ >= min_occ:
                qualified += 1
                if cite / occ >= 0.5:
                    citational += 1
        if not qualified:
            return 0.0, 0
        return round(citational / qualified, 4), qualified

    # ── the pass ────────────────────────────────────────────────────────

    def _slug_for(self, nodes: dict[str, dict], group: set[str]) -> str:
        """A stable address, insensitive to peripheral membership drift.

        Alphabetically first of the group's top five by mentions: adding or
        dropping a rare member cannot move it, and it never contains '+',
        which the schema forbids because a derived shelf name is a
        '+'-join and a collision would silently suffix the address.
        """
        core = sorted(group, key=lambda t: -nodes[t]["mentions"])[:5]
        return sorted(core)[0].replace("+", "")

    def propose(self, min_confirmations: int = 2, scan: bool = True) -> dict:
        """Assemble groups and file the accepted ones as proposals."""
        cfg = self._config
        nodes = self._load_nodes(min_confirmations)
        if len(nodes) < 2:
            return {"nodes": len(nodes), "groups": 0, "proposed": 0,
                    "note": "not enough gap observations carrying regions"}

        pairs = self._candidate_pairs(nodes)
        edges = self._scored_edges(nodes, pairs)
        bridges = self._ortho_bridges(nodes, set(edges))
        groups = self._escalating_components(nodes, edges, bridges)

        accepted, rejected = [], []
        for group in groups:
            if len(group) < cfg.meta_shelf_min_size:
                continue
            docj = self._doc_jaccard(nodes, group)
            verdict, rule = "accept", None
            if len(group) > cfg.meta_shelf_max_size:
                verdict, rule = "reject", f"too large ({len(group)})"
            elif docj < cfg.meta_shelf_min_docj:
                verdict, rule = "reject", f"scattered (docJ {docj})"
            ref_hi, qualified = (0.0, 0)
            if verdict == "accept" and scan:
                ref_hi, qualified = self._ref_hi(nodes, group)
                if ref_hi > cfg.meta_shelf_max_ref_hi:
                    verdict, rule = "reject", f"bibliography (refHi {ref_hi})"
            arcs = [nodes[t]["around"] for t in group]
            arc_mean = 0.0
            if len(arcs) > 1:
                vals = [_cosine(arcs[i], arcs[j])
                        for i in range(len(arcs))
                        for j in range(i + 1, len(arcs))]
                arc_mean = round(sum(vals) / len(vals), 4) if vals else 0.0
            record = {
                "slug": self._slug_for(nodes, group),
                "members": sorted(group, key=lambda t: -nodes[t]["mentions"]),
                "docj": docj, "ref_hi": ref_hi, "qualified": qualified,
                "arc_mean": arc_mean, "n": len(group), "rule": rule,
            }
            (accepted if verdict == "accept" else rejected).append(record)

        filed = sum(self._file(r, bridges) for r in accepted)
        return {
            "nodes": len(nodes), "candidate_pairs": len(pairs),
            "scored_edges": len(edges), "ortho_bridges": len(bridges),
            "groups": len(groups), "accepted": len(accepted),
            "rejected": len(rejected), "proposed": filed,
            "rejects": sorted(rejected, key=lambda r: -r["n"])[:20],
        }

    def _matching_proposal(self, members: list[str]):
        """An open proposal that is substantially this same group.

        The slug is derived from the group's largest members, so it is
        stable under peripheral drift but not immune to it — and a slug
        that moves files the SAME group a second time under a new name.
        Observed exactly that: the Tractatus vocabulary landed twice, once
        as `logik` and once as `eigenschaften`, 51 identical members each.

        Matching on membership rather than name is what makes the pass
        idempotent across laps.  Only open proposals are candidates; a
        confirmed charter's membership is a person's, not the walk's, and
        must not be silently rewritten by a later pass.
        """
        if not members:
            return None
        want = set(members)
        for r in self._db.execute(
            "SELECT s.id, s.status, GROUP_CONCAT(m.concept, char(10)) AS names "
            "FROM meta_shelves s JOIN meta_shelf_members m ON m.shelf_id = s.id "
            "WHERE s.status = 'proposed' GROUP BY s.id"
        ).fetchall():
            have = set((r["names"] or "").split("\n"))
            if not have:
                continue
            overlap = len(want & have) / max(len(want | have), 1)
            if overlap >= 0.60:
                return r
        return None

    def _file(self, record: dict, bridges: set[tuple[str, str]]) -> int:
        """Upsert one proposal, touching nothing a person has acted on.

        Only ``status='proposed'`` rows are writable.  A confirmed,
        materialized or retired charter is immutable to the patrol — which
        is what keeps the shelf tree a function of stored state rather than
        of whenever the walk last ran.
        """
        slug, members = record["slug"], record["members"]
        row = self._db.execute(
            "SELECT id, status FROM meta_shelves WHERE slug = ?", (slug,),
        ).fetchone()
        if row is None:
            row = self._matching_proposal(members)
        if row is not None and row["status"] != "proposed":
            return 0
        if row is not None:
            slug = self._db.execute(
                "SELECT slug FROM meta_shelves WHERE id = ?", (row["id"],),
            ).fetchone()["slug"]

        seed = json.dumps({"members": members}, sort_keys=True)
        if row is None:
            self._db.execute(
                "INSERT INTO meta_shelves (slug, status, origin, seed_json, "
                "docj, ref_hi, arc_mean, n_core) "
                "VALUES (?, 'proposed', 'patrol', ?, ?, ?, ?, ?)",
                (slug, seed, record["docj"], record["ref_hi"],
                 record["arc_mean"], record["n"]),
            )
            shelf_id = self._db.execute(
                "SELECT id FROM meta_shelves WHERE slug = ?", (slug,),
            ).fetchone()["id"]
        else:
            shelf_id = row["id"]
            self._db.execute(
                "UPDATE meta_shelves SET seed_json = ?, docj = ?, ref_hi = ?, "
                "arc_mean = ?, n_core = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') WHERE id = ?",
                (seed, record["docj"], record["ref_hi"], record["arc_mean"],
                 record["n"], shelf_id),
            )
            self._db.execute(
                "DELETE FROM meta_shelf_members WHERE shelf_id = ? "
                "AND tag_id IS NULL", (shelf_id,),
            )

        variants = {a: b for a, b in bridges} | {b: a for a, b in bridges}
        self._db.executemany(
            "INSERT OR IGNORE INTO meta_shelf_members "
            "(shelf_id, concept, role, variant_of, added_by) "
            "VALUES (?, ?, ?, ?, 'patrol')",
            [(shelf_id, m,
              "variant" if m in variants and variants[m] in members else "core",
              variants.get(m) if variants.get(m) in members else None)
             for m in members],
        )
        self._db.connection.commit()
        return 1
