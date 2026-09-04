"""Librarian: write gate, quarantine, linear sweep, and subject roster.

The librarian is the custody layer over the memory store:

- WriteGate scores an incoming memory's embedding against tag centroids
  (its own claimed tags first, all known centroids as fallback) and
  diverts off-topic or unclassifiable writes to the 'quarantine'
  ingestion tier instead of rejecting them.  In shadow mode it only
  logs what it would have done — every verdict lands in write_gate_log
  so the threshold can be calibrated before enforcement.
- LinearSweep is a cursor-driven patrol over all memories (shelf-reading):
  each pass visits the least-recently-swept batch, runs mechanical checks
  (missing embeddings, near-duplicates, tag drift), and journals findings
  to sweep_findings.  Coverage is durable — the cursor is the swept_at
  column, so a restart never loses patrol progress.
- SubjectRoster derives topical clusters from the tag co-retrieval graph
  (centroid_edges) and persists them as named subjects with centroids.
  Subjects that stop being derivable are retired, never deleted.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine

_log = logging.getLogger(__name__)

# Tag-name prefixes that never describe topical content
_NON_TOPICAL_PREFIXES = ("month-", "day-", "minute-")


def _array_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _blob_to_array(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# Custody states a memory can hold; transitions are owned (gate/sweep/judge)
# and journaled to custody_events — status changes are never silent.
CUSTODY_STATES = ("unswept", "clean", "flagged", "quarantined", "certified")


def actor_allowed(actor: str, allow: list[str]) -> bool:
    """Whether this actor is on an allow-list.

    Two forms.  A plain name matches exactly.  A name ending in a colon
    matches the whole namespace beneath it, so "specialist:" covers
    specialist:zohar and every other shift without enumerating subjects —
    which matters because subjects are derived and renameable, so a list
    of them would go stale the next time the roster rebuilt.

    An empty list means "no filter", which is how a caller opts out of
    attribution entirely.
    """
    if not allow:
        return True
    for entry in allow:
        if entry.endswith(":"):
            if actor.startswith(entry):
                return True
        elif actor == entry:
            return True
    return False


def record_custody_transition(
    db: DatabaseEngine,
    memory_id: str,
    to_status: str,
    actor: str,
    reason: str = "",
) -> bool:
    """Move a memory to a custody status, journaling the transition.

    No-ops (and writes no event) when the status is unchanged.
    Returns True if a transition occurred.
    """
    row = db.execute(
        "SELECT custody_status FROM memories WHERE id = ?", (memory_id,)
    ).fetchone()
    if row is None:
        return False
    from_status = row["custody_status"]
    if from_status == to_status:
        return False
    db.execute(
        "UPDATE memories SET custody_status = ? WHERE id = ?",
        (to_status, memory_id),
    )
    db.execute(
        "INSERT INTO custody_events (memory_id, from_status, to_status, actor, reason) "
        "VALUES (?, ?, ?, ?, ?)",
        (memory_id, from_status, to_status, actor, reason),
    )
    return True


@dataclass
class GateVerdict:
    verdict: str  # 'pass', 'quarantine', 'shadow_quarantine', 'cold_start'
    score: float
    best_tag_id: int | None
    enforced: bool

    @property
    def diverted(self) -> bool:
        return self.verdict == "quarantine" and self.enforced


class WriteGate:
    """On-topic check for incoming memories against tag centroids."""

    def __init__(
        self,
        config: ChicoryConfig,
        db: DatabaseEngine,
        tag_manager,
        centroid_subgraph,
    ) -> None:
        self._config = config
        self._db = db
        self._tags = tag_manager
        self._centroids = centroid_subgraph
        # (version_key, tag_ids, matrix) — refreshed when centroids change
        # (count or max updated_at), since EMA updates shift values in place
        self._matrix_cache: tuple[tuple, list[int], np.ndarray] | None = None

    def evaluate_content(
        self, embeddings: list[np.ndarray], tag_names: list[str],
    ) -> GateVerdict:
        """Gate a whole memory: every chunk must belong somewhere.

        The verdict is the worst chunk's verdict, so a payload can't ride
        in on the strength of its first paragraph.
        """
        worst: GateVerdict | None = None
        for vec in embeddings:
            v = self.evaluate(vec, tag_names)
            if worst is None or v.score < worst.score:
                worst = v
        return worst if worst is not None else GateVerdict(
            "cold_start", 1.0, None, self._config.librarian_gate_enforce,
        )

    def evaluate(self, embedding: np.ndarray, tag_names: list[str]) -> GateVerdict:
        """Score an incoming embedding; decide pass/quarantine.

        The score is the best cosine against the claimed tags' centroids,
        falling back to the best match across all centroids — content that
        belongs *somewhere* passes; only content that belongs nowhere is
        below threshold.  Mis-shelving is the sweep's job, not the gate's.
        """
        enforce = self._config.librarian_gate_enforce

        version_row = self._db.execute(
            "SELECT COUNT(*) AS n, MAX(updated_at) AS u FROM tag_centroids"
        ).fetchone()
        n_centroids = version_row["n"] if version_row else 0
        version_key = (n_centroids, version_row["u"] if version_row else None)
        if n_centroids < self._config.librarian_gate_min_centroids:
            return GateVerdict("cold_start", 1.0, None, enforce)

        vec = _unit(embedding.astype(np.float32))

        best_score = -1.0
        best_tag_id: int | None = None

        claimed_ids = []
        for name in tag_names:
            tag = self._tags.get_by_name(name)
            if tag is not None:
                claimed_ids.append(tag.id)
        if claimed_ids:
            centroids = self._centroids.get_centroids_batch(claimed_ids)
            for tid, cvec in centroids.items():
                s = float(np.dot(vec, cvec))
                if s > best_score:
                    best_score, best_tag_id = s, tid

        threshold = self._config.librarian_gate_threshold
        if best_score < threshold:
            global_tid, global_score = self._best_global_match(vec, version_key)
            if global_score > best_score:
                best_score, best_tag_id = global_score, global_tid

        if best_score >= threshold:
            return GateVerdict("pass", best_score, best_tag_id, enforce)
        verdict = "quarantine" if enforce else "shadow_quarantine"
        return GateVerdict(verdict, max(best_score, 0.0), best_tag_id, enforce)

    def log(self, memory_id: str, verdict: GateVerdict) -> None:
        self._db.execute(
            "INSERT INTO write_gate_log (memory_id, best_tag_id, score, verdict, enforced) "
            "VALUES (?, ?, ?, ?, ?)",
            (memory_id, verdict.best_tag_id, verdict.score,
             verdict.verdict, 1 if verdict.enforced else 0),
        )

    def invalidate_cache(self) -> None:
        self._matrix_cache = None

    def _best_global_match(
        self, unit_vec: np.ndarray, version_key: tuple,
    ) -> tuple[int | None, float]:
        if self._matrix_cache is None or self._matrix_cache[0] != version_key:
            rows = self._db.execute(
                "SELECT tag_id, centroid FROM tag_centroids"
            ).fetchall()
            if not rows:
                return None, -1.0
            tag_ids = [r["tag_id"] for r in rows]
            matrix = np.stack([_blob_to_array(r["centroid"]) for r in rows])
            self._matrix_cache = (version_key, tag_ids, matrix)
        _, tag_ids, matrix = self._matrix_cache
        sims = matrix @ unit_vec
        idx = int(np.argmax(sims))
        return tag_ids[idx], float(sims[idx])


class LinearSweep:
    """Cursor-driven patrol: every memory periodically re-read by checks."""

    def __init__(
        self,
        config: ChicoryConfig,
        db: DatabaseEngine,
        embedding_engine,
        tag_manager,
        centroid_subgraph,
        penumbra=None,
    ) -> None:
        self._config = config
        self._db = db
        self._embedding = embedding_engine
        self._tags = tag_manager
        self._centroids = centroid_subgraph
        self._penumbra = penumbra

    def run_batch(self, batch_size: int | None = None) -> dict:
        """Sweep the least-recently-visited batch of memories.

        Ordering puts never-swept memories first, then oldest sweep
        timestamps — so coverage is guaranteed and restart-safe.
        """
        size = batch_size or self._config.librarian_sweep_batch_size
        # SQLite sorts NULLs first in ASC order, so never-swept memories
        # lead naturally and idx_memories_swept_at can serve the sort.
        rows = self._db.execute(
            """
            SELECT id, ingestion_tier, custody_status FROM memories
            WHERE is_archived = 0
            ORDER BY swept_at ASC, created_at ASC
            LIMIT ?
            """,
            (size,),
        ).fetchall()
        if not rows:
            return {"swept": 0, "findings": {}}

        memory_ids = [r["id"] for r in rows]
        tier_map = {r["id"]: r["ingestion_tier"] for r in rows}
        status_map = {r["id"]: r["custody_status"] for r in rows}
        findings: dict[str, int] = {}

        placeholders = ",".join("?" * len(memory_ids))
        emb_rows = self._db.execute(
            f"SELECT memory_id, embedding FROM embeddings "
            f"WHERE memory_id IN ({placeholders}) AND chunk_index = 0",
            tuple(memory_ids),
        ).fetchall()
        emb_map = {r["memory_id"]: _blob_to_array(r["embedding"]) for r in emb_rows}

        tags_map = self._tags.get_tags_for_memories(memory_ids)

        new_findings: list[tuple[str, str, str]] = []
        flag_now: set[str] = set()      # batch members implicated this run
        offbatch_flags: set[str] = set()  # near-dup partners outside the batch
        resolutions: list[tuple[str, str]] = []  # checks that conclusively passed
        allow_near_dup = self._has_live_index()
        batch_set = set(memory_ids)

        for mid in memory_ids:
            if mid not in emb_map:
                new_findings.append((mid, "missing_embedding", ""))
                flag_now.add(mid)
                continue
            resolutions.append((mid, "missing_embedding"))
            vec = _unit(emb_map[mid])

            if allow_near_dup:
                near = self._near_duplicate(mid, vec)
                if near is not None:
                    # File under the lexically-first id with a stable pair
                    # key, so A~B and B~A collapse onto a single row — but
                    # flag BOTH memories, including an out-of-batch partner
                    pair = "|".join(sorted((mid, near)))
                    key = min(mid, near)
                    new_findings.append((key, "near_duplicate", pair))
                    flag_now.add(mid)
                    if key not in batch_set:
                        offbatch_flags.add(key)
                else:
                    resolutions.append((mid, "near_duplicate"))

            drift = self._tag_drift(vec, tags_map.get(mid, []))
            if drift == "drift":
                new_findings.append((mid, "tag_drift", "below_threshold"))
                flag_now.add(mid)
            elif drift == "ok":
                resolutions.append((mid, "tag_drift"))

        # Penumbra: chronically almost-retrieved memories become 'buried'
        # findings — rescue candidates, NOT custody flags (buried is a
        # service problem, not an integrity problem, so it never demotes
        # a memory to 'flagged').
        if self._penumbra is not None:
            for mid in self._penumbra.buried_candidates(memory_ids):
                # Quarantined memories are held out deliberately — being
                # unservable is not burial
                if tier_map.get(mid) != "quarantine":
                    new_findings.append((mid, "buried", ""))
            for mid in self._penumbra.rescued(memory_ids):
                resolutions.append((mid, "buried"))

        # Auto-resolve findings whose mechanical check now passes — reality
        # outranks stale bookkeeping.  Runs BEFORE the inserts so a finding
        # re-detected this same pass ends up open, not resolved.
        if resolutions:
            self._db.executemany(
                "UPDATE sweep_findings SET resolved = 1 "
                "WHERE memory_id = ? AND finding_type = ? AND resolved = 0",
                resolutions,
            )
        if new_findings:
            # Re-detection reopens a previously resolved finding: if the
            # check fails again after judgment, the condition is real again.
            self._db.executemany(
                "INSERT INTO sweep_findings (memory_id, finding_type, detail) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(memory_id, finding_type, detail) "
                "DO UPDATE SET resolved = 0",
                new_findings,
            )
            for _, ftype, _ in new_findings:
                findings[ftype] = findings.get(ftype, 0) + 1

        now = datetime.utcnow().isoformat()
        self._db.executemany(
            "UPDATE memories SET swept_at = ? WHERE id = ?",
            [(now, mid) for mid in memory_ids],
        )

        # Custody transitions: flagged if open findings exist, clean
        # otherwise.  The sweep never touches quarantined memories —
        # that status is the gate's and the judge's to change.
        open_rows = self._db.execute(
            f"SELECT DISTINCT memory_id FROM sweep_findings "
            f"WHERE resolved = 0 AND finding_type != 'buried' "
            f"AND memory_id IN ({placeholders})",
            tuple(memory_ids),
        ).fetchall()
        has_open = {r["memory_id"] for r in open_rows}
        for mid in memory_ids:
            if tier_map.get(mid) == "quarantine" or status_map.get(mid) == "quarantined":
                continue
            implicated = mid in has_open or mid in flag_now
            # A clean pass never strips certification — only new evidence
            # (open findings) can demote certified back to flagged.
            if status_map.get(mid) == "certified" and not implicated:
                continue
            target = "flagged" if implicated else "clean"
            record_custody_transition(
                self._db, mid, target, actor="sweep",
                reason="open findings" if implicated else "clean pass",
            )

        # Out-of-batch near-dup partners carry the finding row — flag them
        # now rather than waiting a full patrol cycle for their next visit.
        for pid in offbatch_flags:
            prow = self._db.execute(
                "SELECT ingestion_tier, custody_status FROM memories WHERE id = ?",
                (pid,),
            ).fetchone()
            if (
                prow
                and prow["ingestion_tier"] != "quarantine"
                and prow["custody_status"] != "quarantined"
            ):
                record_custody_transition(
                    self._db, pid, "flagged", actor="sweep",
                    reason="near-duplicate partner",
                )

        return {"swept": len(memory_ids), "findings": findings}

    def progress(self) -> dict:
        row = self._db.execute(
            """
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN swept_at IS NOT NULL THEN 1 ELSE 0 END) AS swept
            FROM memories WHERE is_archived = 0
            """
        ).fetchone()
        open_rows = self._db.execute(
            "SELECT finding_type, COUNT(*) AS n FROM sweep_findings "
            "WHERE resolved = 0 GROUP BY finding_type"
        ).fetchall()
        return {
            "total_memories": row["total"] or 0,
            "swept_memories": row["swept"] or 0,
            "open_findings": {r["finding_type"]: r["n"] for r in open_rows},
        }

    def _has_live_index(self) -> bool:
        """Only run near-dup checks against an already-built vector index.

        search_similar lazily rebuilds the FAISS index, and this runs in a
        background phase holding the foreground DB lock — a full rebuild
        there would stall foreground retrievals.  Engines without a lazy
        index (e.g. brute-force test mocks) are always allowed.
        """
        marker = object()
        vi = getattr(self._embedding, "_vector_index", marker)
        if vi is marker:
            return True
        return vi is not None and getattr(vi, "is_built", False)

    def _near_duplicate(self, memory_id: str, unit_vec: np.ndarray) -> str | None:
        threshold = self._config.librarian_sweep_near_dup_threshold
        try:
            hits = self._embedding.search_similar(
                unit_vec, top_k=2, threshold=threshold,
            )
        except Exception:
            return None
        for mid, score in hits:
            if mid != memory_id and score >= threshold:
                return mid
        return None

    def _tag_drift(self, unit_vec: np.ndarray, tag_names: list[str]) -> str:
        """A memory whose embedding matches none of its own tags' centroids
        is mis-shelved — flag it for review rather than moving it.

        Returns 'drift' (flag), 'ok' (conclusive pass — safe to auto-resolve
        an earlier drift finding), or 'skip' (inconclusive: no topical tags
        or no centroids to check against).
        """
        topical = [
            t for t in tag_names
            if len(t) > 1 and not t.startswith(_NON_TOPICAL_PREFIXES)
        ]
        if not topical:
            return "skip"
        tag_ids = []
        for name in topical:
            tag = self._tags.get_by_name(name)
            if tag is not None:
                tag_ids.append(tag.id)
        centroids = self._centroids.get_centroids_batch(tag_ids)
        if not centroids:
            return "skip"
        best = max(float(np.dot(unit_vec, c)) for c in centroids.values())
        if best < self._config.librarian_sweep_drift_threshold:
            return "drift"
        return "ok"


class SubjectRoster:
    """Derive named topical clusters (subjects) from the co-retrieval graph."""

    def __init__(
        self,
        config: ChicoryConfig,
        db: DatabaseEngine,
        centroid_subgraph,
    ) -> None:
        self._config = config
        self._db = db
        self._centroids = centroid_subgraph

    def rebuild(self, cold: bool = False) -> dict:
        """Recompute the roster as a TREE from the co-retrieval graph.

        Warm by default: both the component hysteresis and the split
        seeding inherit from the standing roster, so shelf paths hold
        still across rebuilds and stay usable as addresses.  ``cold=True``
        discards that inheritance and re-derives the tree purely from the
        geometry — the escape hatch for a roster that has drifted somewhere
        it should not have, and the only way to adopt a better partition
        the warm path would never find on its own.

        Connected components give the roots; any node too large to be
        taught is split in centroid space and its children persisted
        beneath it, recursively, to subject_max_depth.  Internal nodes
        are kept, not replaced by their children: a generalist works the
        parent, a sub-specialist works a child.

        "Too large" is judged on MEMORY share as well as tag count — a
        99-tag subject can hold three quarters of the library and never
        trip a tag-fraction cap.

        Subjects are a derived view — safe to rebuild at any time.  Nodes
        whose cluster no longer derives are retired, not deleted.
        """
        topical_ids = self._topical_tag_ids()
        if not topical_ids:
            # Nothing derives right now (no centroids yet, or all tags are
            # hubs).  Leave the existing roster alone rather than reporting
            # an empty one — it is still active and still routable.
            current = self._db.execute(
                "SELECT COUNT(*) AS n FROM subjects WHERE status = 'active'"
            ).fetchone()
            return {
                "subjects_active": current["n"] if current else 0,
                "subjects_retired": 0,
                "note": "no eligible tags; roster left unchanged",
            }

        total_row = self._db.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE is_archived = 0"
        ).fetchone()
        total_memories = max(total_row["n"] or 0, 1)

        min_size = self._config.subject_min_cluster_size
        # A chartered shelf is WITHHELD from component-finding rather than
        # exempted from retirement.  Its tags are reserved before the walk,
        # so the sweep at the end of this method stays the plain rule it has
        # always been — not derived this run, therefore retired.
        #
        # Reserving rather than un-writing the edges is deliberate: a sefirot
        # tag gets an edge at strength exactly 1.0 to `mysticism` and
        # `kabbalah` from _seed_tag_edges, both of which sit inside the
        # 1,021-tag giant, so absorption is otherwise structurally certain.
        # Deleting those edges would make the tag an island for retrieval,
        # the canopy and centroid reweighting too.  This way the edges exist
        # everywhere else and only the roster's component pass declines to
        # walk through them.
        charters, charter_notes = self._charters(topical_ids, min_size)
        reserved: set[int] = set()
        for _display, tags in charters.values():
            reserved |= tags
        roots = [
            c for c in self._connected_components(topical_ids - reserved, cold=cold)
            if len(c) >= min_size
        ]

        state = {
            "used_names": set(),
            "active_paths": set(),
            "eligible_tags": len(topical_ids),
            "total_memories": total_memories,
            "max_depth_seen": 0,
            "tag_members": self._load_tag_members(),
            # Snapshot before the first _persist_node write, so children are
            # seeded from the previous roster and not from this one.
            "prior_by_depth": {} if cold else self._prior_by_depth(),
        }
        # Charters first, so an authored name claims the slot before any
        # derived name can take it — the '~2' suffix then falls on the
        # derived node, which is free to move, instead of on the address a
        # charter exists to hold still.
        for slug, (display, tags) in charters.items():
            self._persist_node(tags, None, 0, "", state,
                               forced_name=slug, display_name=display)
        for cluster in roots:
            self._persist_node(cluster, None, 0, "", state)

        retired = 0
        rows = self._db.execute(
            "SELECT id, path, name FROM subjects WHERE status = 'active'"
        ).fetchall()
        for r in rows:
            if (r["path"] or r["name"]) not in state["active_paths"]:
                self._db.execute(
                    "UPDATE subjects SET status = 'retired', "
                    "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') WHERE id = ?",
                    (r["id"],),
                )
                retired += 1

        result = {
            "subjects_active": len(state["active_paths"]),
            "subjects_retired": retired,
            "max_depth": state["max_depth_seen"],
            "mode": "cold" if cold else "warm",
        }
        if charters:
            result["charters"] = {s: len(tg) for s, (_d, tg) in charters.items()}
        # Surfaced, never swallowed.  A charter whose members fall below
        # min_size stops deriving and its path retires through the ordinary
        # sweep — which would otherwise look exactly like a shelf silently
        # disappearing.
        if charter_notes:
            result["charter_warnings"] = charter_notes
        return result

    def _charters(
        self, topical_ids: set[int], min_size: int,
    ) -> tuple[dict[str, tuple[str | None, set[int]]], list[str]]:
        """Materialized charters, as slug -> (display name, member tag ids).

        Membership is the authored part and it is intersected with
        ``topical_ids`` here, so a member that has been deactivated, merged,
        or pushed over the hub cap simply stops counting — but says so, in
        the returned notes, rather than thinning the shelf in silence.

        Overlap is resolved by lowest charter id, not by refusing the run.
        Two charters claiming one tag would produce non-partitioning roots,
        and the roster is not the right place to discover that: an exception
        here would abort the whole maintain — sweep, roster and demand
        ledger together — over a data problem the desk should have caught.
        """
        notes: list[str] = []
        try:
            rows = self._db.execute(
                "SELECT s.id, s.slug, s.display_name, m.concept, m.tag_id "
                "FROM meta_shelves s JOIN meta_shelf_members m "
                "  ON m.shelf_id = s.id "
                "WHERE s.status = 'materialized' AND m.tag_id IS NOT NULL "
                "ORDER BY s.id ASC"
            ).fetchall()
        except Exception:
            return {}, notes           # pre-v44 database; nothing chartered

        claimed: dict[int, str] = {}
        grouped: dict[str, tuple[str | None, set[int]]] = {}
        for r in rows:
            slug, tag_id = r["slug"], r["tag_id"]
            owner = claimed.get(tag_id)
            if owner is not None and owner != slug:
                notes.append(
                    f"{slug}: tag {tag_id} ({r['concept']}) already claimed "
                    f"by {owner}; left with the earlier charter"
                )
                continue
            claimed[tag_id] = slug
            display, tags = grouped.setdefault(slug, (r["display_name"], set()))
            if tag_id in topical_ids:
                tags.add(tag_id)

        out: dict[str, tuple[str | None, set[int]]] = {}
        for slug, (display, tags) in grouped.items():
            if len(tags) < min_size:
                notes.append(
                    f"{slug}: only {len(tags)} of its members are eligible "
                    f"tags (needs {min_size}); shelf not derived this run"
                )
                continue
            out[slug] = (display, tags)
        return out, notes

    def _persist_node(
        self,
        cluster: set[int],
        parent_id: int | None,
        depth: int,
        parent_path: str,
        state: dict,
        forced_name: str | None = None,
        display_name: str | None = None,
    ) -> None:
        """Write one node, then recurse into its children if it is too big.

        ``forced_name`` overrides only the NAME, for a chartered meta-shelf.
        Everything else — description, centroid, memory count, routing
        eligibility — stays fully derived from the live data, so a charter
        can assert what belongs together but cannot assert what it is like.

        The name matters enough to author because it is the UPSERT conflict
        key AND the address.  A derived name is the top three member tags
        joined with '+', so it moves the instant their ranking changes:
        ``malchut+binah+yesod`` becomes a different row the moment binah's
        memory count overtakes yesod's, retiring the old row and orphaning
        its display name.  Four retired rows in the live library already
        carry display names that are now unreachable that way.
        """
        name, description, centroid, memory_count = self._describe(
            cluster, state.get("tag_members"),
        )
        if name is None:
            return
        if forced_name:
            name = forced_name

        # Names stay globally unique (the column's constraint); the path is
        # the address a specialist is dispatched to.
        if name in state["used_names"]:
            if forced_name:
                # Charters are persisted before derived roots and slugs are
                # unique, so this cannot happen from the roster's own names.
                # If it ever does, suffixing would silently move the one
                # address that is supposed to be permanent — refuse instead.
                raise ValueError(
                    f"chartered shelf {forced_name!r} collides with an "
                    f"already-persisted node; refusing to rename it"
                )
            suffix = 2
            while f"{name}~{suffix}" in state["used_names"]:
                suffix += 1
            name = f"{name}~{suffix}"
        state["used_names"].add(name)
        # The node's shape as it stood BEFORE this rebuild touched it.
        # Read here because the upsert below is about to make "prior" and
        # "current" the same row.
        was = self._db.execute(
            "SELECT id FROM subjects WHERE name = ?", (name,),
        ).fetchone()
        prior_arity = 0
        if was is not None:
            row_k = self._db.execute(
                "SELECT COUNT(*) AS n FROM subjects WHERE parent_id = ? "
                "AND status = 'active'", (was["id"],),
            ).fetchone()
            prior_arity = row_k["n"] if row_k else 0
        existed = was is not None
        path = f"{parent_path}/{name}" if parent_path else name
        state["active_paths"].add(path)
        state["max_depth_seen"] = max(state["max_depth_seen"], depth)

        self._db.execute(
            """
            INSERT INTO subjects (name, description, centroid, member_tag_ids,
                                  memory_count, status, parent_id, depth, path,
                                  display_name)
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                description = excluded.description,
                centroid = excluded.centroid,
                member_tag_ids = excluded.member_tag_ids,
                memory_count = excluded.memory_count,
                status = 'active',
                parent_id = excluded.parent_id,
                depth = excluded.depth,
                path = excluded.path,
                updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
            """,
            (name, description,
             _array_to_blob(centroid) if centroid is not None else None,
             json.dumps(sorted(cluster)), memory_count, parent_id, depth, path,
             display_name),
        )
        row = self._db.execute(
            "SELECT id FROM subjects WHERE name = ?", (name,),
        ).fetchone()
        node_id = row["id"] if row else None

        if not self._should_split(cluster, memory_count, depth, state):
            return
        # A node that already exists keeps the SHAPE it had.  Inheriting the
        # arity was not enough: the seed is drawn from a whole depth, so how
        # many groups a node's tags fall into depends on whether that node
        # has children, and the two answers sustain each other.  Measured on
        # the live roster, `life+wisdom+power` alternated 0 and 2 children on
        # every rebuild for eight consecutive runs — with children the split
        # came out [2,7,7] and stood, without them [2,14], which the
        # memory-separation test rejected, restoring the first.
        #
        # So the decision is inherited, not recomputed. A node that was a
        # leaf stays a leaf; a node that had children keeps that many. Only
        # a genuinely new node derives its own shape, and rebuild(cold=True)
        # is the way to re-derive the rest.
        if existed and prior_arity < 2:
            return
        # A node at depth d inherits the arity depth d+1 actually had.
        children = self._split_in_centroid_space(
            cluster,
            prior_leaf=(state.get("prior_by_depth") or {}).get(depth + 1),
            target_k=prior_arity,
        )
        children = self._absorb_undersized(children)
        if len(children) < 2:
            return  # geometry could not separate it; leave the node as a leaf

        # A "split" whose largest child is nearly the whole parent has not
        # separated anything — it just re-labels the parent one level down
        # (the X / X~2 / X~3 chains).  Treat the node as a leaf instead.
        #
        # Judge that on MEMORY share as well as tag share: a child can shed
        # a third of the parent's tags and still carry ~all of its material,
        # which looks like a split and isn't one.
        # Asymmetric: a node that already had children keeps them on
        # slightly weaker evidence than it would need to acquire them.
        # Equal thresholds let a marginal node flip forever, because the
        # warm seed differs depending on which side of the flip it is on.
        separation = self._config.subject_split_min_separation
        had_children = bool(
            self._db.execute(
                "SELECT 1 FROM subjects WHERE parent_id = (SELECT id FROM "
                "subjects WHERE name = ?) AND status = 'active' LIMIT 1",
                (name,),
            ).fetchone()
        )
        if had_children:
            separation += self._config.subject_split_hysteresis
        if max(len(c) for c in children) / len(cluster) >= separation:
            return
        tag_members = state.get("tag_members")
        if tag_members is not None and memory_count > 0:
            def child_memories(child: set[int]) -> int:
                seen: set[int] = set()
                for t in child:
                    seen.update(tag_members.get(t, ()))
                return len(seen)
            if max(child_memories(c) for c in children) / memory_count >= separation:
                return

        for child in children:
            self._persist_node(child, node_id, depth + 1, path, state)

    def _absorb_undersized(self, groups: list[set[int]]) -> list[set[int]]:
        """Fold groups below the minimum size into their nearest neighbour.

        Dropping them would leave their tags in no persisted child while
        still making the parent internal — and an internal node is not on
        the routing frontier, so those tags would become unreachable.
        Children must partition their parent, exactly.
        """
        min_size = self._config.subject_min_cluster_size
        keepers = [g for g in groups if len(g) >= min_size]
        strays = [g for g in groups if len(g) < min_size]
        if not keepers:
            return groups
        if not strays:
            return keepers

        centroids = self._centroids.get_centroids_batch(
            sorted({t for g in groups for t in g})
        )

        def mean_vec(group: set[int]) -> np.ndarray | None:
            vecs = [centroids[t] for t in group if t in centroids]
            return _unit(np.mean(np.stack(vecs), axis=0)) if vecs else None

        keeper_vecs = [(g, mean_vec(g)) for g in keepers]
        for stray in strays:
            sv = mean_vec(stray)
            best, best_sim = keeper_vecs[0][0], -2.0
            if sv is not None:
                for g, gv in keeper_vecs:
                    if gv is None:
                        continue
                    sim = float(np.dot(sv, gv))
                    if sim > best_sim:
                        best, best_sim = g, sim
            best |= stray
        return keepers

    def _should_split(
        self, cluster: set[int], memory_count: int, depth: int, state: dict,
    ) -> bool:
        if depth >= self._config.subject_max_depth:
            return False
        min_size = self._config.subject_min_cluster_size
        if len(cluster) < 2 * min_size:
            return False
        too_many_tags = len(cluster) > max(
            min_size,
            int(state["eligible_tags"] * self._config.subject_max_component_fraction),
        )
        too_many_memories = (
            memory_count / state["total_memories"]
            > self._config.subject_max_memory_fraction
        )
        return too_many_tags or too_many_memories

    @staticmethod
    def _row_to_subject(r) -> dict:
        keys = r.keys()
        display = r["display_name"] if "display_name" in keys else None
        return {
            "id": r["id"],
            "name": r["name"],
            "display_name": display,
            # What a person should see: the given name if it has one,
            # otherwise the derived tag-join
            "label": display or r["name"],
            "path": r["path"] or r["name"],
            "depth": r["depth"],
            "parent_id": r["parent_id"],
            "description": r["description"],
            "member_tag_ids": json.loads(r["member_tag_ids"]),
            "memory_count": r["memory_count"],
        }

    _SELECT_COLS = (
        "id, name, display_name, path, depth, parent_id, description, "
        "member_tag_ids, memory_count"
    )

    def rename(self, path: str, display_name: str | None) -> dict | None:
        """Give a shelf a human name (or clear it with None).

        display_name is authored, never derived — the roster rebuild
        refreshes everything else about a node and leaves this alone.
        """
        node = self.resolve(path).get("match")
        if node is None:
            return None
        self._db.execute(
            "UPDATE subjects SET display_name = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') WHERE id = ?",
            (display_name, node["id"]),
        )
        node["display_name"] = display_name
        node["label"] = display_name or node["name"]
        return node

    def resolve(self, query: str, limit: int = 6) -> dict:
        """Address a shelf the way a person would name it.

        Tries, in order: exact path, exact given name, exact derived name,
        a unique final path segment, a tag the shelf actually holds, then
        a substring of any of those.  Returns {"match": node} when one
        candidate clearly wins, {"ambiguous": [...]} when several tie, or
        {"no_match": True} with the nearest suggestions.

        A tag hit resolves to the DEEPEST shelf holding that tag — asking
        for "gnosticism" should land on the gnosticism shelf, not on the
        library-sized ancestor that also contains it.
        """
        q = (query or "").strip()
        if not q:
            return {"no_match": True, "suggestions": []}
        ql = q.lower()

        rows = self._db.execute(
            f"SELECT {self._SELECT_COLS} FROM subjects WHERE status = 'active'"
        ).fetchall()
        if not rows:
            return {"no_match": True, "suggestions": []}
        nodes = [self._row_to_subject(r) for r in rows]

        # 1. Exact identifiers
        for n in nodes:
            if q in (n["path"], n["name"]) or (
                n["display_name"] and n["display_name"].lower() == ql
            ):
                return {"match": n, "how": "exact"}

        # 2. Unique final path segment
        seg = [n for n in nodes if n["path"].split("/")[-1].lower() == ql]
        if len(seg) == 1:
            return {"match": seg[0], "how": "path-segment"}

        # 3. A tag the shelf holds — the deepest (most specific) one wins.
        # Tags are hyphenated by convention but people type spaces, so
        # "rocket league" has to find `rocket-league`.
        qh = ql.replace(" ", "-")
        tag_rows = self._db.execute(
            "SELECT id, name FROM tags WHERE lower(name) IN (?, ?) "
            "OR lower(name) LIKE ? OR lower(name) LIKE ?",
            (ql, qh, f"%{ql}%", f"%{qh}%"),
        ).fetchall()
        exact_tag_ids = {
            r["id"] for r in tag_rows if r["name"].lower() in (ql, qh)
        }
        loose_tag_ids = {r["id"] for r in tag_rows}
        for tag_ids, how in ((exact_tag_ids, "tag"), (loose_tag_ids, "tag-partial")):
            if not tag_ids:
                continue
            holders = [
                n for n in nodes if tag_ids & set(n["member_tag_ids"])
            ]
            if holders:
                holders.sort(key=lambda n: (-n["depth"], n["memory_count"]))
                best = holders[0]
                peers = [
                    n for n in holders
                    if n["depth"] == best["depth"]
                    and n["memory_count"] == best["memory_count"]
                ]
                if len(peers) == 1:
                    return {"match": best, "how": how}
                return {"ambiguous": peers[:limit], "how": how}

        # 4. Substring of a name or path
        subs = [
            n for n in nodes
            if ql in n["name"].lower() or qh in n["name"].lower()
            or ql in n["path"].lower() or qh in n["path"].lower()
            or (n["display_name"] and ql in n["display_name"].lower())
        ]
        if len(subs) == 1:
            return {"match": subs[0], "how": "substring"}
        if subs:
            subs.sort(key=lambda n: (-n["depth"], n["memory_count"]))
            return {"ambiguous": subs[:limit], "how": "substring"}

        # 5. Nothing matched — offer the nearest by token overlap
        tokens = {t for t in ql.replace("/", " ").replace("+", " ").split() if t}
        scored = []
        for n in nodes:
            hay = f"{n['name']} {n['display_name'] or ''}".lower()
            hit = sum(1 for t in tokens if t in hay)
            if hit:
                scored.append((hit, n))
        scored.sort(key=lambda x: (-x[0], x[1]["memory_count"]))
        return {
            "no_match": True,
            "suggestions": [n for _, n in scored[:limit]],
        }

    def list_subjects(
        self,
        status: str = "active",
        limit: int = 50,
        max_depth: int | None = None,
    ) -> list[dict]:
        sql = (
            f"SELECT {self._SELECT_COLS}, status, updated_at "
            "FROM subjects WHERE status = ?"
        )
        params: list = [status]
        if max_depth is not None:
            sql += " AND depth <= ?"
            params.append(max_depth)
        sql += " ORDER BY depth ASC, memory_count DESC LIMIT ?"
        params.append(limit)
        rows = self._db.execute(sql, tuple(params)).fetchall()
        out = []
        for r in rows:
            rec = self._row_to_subject(r)
            rec["status"] = r["status"]
            rec["updated_at"] = r["updated_at"]
            out.append(rec)
        return out

    def route_depth(self) -> int:
        """The depth routing should dispatch at.

        Honours an explicit config value; when it is negative, picks the
        shallowest depth holding more than one shelf — dispatching to a
        lone root that spans the whole library is not a choice.
        """
        configured = self._config.consult_route_depth
        if configured >= 0:
            return configured
        rows = self._db.execute(
            "SELECT depth, COUNT(*) AS n FROM subjects "
            "WHERE status = 'active' GROUP BY depth ORDER BY depth ASC"
        ).fetchall()
        for r in rows:
            if r["n"] > 1:
                return r["depth"]
        # No level offers a choice (a chain).  Fall back to the SHALLOWEST
        # node, which covers everything — descending would route to one
        # narrow leaf and silently drop the tags peeled off above it.
        return rows[0]["depth"] if rows else 0

    def get_by_path(self, path: str) -> dict | None:
        """Address a node by path, name, given name, or anything close.

        Delegates to resolve(); returns None when the query is ambiguous
        or matches nothing, so callers that need the alternatives should
        use resolve() directly.
        """
        return self.resolve(path).get("match")

    def children_of(self, path: str) -> list[dict]:
        rows = self._db.execute(
            "SELECT s.id, s.name, s.display_name, s.path, s.depth, s.parent_id, "
            "s.description, s.member_tag_ids, s.memory_count FROM subjects s "
            "JOIN subjects p ON p.id = s.parent_id "
            "WHERE s.status = 'active' AND (p.path = ? OR p.name = ?) "
            "ORDER BY s.memory_count DESC",
            (path, path),
        ).fetchall()
        return [self._row_to_subject(r) for r in rows]

    def tree(self, limit: int = 200) -> list[dict]:
        """The whole active roster as an ordered, indented outline."""
        rows = self._db.execute(
            f"SELECT {self._SELECT_COLS} FROM subjects "
            "WHERE status = 'active' ORDER BY path ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_subject(r) for r in rows]

    # The routing frontier: descend from the roots until a node is small
    # enough to be a meaningful choice, and stop there.
    #
    # A uniform depth cannot express this on an unbalanced tree — one
    # branch's depth-1 node may hold three quarters of the library while
    # another's is already specific.  So a node routes iff it is itself
    # under the size cap (or a leaf), AND its parent was over the cap.
    # That yields complete coverage with no redundancy, and automatically
    # excludes both the "everything" root and any oversized internal node.
    _ROUTABLE = (
        "((subjects.memory_count <= ? OR NOT EXISTS ("
        "   SELECT 1 FROM subjects c WHERE c.parent_id = subjects.id"
        "   AND c.status = 'active'))"
        " AND (subjects.parent_id IS NULL OR EXISTS ("
        "   SELECT 1 FROM subjects p WHERE p.id = subjects.parent_id"
        "   AND p.memory_count > ?)))"
    )

    def _routing_cap(self) -> int:
        """Memory count above which a shelf is too broad to route to."""
        row = self._db.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE is_archived = 0"
        ).fetchone()
        total = max(row["n"] or 0, 1)
        return max(1, int(total * self._config.subject_max_memory_fraction))

    def active_with_centroids(self, max_depth: int | None = None) -> list[dict]:
        """Active subjects with centroid vectors, largest first."""
        sql = (
            f"SELECT {self._SELECT_COLS}, centroid "
            "FROM subjects WHERE status = 'active' AND centroid IS NOT NULL"
        )
        params: list = []
        if max_depth is not None:
            cap = self._routing_cap()
            sql += " AND " + self._ROUTABLE
            params.extend([cap, cap])
        sql += " ORDER BY memory_count DESC"
        rows = self._db.execute(sql, tuple(params)).fetchall()
        out = []
        for r in rows:
            rec = self._row_to_subject(r)
            rec["centroid"] = _blob_to_array(r["centroid"])
            out.append(rec)
        return out

    def match(
        self,
        query_vec: np.ndarray,
        threshold: float,
        top_m: int,
        max_depth: int | None = None,
    ) -> list[tuple[dict, float]]:
        """Active subjects whose centroids match a query vector, best first.

        max_depth keeps routing coarse: dispatching to a handful of broad
        shelves is more reliable than choosing among many narrow ones, and
        a parent's tag set covers all of its children's material anyway.
        Teaching addresses deeper nodes directly by path instead.
        """
        sql = (
            f"SELECT {self._SELECT_COLS}, centroid "
            "FROM subjects WHERE status = 'active' AND centroid IS NOT NULL"
        )
        params: list = []
        if max_depth is not None:
            cap = self._routing_cap()
            sql += " AND " + self._ROUTABLE
            params.extend([cap, cap])
        rows = self._db.execute(sql, tuple(params)).fetchall()
        if not rows:
            return []
        vec = _unit(query_vec.astype(np.float32))
        scored = []
        for r in rows:
            sim = float(np.dot(vec, _blob_to_array(r["centroid"])))
            if sim >= threshold:
                scored.append((self._row_to_subject(r), sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:top_m]

    def _topical_tag_ids(self) -> set[int]:
        """Tags eligible for subject membership: have centroids, aren't
        temporal/letter bookkeeping tags, and aren't hub/format tags.

        The hub rule is the stop-word rule for tags: a tag attached to
        most of the corpus (document, pdf) carries no topical signal and
        welds every real topic into one component.
        """
        total_row = self._db.execute(
            "SELECT COUNT(*) AS n FROM memories WHERE is_archived = 0"
        ).fetchone()
        total = max(total_row["n"] or 0, 1)
        hub_cap = self._config.subject_hub_tag_max_fraction * total

        rows = self._db.execute(
            """
            SELECT tc.tag_id, t.name, tc.memory_count FROM tag_centroids tc
            JOIN tags t ON t.id = tc.tag_id
            WHERE t.is_active = 1
            """
        ).fetchall()
        return {
            r["tag_id"] for r in rows
            if len(r["name"]) > 1
            and not r["name"].startswith(_NON_TOPICAL_PREFIXES)
            and r["memory_count"] <= hub_cap
        }

    def _warm_centers(
        self,
        have: list[int],
        matrix: np.ndarray,
        prior_leaf: dict[int, int],
    ) -> np.ndarray | None:
        """Initial centres taken from how these tags were grouped before.

        This is the fix for the splitter's real instability.  k-means++
        seeds with ``matrix[rng.randint(len(have))]`` — an index drawn from
        the array's LENGTH — and ``have`` is sorted tag ids, so inserting a
        handful of tags both changes the draw and shifts every position
        after them.  Seed 42 does not help: it is the draw that moves.
        Measured, six new tags in a 1,020-tag root reseeded the clustering
        and re-partitioned all three levels beneath it.

        Grouping by the standing roster's leaf assignment removes the
        length dependence entirely — centres come from the data's own prior
        structure, so a few new tags perturb the assignment instead of
        reseeding it.  Returns None when there is nothing to inherit (cold
        start, or a parent whose children no longer resolve), and the
        caller falls back to k-means++.
        """
        min_size = self._config.subject_min_cluster_size
        groups: dict[int, list[int]] = {}
        for idx, tag_id in enumerate(have):
            leaf = prior_leaf.get(tag_id)
            if leaf is not None:
                groups.setdefault(leaf, []).append(idx)

        usable = [idxs for idxs in groups.values() if len(idxs) >= min_size]
        if len(usable) < 2:
            return None
        # Largest first, so if the cap bites it drops the least-established
        # groups rather than an arbitrary set.
        usable.sort(key=len, reverse=True)
        return np.stack([
            _unit(matrix[idxs].mean(axis=0).astype(np.float32))
            for idxs in usable
        ])

    def _split_in_centroid_space(
        self, cluster: set[int], prior_leaf: dict[int, int] | None = None,
        target_k: int | None = None,
    ) -> list[set[int]]:
        """K-means over tag centroids (pure numpy).

        Seeded from the standing roster's grouping where there is one, and
        from k-means++ (fixed seed) where there is not.  Falls back to the
        unsplit cluster if centroids are missing.

        Warm seeding means the tree's SHAPE is inherited: each rebuild
        starts in the basin the last one settled in, and refines it rather
        than re-deriving it.  That is deliberate — a shelf path is an
        address that digests, commissions and the teach cycle all quote,
        and an address that re-derives is not an address.  The cost is that
        the tree can no longer discover a better partition on its own;
        ``rebuild(cold=True)`` is the way back to a clean derivation.
        """
        ids = sorted(cluster)
        centroids = self._centroids.get_centroids_batch(ids)
        have = [t for t in ids if t in centroids]
        if len(have) < 2 * self._config.subject_min_cluster_size:
            return [cluster]
        matrix = np.stack([centroids[t] for t in have])

        C = self._warm_centers(have, matrix, prior_leaf) if prior_leaf else None
        if target_k and target_k >= 2:
            # The node's OWN arity, which is the only stable thing here.
            # prior_leaf spans a whole depth, so how many groups a node's
            # tags fall into changes with whether that node has children —
            # and that made the seed alternate: two children fed the split
            # [2,7,7], none fed [2,14], and the second was killed by the
            # memory-separation test, restoring the first. A live 2-cycle.
            if C is not None and len(C) != target_k:
                C = None
            k = target_k
        elif C is not None:
            k = len(C)
        else:
            k = self._config.subject_split_k
            if k <= 0:
                k = max(2, int(round((len(have) / 2) ** 0.5)))
            k = min(
                k, len(have) // self._config.subject_min_cluster_size, len(have),
            )
            if k < 2:
                return [cluster]

            rng = np.random.RandomState(42)
            # k-means++ seeding on cosine (unit vectors: euclidean ~ cosine)
            centers = [matrix[rng.randint(len(have))]]
            for _ in range(k - 1):
                dists = np.min(
                    np.stack([1.0 - matrix @ c for c in centers]), axis=0,
                )
                dists = np.clip(dists, 0, None)
                total = dists.sum()
                if total <= 0:
                    centers.append(matrix[rng.randint(len(have))])
                    continue
                centers.append(matrix[rng.choice(len(have), p=dists / total)])
            C = np.stack(centers)

        assign = np.zeros(len(have), dtype=np.int64)
        for _ in range(50):
            new_assign = np.argmax(matrix @ C.T, axis=1)
            if np.array_equal(new_assign, assign) and _ > 0:
                break
            assign = new_assign
            for j in range(k):
                members = matrix[assign == j]
                if len(members):
                    C[j] = _unit(members.mean(axis=0))

        groups: dict[int, set[int]] = {}
        for idx, tag_id in enumerate(have):
            groups.setdefault(int(assign[idx]), set()).add(tag_id)
        out = list(groups.values())
        # Tags with no centroid can't be placed geometrically, but they
        # must still land somewhere: children have to partition the parent
        # exactly, or their material falls off the routing frontier.
        unplaced = cluster - set(have)
        if unplaced and out:
            max(out, key=len).update(unplaced)
        return out

    def _prior_by_depth(self) -> dict[int, dict[int, int]]:
        """Tag -> shelf, one map per depth of the standing roster.

        Level-aware on purpose.  Seeding a split from the roster's *leaf*
        assignment collapses the tree: leaves from every depth arrive as
        one flat set of sibling centres, so a root inherits ~50 children at
        once and none of them is large enough to split again.  Measured, it
        took a three-level tree down to one.

        Splitting a node at depth d therefore seeds from depth d+1 — the
        arity that level actually had — which inherits the shape rather
        than flattening it.

        Snapshotted once at the start of a rebuild and never re-read:
        _persist_node writes each node before recursing into it, so a lazy
        read would inherit from the rebuild in progress instead of the one
        before it.
        """
        rows = self._db.execute(
            "SELECT id, depth, member_tag_ids FROM subjects "
            "WHERE status = 'active' ORDER BY depth ASC, id ASC"
        ).fetchall()
        by_depth: dict[int, dict[int, int]] = {}
        for r in rows:
            level = by_depth.setdefault(r["depth"], {})
            try:
                for tag_id in json.loads(r["member_tag_ids"]):
                    level[tag_id] = r["id"]
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return by_depth

    def _prior_components(self) -> dict[int, int]:
        """Which root component each tag sat in under the standing roster.

        Read before anything is persisted, so it describes the partition
        readers are currently addressing — the thing hysteresis exists to
        hold still.  Depth 0 only: those nodes carry their whole subtree's
        tags, which is exactly the component boundary that matters.

        Empty on a cold start (no roster yet), and callers treat that as
        "no hysteresis" rather than "nothing may merge" — a fresh install
        must still be able to form its first partition.
        """
        rows = self._db.execute(
            "SELECT id, member_tag_ids FROM subjects "
            "WHERE status = 'active' AND depth = 0"
        ).fetchall()
        prior: dict[int, int] = {}
        for r in rows:
            try:
                for tag_id in json.loads(r["member_tag_ids"]):
                    prior[tag_id] = r["id"]
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
        return prior

    def _connected_components(
        self, tag_ids: set[int], cold: bool = False,
    ) -> list[set[int]]:
        """Group tags into components, with hysteresis against churn.

        A single co-retrieval is enough to KEEP two tags in the shelf they
        already share, but not enough to MERGE two shelves that were
        separate.  Without that asymmetry, component formation is a
        knife-edge — see subject_edge_merge_min_count for the measurement.

        The cost of the asymmetry, stated plainly: the roster is no longer
        a pure function of the geometry.  It depends on the roster before
        it, which is what makes a shelf path a stable address instead of a
        fresh derivation each rebuild.  It remains idempotent run to run.
        """
        min_strength = self._config.subject_min_edge_strength
        keep_min = self._config.subject_edge_keep_min_count
        merge_min = self._config.subject_edge_merge_min_count
        prior = {} if cold else self._prior_components()

        rows = self._db.execute(
            "SELECT tag_a_id, tag_b_id, co_retrieval_count FROM centroid_edges "
            "WHERE edge_strength >= ?",
            (min_strength,),
        ).fetchall()

        adjacency: dict[int, set[int]] = {t: set() for t in tag_ids}
        for r in rows:
            a, b = r["tag_a_id"], r["tag_b_id"]
            if a not in tag_ids or b not in tag_ids:
                continue
            count = r["co_retrieval_count"]
            if count < keep_min:
                continue
            if prior and count < merge_min:
                root_a = prior.get(a)
                if root_a is None or root_a != prior.get(b):
                    # Too little evidence to fuse groups that were apart —
                    # or to attach a tag no shelf has held yet.
                    continue
            adjacency[a].add(b)
            adjacency[b].add(a)

        visited: set[int] = set()
        components: list[set[int]] = []
        for tag in tag_ids:
            if tag in visited:
                continue
            component: set[int] = set()
            stack = [tag]
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                stack.extend(adjacency[node] - visited)
            components.append(component)
        return components

    def _describe(
        self, cluster: set[int], tag_members: dict[int, set[int]] | None = None,
    ) -> tuple[str | None, str, np.ndarray | None, int]:
        """Name, description, weighted centroid, and memory count for a cluster."""
        placeholders = ",".join("?" * len(cluster))
        rows = self._db.execute(
            f"""
            SELECT tc.tag_id, tc.centroid, tc.memory_count, t.name
            FROM tag_centroids tc JOIN tags t ON t.id = tc.tag_id
            WHERE tc.tag_id IN ({placeholders})
            ORDER BY tc.memory_count DESC
            """,
            tuple(cluster),
        ).fetchall()
        if not rows:
            return None, "", None, 0

        # '+' separator: tag names already contain hyphens, so a '-' join
        # would be ambiguous ('a-b'+'c' vs 'a'+'b-c') and collide
        top_names = [r["name"] for r in rows[: self._config.subject_max_name_tags]]
        name = "+".join(top_names)
        description = "members: " + ", ".join(r["name"] for r in rows[:12])

        weights = np.array([max(r["memory_count"], 1) for r in rows], dtype=np.float32)
        matrix = np.stack([_blob_to_array(r["centroid"]) for r in rows])
        centroid = _unit((matrix * weights[:, None]).sum(axis=0) / weights.sum())

        if tag_members is not None:
            members: set[int] = set()
            for t in cluster:
                members.update(tag_members.get(t, ()))
            memory_count = len(members)
        else:
            count_row = self._db.execute(
                f"SELECT COUNT(DISTINCT mt.memory_id) AS n FROM memory_tags mt "
                f"JOIN memories m ON m.id = mt.memory_id "
                f"WHERE mt.tag_id IN ({placeholders}) "
                f"AND m.is_archived = 0 AND m.ingestion_tier != 'quarantine'",
                tuple(cluster),
            ).fetchone()
            memory_count = count_row["n"] if count_row else 0

        return name, description, centroid, memory_count

    def _load_tag_members(self) -> dict[int, set[int]]:
        """tag_id -> set of live memory ids, as compact ints.

        One scan replaces a COUNT(DISTINCT) per node.  A recursive rebuild
        describes dozens of nodes and runs under the foreground lock, so
        paying for the join once matters.
        """
        rows = self._db.execute(
            "SELECT mt.tag_id, mt.memory_id FROM memory_tags mt "
            "JOIN memories m ON m.id = mt.memory_id "
            "WHERE m.is_archived = 0 AND m.ingestion_tier != 'quarantine'"
        ).fetchall()
        index: dict[str, int] = {}
        out: dict[int, set[int]] = {}
        for r in rows:
            mid = r["memory_id"]
            key = index.get(mid)
            if key is None:
                key = len(index)
                index[mid] = key
            out.setdefault(r["tag_id"], set()).add(key)
        return out


class FaqDesk:
    """Crystallized answers to recurring questions — the front desk's card file.

    Entries are promoted as candidates by the demand ledger and only become
    servable after judgment (approve).  Serving verifies the supporting
    memories are still intact; a broken entry goes stale instead of lying.
    """

    def __init__(self, config: ChicoryConfig, db: DatabaseEngine) -> None:
        self._config = config
        self._db = db

    def serve(self, query_vec: np.ndarray) -> dict | None:
        """Fast path: match an active FAQ entry, verify it, serve it."""
        rows = self._db.execute(
            "SELECT id, question_text, question_centroid, answer, "
            "supporting_memory_ids, subject_ids, author, last_verified_at "
            "FROM faq_entries WHERE status = 'active'"
        ).fetchall()
        if not rows:
            return None
        vec = _unit(query_vec.astype(np.float32))
        threshold = self._config.faq_serve_threshold
        best_row, best_sim = None, threshold
        for r in rows:
            sim = float(np.dot(vec, _blob_to_array(r["question_centroid"])))
            if sim >= best_sim:
                best_row, best_sim = r, sim
        if best_row is None:
            return None

        supporting = json.loads(best_row["supporting_memory_ids"])
        if not self._verify_supporting(supporting):
            self._db.execute(
                "UPDATE faq_entries SET status = 'stale' WHERE id = ?",
                (best_row["id"],),
            )
            return None

        self._db.execute(
            "UPDATE faq_entries SET hit_count = hit_count + 1 WHERE id = ?",
            (best_row["id"],),
        )
        return {
            "faq_id": best_row["id"],
            "question": best_row["question_text"],
            "answer": best_row["answer"],
            "supporting_memory_ids": supporting,
            "author": best_row["author"],
            "last_verified_at": best_row["last_verified_at"],
            "match_score": round(best_sim, 3),
        }

    def _verify_supporting(self, memory_ids: list[str]) -> bool:
        """A cached answer is only as good as its sources: every supporting
        memory must still exist, unarchived and unquarantined."""
        if not memory_ids:
            return False
        placeholders = ",".join("?" * len(memory_ids))
        rows = self._db.execute(
            f"SELECT COUNT(*) AS n FROM memories "
            f"WHERE id IN ({placeholders}) AND is_archived = 0 "
            f"AND ingestion_tier != 'quarantine'",
            tuple(memory_ids),
        ).fetchone()
        return (rows["n"] or 0) == len(memory_ids)

    def create_candidate(
        self,
        question_text: str,
        question_vec: np.ndarray,
        answer_draft: str,
        supporting_memory_ids: list[str],
        subject_ids: list[int],
    ) -> int:
        cur = self._db.execute(
            "INSERT INTO faq_entries (question_text, question_centroid, answer, "
            "supporting_memory_ids, subject_ids, status, author) "
            "VALUES (?, ?, ?, ?, ?, 'candidate', 'librarian:mechanical')",
            (question_text, _array_to_blob(_unit(question_vec.astype(np.float32))),
             answer_draft, json.dumps(supporting_memory_ids), json.dumps(subject_ids)),
        )
        return cur.lastrowid

    def has_similar(self, query_vec: np.ndarray, threshold: float) -> bool:
        """A live (candidate/active) entry near this vector — dedup for
        promotion.  Stale and retired entries don't block: stale ones are
        refreshed instead, retired ones were judged away."""
        rows = self._db.execute(
            "SELECT question_centroid FROM faq_entries "
            "WHERE status IN ('candidate', 'active')"
        ).fetchall()
        vec = _unit(query_vec.astype(np.float32))
        return any(
            float(np.dot(vec, _blob_to_array(r["question_centroid"]))) >= threshold
            for r in rows
        )

    def refresh_stale_if_similar(
        self,
        query_vec: np.ndarray,
        threshold: float,
        answer_draft: str,
        supporting_memory_ids: list[str],
    ) -> int | None:
        """A stale entry near this vector gets a fresh mechanical draft and
        goes back to candidate for re-judgment, instead of blocking the
        cluster forever.  Returns the refreshed id, or None."""
        rows = self._db.execute(
            "SELECT id, question_centroid FROM faq_entries WHERE status = 'stale'"
        ).fetchall()
        vec = _unit(query_vec.astype(np.float32))
        for r in rows:
            if float(np.dot(vec, _blob_to_array(r["question_centroid"]))) >= threshold:
                self._db.execute(
                    "UPDATE faq_entries SET status = 'candidate', answer = ?, "
                    "supporting_memory_ids = ?, author = 'librarian:mechanical', "
                    "last_verified_at = NULL WHERE id = ?",
                    (answer_draft, json.dumps(supporting_memory_ids), r["id"]),
                )
                return r["id"]
        return None

    def candidates(self, limit: int = 20) -> list[dict]:
        rows = self._db.execute(
            "SELECT id, question_text, answer, supporting_memory_ids, created_at "
            "FROM faq_entries WHERE status = 'candidate' "
            "ORDER BY created_at ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "faq_id": r["id"],
                "question": r["question_text"],
                "draft_answer": r["answer"],
                "supporting_memory_ids": json.loads(r["supporting_memory_ids"]),
                "created_at": r["created_at"],
            }
            for r in rows
        ]

    def approve(
        self,
        faq_id: int,
        answer: str | None = None,
        author: str | None = None,
        supporting_memory_ids: list[str] | None = None,
    ) -> bool:
        """Judgment: candidate (or stale) entry becomes servable.

        `supporting_memory_ids` re-grounds the card.  The mechanical draft
        is built from whatever the promotion retrieval happened to return,
        and an approver who rewrites the answer from a better retrieval
        must be able to move the provenance with it — otherwise the served
        card cites one set of memories in its text while `serve` verifies
        another, and a source the answer never used could stale a card
        that is perfectly sound.  Omit it to keep the existing sources.
        """
        row = self._db.execute(
            "SELECT status FROM faq_entries WHERE id = ?", (faq_id,)
        ).fetchone()
        if row is None or row["status"] == "retired":
            return False
        now = datetime.utcnow().isoformat()
        sets = ["status = 'active'", "author = ?", "last_verified_at = ?"]
        params: list = [author or "librarian:approved", now]
        if answer is not None:
            sets.insert(1, "answer = ?")
            params.insert(0, answer)
        if supporting_memory_ids is not None:
            sets.append("supporting_memory_ids = ?")
            params.append(json.dumps(supporting_memory_ids))
        params.append(faq_id)
        self._db.execute(
            f"UPDATE faq_entries SET {', '.join(sets)} WHERE id = ?",
            tuple(params),
        )
        return True

    def retire_to_stale(self, faq_id: int) -> bool:
        """Pull a served entry back to stale (not retired).

        Used when an answer turns out to rest on something refuted: the
        question is still good, the answer needs rebuilding.
        """
        cur = self._db.execute(
            "UPDATE faq_entries SET status = 'stale' WHERE id = ? "
            "AND status != 'retired'", (faq_id,),
        )
        return cur.rowcount > 0

    def retire(self, faq_id: int) -> bool:
        cur = self._db.execute(
            "UPDATE faq_entries SET status = 'retired' WHERE id = ?", (faq_id,)
        )
        return cur.rowcount > 0


class CommissionDesk:
    """Assigned research on known gaps — the library's work orders."""

    def __init__(self, config: ChicoryConfig, db: DatabaseEngine) -> None:
        self._config = config
        self._db = db

    def file(
        self,
        question_text: str,
        question_vec: np.ndarray | None = None,
        source: str = "manual",
        subject_ids: list[int] | None = None,
    ) -> dict:
        """Open a ticket, deduplicating against similar open ones."""
        if question_vec is not None:
            existing = self.match_open(question_vec)
            if existing:
                return {"status": "exists", "commission_id": existing[0]["commission_id"]}
        blob = (
            _array_to_blob(_unit(question_vec.astype(np.float32)))
            if question_vec is not None else None
        )
        cur = self._db.execute(
            "INSERT INTO commissions (question_text, centroid, source, subject_ids) "
            "VALUES (?, ?, ?, ?)",
            (question_text, blob, source, json.dumps(subject_ids or [])),
        )
        return {"status": "filed", "commission_id": cur.lastrowid}

    def has_similar_any(self, query_vec: np.ndarray) -> bool:
        """Any commission — closed included — near this vector.  Used by
        automated filing so a closed ticket isn't re-filed forever; a human
        can always re-open deliberately via manual file."""
        rows = self._db.execute(
            "SELECT centroid FROM commissions WHERE centroid IS NOT NULL"
        ).fetchall()
        vec = _unit(query_vec.astype(np.float32))
        threshold = self._config.demand_cluster_threshold
        return any(
            float(np.dot(vec, _blob_to_array(r["centroid"]))) >= threshold
            for r in rows
        )

    def match_open(self, query_vec: np.ndarray) -> list[dict]:
        """Unclosed commissions near this vector — research pending."""
        rows = self._db.execute(
            "SELECT id, question_text, status, centroid FROM commissions "
            "WHERE status != 'closed' AND centroid IS NOT NULL"
        ).fetchall()
        vec = _unit(query_vec.astype(np.float32))
        threshold = self._config.demand_cluster_threshold
        hits = []
        for r in rows:
            sim = float(np.dot(vec, _blob_to_array(r["centroid"])))
            if sim >= threshold:
                hits.append({
                    "commission_id": r["id"],
                    "question": r["question_text"],
                    "status": r["status"],
                    "match_score": round(sim, 3),
                })
        hits.sort(key=lambda h: -h["match_score"])
        return hits

    def list(self, status: str = "open", limit: int = 20) -> list[dict]:
        rows = self._db.execute(
            "SELECT id, question_text, source, subject_ids, status, "
            "report_memory_id, created_at, updated_at "
            "FROM commissions WHERE status = ? ORDER BY created_at ASC LIMIT ?",
            (status, limit),
        ).fetchall()
        return [
            {
                "commission_id": r["id"],
                "question": r["question_text"],
                "source": r["source"],
                "subject_ids": json.loads(r["subject_ids"]),
                "status": r["status"],
                "report_memory_id": r["report_memory_id"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def set_status(
        self, commission_id: int, status: str, report_memory_id: str | None = None,
    ) -> bool:
        if status not in ("open", "claimed", "reported", "closed"):
            return False
        now = datetime.utcnow().isoformat()
        if report_memory_id is not None:
            cur = self._db.execute(
                "UPDATE commissions SET status = ?, report_memory_id = ?, "
                "updated_at = ? WHERE id = ?",
                (status, report_memory_id, now, commission_id),
            )
        else:
            cur = self._db.execute(
                "UPDATE commissions SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, commission_id),
            )
        return cur.rowcount > 0


class DemandLedger:
    """Queries as a first-class archive: what the library is asked, clustered.

    Recurring well-served questions crystallize into FAQ candidates;
    recurring badly-served questions become commissions.  The ledger is
    the librarian's meta-pattern across retrieval initiations.
    """

    _CLUSTER_EMA_ALPHA = 0.2

    def __init__(
        self,
        config: ChicoryConfig,
        db: DatabaseEngine,
        memory_store,
        faq_desk: FaqDesk,
        commission_desk: CommissionDesk,
        subject_roster: SubjectRoster,
        closure=None,
    ) -> None:
        self._config = config
        self._db = db
        self._memory_store = memory_store
        self._faq = faq_desk
        self._commissions = commission_desk
        self._subjects = subject_roster
        self._closure = closure

    def record(
        self,
        query_text: str,
        query_vec: np.ndarray | None,
        result_count: int,
        top_score: float,
        actor: str = "patron",
    ) -> None:
        """Record every query, whoever asked.

        Filtering happens at clustering time rather than here, so a query
        that did not count as demand still leaves a row.  What was asked of
        this library is part of its record even when it was not a patron
        asking.
        """
        blob = (
            _array_to_blob(_unit(query_vec.astype(np.float32)))
            if query_vec is not None else None
        )
        self._db.execute(
            "INSERT INTO query_history (query_text, embedding, result_count, "
            "top_score, actor) VALUES (?, ?, ?, ?, ?)",
            (query_text, blob, result_count, top_score, actor),
        )

    def _embed_query(self, text: str) -> np.ndarray | None:
        """Embed a query whose caller had no vector to give us.

        Returns None when there is nothing to embed or the engine is
        unavailable, so the caller can count the loss instead of pretending
        the row was clustered.
        """
        if not text or not text.strip():
            return None
        engine = getattr(self._memory_store, "_embedding", None)
        if engine is None:
            return None
        try:
            return _unit(engine.embed(text).astype(np.float32))
        except Exception:
            _log.exception("Demand ledger could not embed %r", text[:60])
            return None

    def process_batch(self, allow_retrieval: bool = True) -> dict:
        """Assign unprocessed queries to clusters, then promote or commission.

        allow_retrieval gates the live retrieval used to gather FAQ
        supporting memories — pass False when the vector index isn't
        built (promotion defers; clusters persist).
        """
        rows = self._db.execute(
            "SELECT id, query_text, embedding, top_score, actor FROM query_history "
            "WHERE processed = 0 ORDER BY id ASC LIMIT ?",
            (self._config.demand_process_batch,),
        ).fetchall()

        # No early return on an empty batch.  Promotion below is the retry
        # path for clusters deferred because the index was not built, and
        # returning here made that retry depend on new query traffic
        # arriving — so a cluster deferred just as traffic stopped was
        # never promoted at all.  Observed: eight hits on one question,
        # mean_top_score 0.92, both gates passed, and no FAQ card, because
        # every later pass took this branch.
        clusters = self._load_clusters()
        threshold = self._config.demand_cluster_threshold
        touched: set[int] = set()

        recovered = 0
        dropped = 0
        not_demand = 0
        for r in rows:
            if not actor_allowed(r["actor"], self._config.demand_actors):
                # Recorded, marked processed, never clustered: an agent's
                # research must not become the library's demand.
                not_demand += 1
                continue
            blob = r["embedding"]
            if blob is None:
                # record() stores the query even when the caller had no
                # vector to give it, and clustering then skipped the row
                # while the update below still marked it processed — demand
                # consumed and lost, silently.  Embed it here instead: it is
                # the same work record() would have done, and _promote
                # already embeds in this same call.
                vec = self._embed_query(r["query_text"])
                if vec is None:
                    dropped += 1
                    continue
                recovered += 1
            else:
                vec = _unit(_blob_to_array(blob))

            best_id, best_sim = None, threshold
            for cid, (centroid, _, _) in clusters.items():
                sim = float(np.dot(vec, centroid))
                if sim >= best_sim:
                    best_id, best_sim = cid, sim
            if best_id is None:
                cur = self._db.execute(
                    "INSERT INTO query_clusters "
                    "(centroid, representative_query, hit_count, mean_top_score) "
                    "VALUES (?, ?, 1, ?)",
                    (_array_to_blob(vec), r["query_text"], r["top_score"]),
                )
                clusters[cur.lastrowid] = (vec, 1, r["top_score"])
                touched.add(cur.lastrowid)
            else:
                centroid, hits, mean_score = clusters[best_id]
                a = self._CLUSTER_EMA_ALPHA
                new_centroid = _unit(a * vec + (1 - a) * centroid)
                new_mean = a * r["top_score"] + (1 - a) * mean_score
                self._db.execute(
                    "UPDATE query_clusters SET centroid = ?, hit_count = ?, "
                    "mean_top_score = ?, "
                    "last_seen_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') WHERE id = ?",
                    (_array_to_blob(new_centroid), hits + 1, new_mean, best_id),
                )
                clusters[best_id] = (new_centroid, hits + 1, new_mean)
                touched.add(best_id)

        # Rows that could not be embedded are marked processed too, and
        # counted above rather than lost quietly: leaving them unprocessed
        # would park them at the head of an ORDER BY id ASC queue forever,
        # starving every query behind them.
        self._db.executemany(
            "UPDATE query_history SET processed = 1 WHERE id = ?",
            [(r["id"],) for r in rows],
        )

        # Promotion considers ALL clusters, not just this batch's — a
        # cluster whose promotion was deferred (index not built) gets
        # retried on the next pass; has_similar dedup keeps it idempotent.
        faq_count, commission_count = self._promote(clusters, allow_retrieval)

        # Retention: processed raw rows older than the window are pruned —
        # the clusters retain the aggregate; the raw ledger is operational
        # telemetry, not memory.
        retention_days = self._config.demand_history_retention_days
        if rows and retention_days > 0:
            self._db.execute(
                "DELETE FROM query_history WHERE processed = 1 "
                "AND created_at < datetime('now', ?)",
                (f"-{int(retention_days)} days",),
            )

        if dropped:
            _log.warning(
                "Demand ledger dropped %d unembeddable quer%s",
                dropped, "y" if dropped == 1 else "ies",
            )
        return {
            "processed": len(rows),
            "not_demand": not_demand,
            "recovered_embeddings": recovered,
            "dropped_unembeddable": dropped,
            "faq_candidates": faq_count,
            "commissions_filed": commission_count,
        }

    def _load_clusters(self) -> dict[int, tuple[np.ndarray, int, float]]:
        rows = self._db.execute(
            "SELECT id, centroid, hit_count, mean_top_score FROM query_clusters"
        ).fetchall()
        return {
            r["id"]: (_blob_to_array(r["centroid"]), r["hit_count"], r["mean_top_score"])
            for r in rows
        }

    def _required_score(self, hits: int) -> float | None:
        """The quality bar this many asks must clear, or None to refuse.

        Repetition is evidence of demand and score is evidence of an
        answer; requiring a fixed amount of both is what got the gates
        calibrated backwards. The ladder trades them off instead — asked
        once, a question has to answer itself convincingly; asked
        repeatedly, a merely good answer earns a card.
        """
        tiers = self._config.faq_promote_tiers
        if not tiers:
            if hits < self._config.faq_promote_min_hits:
                return None
            return self._config.faq_promote_min_score
        for min_hits, min_score in sorted(tiers, key=lambda t: -t[0]):
            if hits >= min_hits:
                return min_score
        return None

    def _promote(
        self,
        clusters: dict[int, tuple[np.ndarray, int, float]],
        allow_retrieval: bool,
    ) -> tuple[int, int]:
        faq_count = commission_count = 0
        for cid in clusters:
            centroid, hits, mean_score = clusters[cid]
            rep_row = self._db.execute(
                "SELECT representative_query FROM query_clusters WHERE id = ?",
                (cid,),
            ).fetchone()
            if rep_row is None:
                continue
            question = rep_row["representative_query"]

            required = self._required_score(hits)
            if (
                required is not None
                and mean_score >= required
                and allow_retrieval
                and not self._faq.has_similar(
                    centroid, self._config.demand_cluster_threshold,
                )
            ):
                try:
                    results = self._memory_store.retrieve_hybrid(
                        question, top_k=5,
                    )
                except Exception:
                    _log.exception("FAQ promotion retrieval failed")
                    continue
                if not results:
                    continue
                supporting = [m.id for m, _ in results]
                # Never crystallize a refuted memory into an answer: the
                # FAQ is the most-served text in the system, and promotion
                # would quietly reinstate what closure just overturned.
                if self._closure is not None:
                    refuted = self._closure.superseded_among(supporting)
                    if refuted:
                        supporting = [m for m in supporting if m not in refuted]
                        results = [
                            (m, sc) for m, sc in results if m.id not in refuted
                        ]
                        if not results:
                            continue
                draft = "\n".join(
                    f"- {m.summary or m.content[:200]}" for m, _ in results
                )
                subject_ids = [
                    s["id"] for s, _ in self._subjects.match(
                        centroid, self._config.consult_subject_threshold,
                        self._config.consult_max_subjects,
                    )
                ]
                refreshed = self._faq.refresh_stale_if_similar(
                    centroid, self._config.demand_cluster_threshold,
                    draft, supporting,
                )
                if refreshed is None:
                    self._faq.create_candidate(
                        question, centroid, draft, supporting, subject_ids,
                    )
                faq_count += 1

            elif (
                hits >= self._config.commission_min_hits
                and mean_score < self._config.commission_weak_score
                and not self._faq.has_similar(
                    centroid, self._config.demand_cluster_threshold,
                )
                and not self._commissions.has_similar_any(centroid)
            ):
                subject_ids = [
                    s["id"] for s, _ in self._subjects.match(
                        centroid, self._config.consult_subject_threshold,
                        self._config.consult_max_subjects,
                    )
                ]
                result = self._commissions.file(
                    question, centroid,
                    source="recurring-weak", subject_ids=subject_ids,
                )
                if result["status"] == "filed":
                    commission_count += 1
        return faq_count, commission_count


# ── Analogic recall geometry ────────────────────────────────────────────
# Recall by role, not association: a query at offset δ from its home
# subject's centroid has counterparts near centroid_B + δ on other
# shelves.  Results are conjectures — label them as such downstream.


def analogic_target(
    home_centroid: np.ndarray, query_vec: np.ndarray, other_centroid: np.ndarray,
) -> np.ndarray:
    """Transport the query's within-subject offset into another subject's frame."""
    delta = _unit(query_vec) - _unit(home_centroid)
    return _unit(_unit(other_centroid) + delta)


def parallelness(
    result_vec: np.ndarray,
    other_centroid: np.ndarray,
    home_centroid: np.ndarray,
    query_vec: np.ndarray,
) -> float:
    """How parallel is the result's own offset to the query's offset?

    1.0 means the counterpart occupies the same role in its domain that
    the query occupies in its home domain."""
    delta = _unit(query_vec) - _unit(home_centroid)
    result_offset = _unit(result_vec) - _unit(other_centroid)
    dn, rn = np.linalg.norm(delta), np.linalg.norm(result_offset)
    if dn == 0 or rn == 0:
        return 0.0
    return float(np.dot(delta / dn, result_offset / rn))


class PenumbraLedger:
    """Near-miss statistics per memory: how often it was adjacent to a
    query and lost the top-k cut, versus how often it was actually served.

    High near_count with zero retrieved_count = measurably buried — a
    rescue candidate for the sweep.  The ledger writes findings, never
    scores: it observes the retrieval loop without joining it.
    """

    def __init__(self, config: ChicoryConfig, db: DatabaseEngine) -> None:
        self._config = config
        self._db = db

    def record(
        self,
        near: list[tuple[str, float]],
        retrieved_ids: list[str],
    ) -> None:
        now = datetime.utcnow().isoformat()
        if near:
            # Quarantined/archived memories can reach the raw runner-up list
            # through score channels that bypass the tier filter (tag-space
            # lexical); they can never be served, so counting them as "near"
            # would manufacture permanently unresolvable buried findings.
            ids = [mid for mid, _ in near]
            placeholders = ",".join("?" * len(ids))
            live = {
                r["id"] for r in self._db.execute(
                    f"SELECT id FROM memories WHERE id IN ({placeholders}) "
                    f"AND is_archived = 0 AND ingestion_tier != 'quarantine'",
                    tuple(ids),
                ).fetchall()
            }
            near = [(mid, s) for mid, s in near if mid in live]
        if near:
            self._db.executemany(
                "INSERT INTO retrieval_penumbra "
                "(memory_id, near_count, retrieved_count, last_near_at) "
                "VALUES (?, 1, 0, ?) "
                "ON CONFLICT(memory_id) DO UPDATE SET "
                "near_count = near_count + 1, last_near_at = excluded.last_near_at",
                [(mid, now) for mid, _ in near],
            )
        if retrieved_ids:
            self._db.executemany(
                "INSERT INTO retrieval_penumbra "
                "(memory_id, near_count, retrieved_count, last_retrieved_at) "
                "VALUES (?, 0, 1, ?) "
                "ON CONFLICT(memory_id) DO UPDATE SET "
                "retrieved_count = retrieved_count + 1, "
                "last_retrieved_at = excluded.last_retrieved_at",
                [(mid, now) for mid in retrieved_ids],
            )

    def buried_candidates(self, memory_ids: list[str]) -> set[str]:
        """Chronically almost-relevant, never served: burial with receipts."""
        if not memory_ids:
            return set()
        placeholders = ",".join("?" * len(memory_ids))
        rows = self._db.execute(
            f"SELECT memory_id FROM retrieval_penumbra "
            f"WHERE memory_id IN ({placeholders}) "
            f"AND near_count >= ? AND retrieved_count = 0",
            (*memory_ids, self._config.penumbra_buried_min_near),
        ).fetchall()
        return {r["memory_id"] for r in rows}

    def rescued(self, memory_ids: list[str]) -> set[str]:
        """Memories that have since been actually retrieved — their open
        'buried' findings can be auto-resolved."""
        if not memory_ids:
            return set()
        placeholders = ",".join("?" * len(memory_ids))
        rows = self._db.execute(
            f"SELECT memory_id FROM retrieval_penumbra "
            f"WHERE memory_id IN ({placeholders}) AND retrieved_count > 0",
            tuple(memory_ids),
        ).fetchall()
        return {r["memory_id"] for r in rows}

    def stats(self) -> dict:
        row = self._db.execute(
            "SELECT COUNT(*) AS tracked, "
            "SUM(CASE WHEN near_count >= ? AND retrieved_count = 0 "
            "THEN 1 ELSE 0 END) AS buried "
            "FROM retrieval_penumbra",
            (self._config.penumbra_buried_min_near,),
        ).fetchone()
        return {
            "tracked_memories": row["tracked"] or 0,
            "buried_candidates": row["buried"] or 0,
        }
