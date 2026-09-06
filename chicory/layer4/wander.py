"""The wandering librarian: errands, a distributed cursor, and a clock.

Every background phase in this system is *traffic-paced*.  The thread that
runs them is spawned by a retrieval, so the library only tidies itself
while somebody is talking to it.  The linear sweep shows the consequence
exactly: it walked all 123,907 memories in nine hours on one busy day,
reached 100% coverage, and then stopped — not because it was finished,
but because the traffic was.

Two things are missing, and they are different sizes.

The small one is a clock.  ``Heartbeat`` gives the background pass a tick
of its own.  It does not bypass anything: it calls the same function a
retrieval calls, which already yields the database lock between phases and
bails when the foreground is waiting.  Idle work and concurrent work are
the same code path, which is why the concurrent case needs no new
concurrency reasoning.

The larger one is somewhere to stand.  A patrol needs to know where it is,
and the useful form of that state is already demonstrated in this codebase:
``memories.swept_at``.  Not a pointer — a stamp on every object, read back
as ``ORDER BY swept_at ASC``, NULLs first.  Nothing to lose on restart,
nothing to lock, coverage guaranteed by the sort, and a new memory joins
the front of the queue for free because its stamp is absent rather than
stale.  ``librarian_state``, the pointer-shaped alternative, has one row in
it after all this time.

``patrol_visits`` is that pattern with the errand un-hardcoded, so several
errands can walk the same library at their own paces.  ``wander_state``
holds only what has no object to live on: the lap, the region, and the
seed — the seed because a walk that chooses randomly should be replayable
rather than merely describable.

Deliberate limits, since this thing runs unattended:

- An errand may repair what is mechanically broken.  It may not decide
  what is true.  Judgment stays with the specialist and the patron.
- Every memory an errand examines gets a visit row, whatever the verdict.
  A failure that recorded nothing would be retried forever at the head of
  the queue; recording it rotates it to the back instead.
- Nothing here writes to a derived cache.  Cooccurrence edges, canopy
  observations and the relational tensor all have decay-and-prune
  lifecycles, and work deposited into them would be eaten by the same
  maintenance this patrol runs beside.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import threading
from difflib import SequenceMatcher
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine

_log = logging.getLogger(__name__)

# Verdicts an errand may return for a memory it examined.
VERDICTS = ("ok", "acted", "failed", "skipped")


# ── Errand registry ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Domain:
    """What an errand walks.

    The distributed cursor was written for memories, but nothing about it
    is specific to them: it needs a table, a string id, and something to
    break ties by on the first lap.  ``patrol_visits.memory_id`` is already
    TEXT and the primary key is (errand, memory_id), so a walk over another
    kind of object needs no new table and cannot collide with the memory
    walk — the errand name keeps them apart.

    That matters because the library's METADATA has the same problem its
    memories do.  A survey found 844 tags with no edge, 392 with no
    centroid and 520 redundant fragments — none of it visible from inside
    normal use, because retrieval worked and the shelves rendered the whole
    time.  A one-off query finds that once; a walk keeps finding it as new
    tags arrive.
    """

    table: str
    id_expr: str
    tiebreak: str
    where: str = "1 = 1"
    label: str = "object"


DOMAIN_MEMORIES = Domain(
    table="memories m", id_expr="m.id", tiebreak="m.created_at",
    where="m.is_archived = 0", label="memory",
)
DOMAIN_TAGS = Domain(
    # Cast so the id matches patrol_visits.memory_id, which is TEXT.
    table="tags m", id_expr="CAST(m.id AS TEXT)", tiebreak="m.id",
    where="m.is_active = 1 AND LENGTH(m.name) > 1",
    label="tag",
)
DOMAIN_OBSERVATIONS = Domain(
    table="patrol_observations m", id_expr="CAST(m.id AS TEXT)",
    tiebreak="m.id", where="m.retracted = 0", label="observation",
)
DOMAINS = {
    "memories": DOMAIN_MEMORIES,
    "tags": DOMAIN_TAGS,
    "observations": DOMAIN_OBSERVATIONS,
}


@dataclass(frozen=True)
class Errand:
    """One standing question the patrol carries around the library.

    ``population`` narrows which objects the errand cares about at all — a
    SQL predicate over alias ``m``.  The cursor then orders *within* that
    population, so a targeted errand still gets the least-recently-visited
    guarantee, and an object it cannot fix rotates to the back instead of
    blocking the queue.

    ``domain`` says WHAT it walks: "memories" by default, or "tags" for an
    errand that patrols the metadata rather than the material.

    ``when`` is "always" for work that is safe beside a conversation, or
    "idle" for work that should wait until nobody is talking.
    """

    name: str
    description: str
    run: Callable[..., dict[str, tuple[str, str | None]]]
    population: str | None = None
    when: str = "always"
    domain: str = "memories"


ERRANDS: dict[str, Errand] = {}


def register(errand: Errand) -> Errand:
    ERRANDS[errand.name] = errand
    return errand


# ── State ───────────────────────────────────────────────────────────────


@dataclass
class WanderRecord:
    """One errand's row in ``wander_state``."""

    errand: str
    lap: int = 0
    cursor_hint: str | None = None
    seed: int = 0
    visited_count: int = 0
    acted_count: int = 0
    runs: int = 0
    enabled: bool = True
    last_run_at: str | None = None
    last_result: str | None = None


class WanderState:
    """Accessor for the small, centralised half of the patrol's state."""

    def __init__(self, db: DatabaseEngine) -> None:
        self._db = db

    def get(self, errand: str) -> WanderRecord:
        """Read an errand's state, creating it on first sight.

        The seed is derived from the errand name rather than a clock, so a
        fresh install and a restored backup walk the same pseudo-random
        route.  An errand that wants an unpredictable route can overwrite
        it; the point is that whatever route was taken can be replayed.
        """
        row = self._db.execute(
            "SELECT * FROM wander_state WHERE errand = ?", (errand,),
        ).fetchone()
        if row is None:
            seed = abs(hash(errand)) % (2**31)
            self._db.execute(
                "INSERT OR IGNORE INTO wander_state (errand, seed) VALUES (?, ?)",
                (errand, seed),
            )
            row = self._db.execute(
                "SELECT * FROM wander_state WHERE errand = ?", (errand,),
            ).fetchone()
        return WanderRecord(
            errand=row["errand"],
            lap=row["lap"],
            cursor_hint=row["cursor_hint"],
            seed=row["seed"],
            visited_count=row["visited_count"],
            acted_count=row["acted_count"],
            runs=row["runs"],
            enabled=bool(row["enabled"]),
            last_run_at=row["last_run_at"],
            last_result=row["last_result"],
        )

    def record_run(
        self,
        errand: str,
        *,
        visited: int,
        acted: int,
        result: str,
        cursor_hint: str | None = None,
    ) -> None:
        self._db.execute(
            "UPDATE wander_state SET "
            "visited_count = visited_count + ?, acted_count = acted_count + ?, "
            "runs = runs + 1, last_run_at = strftime('%Y-%m-%dT%H:%M:%f', 'now'), "
            "last_result = ?, cursor_hint = COALESCE(?, cursor_hint), "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') "
            "WHERE errand = ?",
            (visited, acted, result, cursor_hint, errand),
        )

    def advance_lap(self, errand: str) -> int:
        self._db.execute(
            "UPDATE wander_state SET lap = lap + 1, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') WHERE errand = ?",
            (errand,),
        )
        row = self._db.execute(
            "SELECT lap FROM wander_state WHERE errand = ?", (errand,),
        ).fetchone()
        return row["lap"] if row else 0

    def set_enabled(self, errand: str, enabled: bool) -> None:
        self.get(errand)  # ensure the row exists
        self._db.execute(
            "UPDATE wander_state SET enabled = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') WHERE errand = ?",
            (1 if enabled else 0, errand),
        )

    def all(self) -> list[WanderRecord]:
        rows = self._db.execute(
            "SELECT errand FROM wander_state ORDER BY errand"
        ).fetchall()
        return [self.get(r["errand"]) for r in rows]


# ── Lease ───────────────────────────────────────────────────────────────

_LEASE_KEY = "patrol_lease"


class PatrolLease:
    """One wanderer per library, whatever the process count.

    patrol_visits makes the walk restartable; it does not make it
    singular.  select -> run -> record is not atomic across processes, so
    every server sharing a database walks the same least-recently-visited
    batch at the same moment.  That is not hypothetical: an MCP client
    spawns a server per session, fourteen accumulated on one install, and
    the first heartbeat release had all fourteen patrolling on the same
    30s tick — three logging an identical `examined=50` in the same
    second, ~38% of visits duplicated, UNIQUE collisions in `embeddings`,
    and lock contention on a 12GB file.

    The lease is advisory and expiring rather than held: a holder that
    dies takes nothing with it, and the next tick to notice the expiry
    picks the work up.  It lives in librarian_state because that is
    exactly what that table is for, and the claim is a single conditional
    UPDATE — atomic in SQLite without a transaction, which matters
    because the losers must not block.
    """

    def __init__(self, config: ChicoryConfig, db: DatabaseEngine) -> None:
        self._config = config
        self._db = db
        self._owner = f"{socket.gethostname()}:{os.getpid()}"

    @property
    def owner(self) -> str:
        return self._owner

    def acquire(self) -> bool:
        """Claim the lease, or report that someone else holds a live one."""
        now = datetime.utcnow()
        expires = now + timedelta(seconds=self._config.wander_lease_seconds)
        payload = json.dumps(
            {"owner": self._owner, "expires_at": expires.isoformat()}
        )
        self._db.execute(
            "INSERT OR IGNORE INTO librarian_state (key, value) VALUES (?, NULL)",
            (_LEASE_KEY,),
        )
        # json_valid guards the row against anything that is not ours —
        # a malformed value must be reclaimable, not permanently fatal.
        cur = self._db.execute(
            "UPDATE librarian_state SET value = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') "
            "WHERE key = ? AND ("
            "    value IS NULL"
            " OR NOT json_valid(value)"
            " OR json_extract(value, '$.owner') = ?"
            " OR json_extract(value, '$.expires_at') <= ?"
            ")",
            (payload, _LEASE_KEY, self._owner, now.isoformat()),
        )
        return cur.rowcount == 1

    def release(self) -> None:
        """Drop the lease if it is ours, so the next tick need not wait."""
        self._db.execute(
            "UPDATE librarian_state SET value = NULL, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') "
            "WHERE key = ? AND json_valid(value) "
            "AND json_extract(value, '$.owner') = ?",
            (_LEASE_KEY, self._owner),
        )

    def holder(self) -> dict | None:
        row = self._db.execute(
            "SELECT value FROM librarian_state WHERE key = ?", (_LEASE_KEY,),
        ).fetchone()
        if row is None or not row["value"]:
            return None
        try:
            return json.loads(row["value"])
        except (json.JSONDecodeError, TypeError, ValueError):
            return None


# ── Patrol ──────────────────────────────────────────────────────────────


class Patrol:
    """Walks errands around the library, least-recently-visited first."""

    def __init__(
        self,
        config: ChicoryConfig,
        db: DatabaseEngine,
        embedding=None,
    ) -> None:
        self._config = config
        self._db = db
        self._embedding = embedding
        self._state = WanderState(db)
        self._lease = PatrolLease(config, db)
        self._last_run: dict[str, datetime] = {}

    @property
    def state(self) -> WanderState:
        return self._state

    @property
    def lease(self) -> PatrolLease:
        return self._lease

    # ── Cursor ──────────────────────────────────────────────────────────

    def select(self, errand: Errand, limit: int) -> list[str]:
        """The least-recently-visited memories this errand cares about."""
        return [mid for mid, _ in self.select_with_laps(errand, limit)]

    def select_with_laps(
        self, errand: Errand, limit: int,
    ) -> list[tuple[str, int | None]]:
        """The cursor, each row carrying the lap it was last visited on.

        SQLite sorts NULLs first in ASC order, so a memory with no visit row
        for this errand leads — the same property that makes swept_at work,
        preserved across the join.  Ordering is a scan-and-sort of the
        errand's population; that is affordable here because this runs in a
        background batch, and expensive errands narrow the population.

        The lap comes back with the row because it costs nothing here and
        answers "has this errand wrapped?" for free — see _lap_complete,
        which would otherwise need a COUNT over the population (measured at
        3.4s on a six-figure library, every batch, under the foreground
        lock).
        """
        dom = DOMAINS.get(errand.domain, DOMAIN_MEMORIES)
        where = dom.where
        if errand.population:
            where += f" AND ({errand.population})"
        rows = self._db.execute(
            f"""
            SELECT {dom.id_expr} AS oid, v.lap AS visit_lap FROM {dom.table}
            LEFT JOIN patrol_visits v
                   ON v.memory_id = {dom.id_expr} AND v.errand = ?
            WHERE {where}
            ORDER BY v.visited_at ASC, {dom.tiebreak} ASC
            LIMIT ?
            """,
            (errand.name, limit),
        ).fetchall()
        return [(r["oid"], r["visit_lap"]) for r in rows]

    def record_visits(
        self,
        errand: str,
        lap: int,
        verdicts: dict[str, tuple[str, str | None]],
    ) -> None:
        """Stamp every examined memory, whatever the errand concluded."""
        if not verdicts:
            return
        self._db.executemany(
            "INSERT INTO patrol_visits (errand, memory_id, visited_at, lap, verdict, detail) "
            "VALUES (?, ?, strftime('%Y-%m-%dT%H:%M:%f', 'now'), ?, ?, ?) "
            "ON CONFLICT(errand, memory_id) DO UPDATE SET "
            "visited_at = excluded.visited_at, lap = excluded.lap, "
            "verdict = excluded.verdict, detail = excluded.detail",
            [
                (errand, mid, lap, verdict, detail)
                for mid, (verdict, detail) in verdicts.items()
            ],
        )

    def population_size(self, errand: Errand, cap: int | None = None) -> int:
        """How many objects this errand still cares about.

        This is a full scan of the errand's domain with its predicate
        applied per row — 3.4s on a six-figure library — so it is NEVER on
        the patrol's hot path.  Status callers pass a cap, which
        short-circuits the scan and turns the answer into "at least cap".
        """
        dom = DOMAINS.get(errand.domain, DOMAIN_MEMORIES)
        where = dom.where
        if errand.population:
            where += f" AND ({errand.population})"
        if cap is not None:
            row = self._db.execute(
                f"SELECT COUNT(*) AS n FROM "
                f"(SELECT 1 FROM {dom.table} WHERE {where} LIMIT ?)",
                (cap,),
            ).fetchone()
        else:
            row = self._db.execute(
                f"SELECT COUNT(*) AS n FROM {dom.table} WHERE {where}"
            ).fetchone()
        return row["n"] if row else 0

    def _lap_complete(
        self, errand: Errand, lap: int, head_lap: int | None, examined: bool,
    ) -> bool:
        """Whether this errand has now walked everything it cares about.

        Two ways a lap finishes, and neither costs a scan:

        - The queue emptied.  A repair errand removes memories from its own
          population as it fixes them, so finishing looks like nothing left
          to select.  Only counts if the errand has actually stamped
          something at this lap, otherwise an errand with nothing to do
          would tick its lap forever.
        - The queue wrapped.  Selection is least-recently-visited first, so
          if the *head* of the queue already carries the current lap, so
          does everything behind it.
        """
        if not examined:
            row = self._db.execute(
                "SELECT 1 FROM patrol_visits WHERE errand = ? AND lap >= ? LIMIT 1",
                (errand.name, lap),
            ).fetchone()
            return row is not None
        return head_lap is not None and head_lap >= lap

    # ── Deposits ────────────────────────────────────────────────────────

    def observe(
        self,
        errand: str,
        kind: str,
        subject: str,
        detail: str,
        weight: float = 0.0,
        memory_ids: list[str] | None = None,
        lap: int = 0,
        data: dict | None = None,
    ) -> None:
        """Write down something the walk noticed.

        Authored, not derived: this survives the maintenance that prunes
        the co-occurrence, canopy and tensor caches, because a thing
        somebody looked at and saw must not quietly age out.

        Seeing the same subject again on a later lap CONFIRMS it — the row
        is updated in place and its confirmation count rises — so a point
        of reference gains weight by being seen repeatedly rather than
        accumulating duplicates.
        """
        merged = dict(data or {})
        if data:
            # A walk sees a batch at a time, so a measure taken over the
            # whole corpus has to accumulate here rather than be recomputed.
            # List values union; everything else takes the newer reading.
            row = self._db.execute(
                "SELECT data, lap FROM patrol_observations "
                "WHERE errand = ? AND kind = ? AND subject = ?",
                (errand, kind, subject),
            ).fetchone()
            # Counting is scoped to a LAP.  A walk sees one batch at a time,
            # so a corpus-wide total has to be accumulated across batches —
            # but a lap visits every object exactly once, so carrying the
            # total into the NEXT lap counts everything a second time.
            #
            # Measured: after six laps, `concept_gap.mentions` stood at
            # almost exactly six times the truth for every term checked —
            # `university` 1,318 against 225, `press` 1,759 against 295,
            # `light` 2,437 against 406. Uniform, and therefore invisible in
            # any ratio built from it, which is why it survived so long.
            same_lap = bool(row) and row["lap"] == lap
            if row and row["data"]:
                try:
                    prior = json.loads(row["data"])
                except (json.JSONDecodeError, TypeError, ValueError):
                    prior = {}
                for k, v in prior.items():
                    cur = merged.get(k)
                    if k in _STAMP_KEYS:
                        # Never merged: see _STAMP_KEYS.  The newer stamp
                        # is the only one that describes this reading.
                        continue
                    if isinstance(v, list) and isinstance(cur, list):
                        try:
                            merged[k] = sorted(set(v) | set(cur))
                        except TypeError:
                            # A list of records rather than of names: the
                            # union is undefined and the newer reading is
                            # the one that counts.  `documents` accumulates
                            # because a later pass sees different files;
                            # a measurement like a twin candidate is simply
                            # re-taken, and merging two of them would
                            # invent a state neither pass observed.
                            pass
                    elif (same_lap and isinstance(v, dict)
                          and isinstance(cur, dict)):
                        # Counters sum: a region is built a batch at a
                        # time, and how often two concepts share a memory
                        # is only meaningful over the whole corpus.
                        totals = dict(v)
                        for kk, vv in cur.items():
                            totals[kk] = totals.get(kk, 0) + vv
                        merged[k] = totals
                    elif (same_lap and k in _ACCUMULATING_KEYS
                          and isinstance(v, (int, float))
                          and isinstance(cur, (int, float))):
                        merged[k] = v + cur
                    elif k not in merged:
                        merged[k] = v
        self._db.execute(
            "INSERT INTO patrol_observations "
            "(errand, kind, subject, detail, weight, memory_ids, lap, actor, data) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(errand, kind, subject) DO UPDATE SET "
            "detail = excluded.detail, weight = excluded.weight, "
            "memory_ids = excluded.memory_ids, lap = excluded.lap, "
            "data = excluded.data, "
            "confirmations = patrol_observations.confirmations + 1, "
            "retracted = 0, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')",
            (errand, kind, subject, detail, weight,
             json.dumps(sorted(memory_ids or [])), lap, f"patrol:{errand}",
             json.dumps(merged, sort_keys=True)),
        )

    def observations(
        self, kind: str | None = None, limit: int = 50,
    ) -> list[dict]:
        """What the patrol has noticed, heaviest first."""
        sql = ("SELECT * FROM patrol_observations WHERE retracted = 0")
        params: list = []
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY weight ASC, confirmations DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self._db.execute(sql, tuple(params)).fetchall()]

    def tags_for(self, memory_ids: list[str]) -> dict[str, list[str]]:
        """Tags on these memories, batched — one query, not one per memory."""
        if not memory_ids:
            return {}
        ph = ",".join("?" * len(memory_ids))
        out: dict[str, list[str]] = {}
        for r in self._db.execute(
            f"SELECT mt.memory_id, t.name FROM memory_tags mt "
            f"JOIN tags t ON t.id = mt.tag_id "
            f"WHERE mt.memory_id IN ({ph})",
            tuple(memory_ids),
        ).fetchall():
            out.setdefault(r["memory_id"], []).append(r["name"])
        return out

    def build_observation_edges(
        self, kind: str = "concept_gap", min_confirmations: int = 2,
    ) -> dict:
        """Connect concepts that share MEMORIES, not merely documents.

        The document was the wrong unit and the Zohar proved it: one file
        of 8,969 chunks, so "both appear in the Zohar" says almost
        nothing.  Capped, the entire Kabbalah vocabulary had no edges at
        all; uncapped, that one book became a 1,563-node clique.  Two
        terms in the same chunk are genuinely related.

        Weight is Jaccard over the memories that discuss each — shared
        over the union of their mentions.
        """
        import itertools
        import json as _json

        rows = self._db.execute(
            "SELECT subject, data FROM patrol_observations "
            "WHERE kind = ? AND retracted = 0 AND confirmations >= ?",
            (kind, min_confirmations),
        ).fetchall()
        mentions: dict[str, int] = {}
        cooc: dict[str, dict] = {}
        for r in rows:
            try:
                data = _json.loads(r["data"])
            except (ValueError, TypeError):
                continue
            if data.get("co"):
                cooc[r["subject"]] = data["co"]
                mentions[r["subject"]] = max(1, int(data.get("mentions", 1)))

        pairs: dict[tuple[str, str], int] = {}
        for a, partners in cooc.items():
            for b, n in partners.items():
                if b not in cooc or a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                # Each side counted it, so keep the larger rather than
                # doubling: a saw b in n memories, b saw a in the same n.
                pairs[key] = max(pairs.get(key, 0), int(n))

        self._db.execute("DELETE FROM observation_edges WHERE kind = ?", (kind,))
        written = 0
        batch = []
        for (a, b), shared in pairs.items():
            union = mentions.get(a, 1) + mentions.get(b, 1) - shared
            weight = round(shared / union, 4) if union > 0 else 0.0
            if weight < self._config.gap_edge_min_weight:
                continue
            batch.append((kind, a, b, shared, weight))
        if batch:
            self._db.executemany(
                "INSERT OR REPLACE INTO observation_edges "
                "(kind, subject_a, subject_b, shared, weight) VALUES (?, ?, ?, ?, ?)",
                batch,
            )
            written = len(batch)
        return {"nodes": len(cooc), "edges": written}

    def gap_neighbors(
        self, subject: str, kind: str = "concept_gap", limit: int = 12,
    ) -> list[dict]:
        """What else the same material raises — one step out in the net."""
        rows = self._db.execute(
            "SELECT subject_b AS other, shared, weight FROM observation_edges "
            "WHERE kind = ? AND subject_a = ? "
            "UNION ALL "
            "SELECT subject_a AS other, shared, weight FROM observation_edges "
            "WHERE kind = ? AND subject_b = ? "
            "ORDER BY weight DESC, shared DESC LIMIT ?",
            (kind, subject, kind, subject, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def gap_clusters(
        self, kind: str = "concept_gap", min_size: int = 3, limit: int = 20,
    ) -> list[dict]:
        """Connected components of the net — coherent missing subjects.

        A list of unfindable words is a nuisance; a component of them is a
        finding.  "malchut, chochmah, godhead, tabernacle" together say
        the Kabbalah vocabulary is unindexed, which is one problem with
        one fix rather than four.
        """
        adj: dict[str, set] = {}
        for r in self._db.execute(
            "SELECT subject_a, subject_b FROM observation_edges WHERE kind = ?",
            (kind,),
        ).fetchall():
            adj.setdefault(r["subject_a"], set()).add(r["subject_b"])
            adj.setdefault(r["subject_b"], set()).add(r["subject_a"])

        seen: set = set()
        out: list[dict] = []
        for node in adj:
            if node in seen:
                continue
            comp, stack = set(), [node]
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                comp.add(n)
                stack.extend(adj[n] - seen)
            if len(comp) >= min_size:
                out.append({"size": len(comp), "members": sorted(comp)})
        out.sort(key=lambda c: -c["size"])
        return out[:limit]

    # ── Running ─────────────────────────────────────────────────────────

    def run_errand(
        self,
        name: str,
        *,
        batch_size: int | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict:
        errand = ERRANDS.get(name)
        if errand is None:
            _log.warning("Unknown wander errand %r — skipped", name)
            return {"errand": name, "error": "unknown"}

        record = self._state.get(name)
        if not record.enabled:
            return {"errand": name, "skipped": "disabled"}

        limit = batch_size or self._config.wander_batch_size
        selected = self.select_with_laps(errand, limit)
        head_lap = selected[0][1] if selected else None
        memory_ids = [mid for mid, _ in selected]

        verdicts: dict[str, tuple[str, str | None]] = {}
        if memory_ids:
            verdicts = errand.run(
                db=self._db,
                config=self._config,
                embedding=self._embedding,
                memory_ids=memory_ids,
                state=record,
                should_stop=should_stop,
                patrol=self,
            )
            self.record_visits(name, record.lap, verdicts)

        acted = sum(1 for v, _ in verdicts.values() if v == "acted")
        failed = sum(1 for v, _ in verdicts.values() if v == "failed")
        lap = record.lap
        if verdicts:
            result = f"examined={len(verdicts)} acted={acted} failed={failed}"
            self._state.record_run(
                name, visited=len(verdicts), acted=acted, result=result,
            )
        if self._lap_complete(errand, record.lap, head_lap, bool(memory_ids)):
            lap = self._state.advance_lap(name)

        return {
            "errand": name,
            "examined": len(verdicts),
            "acted": acted,
            "failed": failed,
            "lap": lap,
        }

    def run_due(
        self,
        *,
        idle: bool,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict:
        """Run every configured errand whose interval has elapsed.

        Errands marked ``when="idle"`` are held back while the library is
        being used, so a walk that reshapes retrieval geometry cannot do it
        underneath a patron's question.
        """
        if not self._lease.acquire():
            holder = self._lease.holder() or {}
            return {
                "idle": idle,
                "errands": [],
                "skipped": "lease held elsewhere",
                "lease_holder": holder.get("owner"),
            }
        try:
            return self._run_due_locked(idle=idle, should_stop=should_stop)
        finally:
            # Released rather than left to expire, so the next tick — very
            # likely a different process — can pick the walk straight up.
            self._lease.release()

    def _run_due_locked(
        self,
        *,
        idle: bool,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict:
        now = datetime.utcnow()
        interval = timedelta(minutes=self._config.wander_interval_minutes)
        results = []
        for name in self._config.wander_errands:
            errand = ERRANDS.get(name)
            if errand is None:
                _log.warning("Unknown wander errand %r — skipped", name)
                continue
            if errand.when == "idle" and not idle:
                continue
            last = self._last_run.get(name)
            if last is not None and (now - last) < interval:
                continue
            if should_stop is not None and should_stop():
                break
            self._last_run[name] = now
            try:
                results.append(self.run_errand(name, should_stop=should_stop))
            except Exception:
                _log.exception("Wander errand %r failed", name)
        return {"idle": idle, "errands": results}


# ── Errand: repair missing embeddings ───────────────────────────────────


def _repair_missing_embedding(
    *,
    db: DatabaseEngine,
    config: ChicoryConfig,
    embedding,
    memory_ids: list[str],
    state: WanderRecord,
    should_stop: Callable[[], bool] | None = None,
    **_: object,
) -> dict[str, tuple[str, str | None]]:
    """Re-embed memories that have no embedding row.

    A memory without an embedding is stored but not *held*: it cannot be
    reached semantically, only by tag, so it is effectively invisible to
    the retrieval path that matters.  The linear sweep has been filing
    these as ``missing_embedding`` findings and resolving none of them,
    because filing is all it can do.

    This is the safest possible first errand — it restores an invariant the
    system already asserts, is idempotent, and decides nothing.
    """
    from chicory.ingest.chunker import chunk_text_for_embedding

    verdicts: dict[str, tuple[str, str | None]] = {}
    if embedding is None:
        return {mid: ("skipped", "no embedding engine") for mid in memory_ids}

    for mid in memory_ids:
        if should_stop is not None and should_stop():
            break
        row = db.execute(
            "SELECT content FROM memories WHERE id = ?", (mid,),
        ).fetchone()
        if row is None:
            verdicts[mid] = ("skipped", "memory gone")
            continue
        content = row["content"] or ""
        if not content.strip():
            # Nothing to embed and nothing to fix — record it so the empty
            # memory stops returning to the head of the queue every batch.
            verdicts[mid] = ("skipped", "empty content")
            continue
        try:
            chunks = chunk_text_for_embedding(content)
            if len(chunks) <= 1:
                embedding.store_cached(mid, embedding.embed(content))
            else:
                embedding.store_chunks(mid, list(embedding.embed_batch(chunks)))
            verdicts[mid] = ("acted", f"chunks={max(1, len(chunks))}")
        except Exception as exc:  # noqa: BLE001 - one bad memory must not stop the lap
            # Another writer may have embedded this memory in the moment we
            # were working on it — the lease makes that rare, not
            # impossible, since ingest and re-embed write here too.  If the
            # embedding now exists the invariant is satisfied, whoever
            # satisfied it, and calling that a failure would rotate a
            # perfectly good memory to the back of the queue.
            if _has_embedding(db, mid):
                verdicts[mid] = ("ok", "embedded concurrently")
            else:
                _log.warning("Re-embed failed for %s: %s", mid, exc)
                verdicts[mid] = ("failed", str(exc)[:200])

    _resolve_findings(db, [m for m, (v, _) in verdicts.items() if v == "acted"])
    return verdicts


def _has_embedding(db: DatabaseEngine, memory_id: str) -> bool:
    return db.execute(
        "SELECT 1 FROM embeddings WHERE memory_id = ? LIMIT 1", (memory_id,),
    ).fetchone() is not None


def _resolve_findings(db: DatabaseEngine, memory_ids: list[str]) -> None:
    """Close the sweep findings this repair answers.

    Only ``missing_embedding`` findings, and only for memories that now
    actually have an embedding — the patrol resolves what it mechanically
    fixed, never what it merely visited.
    """
    if not memory_ids:
        return
    placeholders = ",".join("?" * len(memory_ids))
    db.execute(
        f"UPDATE sweep_findings SET resolved = 1 "
        f"WHERE finding_type = 'missing_embedding' AND resolved = 0 "
        f"AND memory_id IN ({placeholders}) "
        f"AND EXISTS (SELECT 1 FROM embeddings e WHERE e.memory_id = sweep_findings.memory_id)",
        tuple(memory_ids),
    )


register(Errand(
    name="missing_embedding",
    description="Re-embed memories that have no embedding row",
    run=_repair_missing_embedding,
    population=(
        "NOT EXISTS (SELECT 1 FROM embeddings e WHERE e.memory_id = m.id)"
    ),
    when="always",
))


# ── Errand: recover source provenance, and notice what the shelves hold ──

_CHUNK_RE = re.compile(r"^(?P<doc>.+?)\s*\[(?P<n>\d+)/(?P<total>\d+)\]")


def _source_provenance(
    *,
    db: DatabaseEngine,
    memory_ids: list[str],
    state: WanderRecord,
    patrol: "Patrol",
    should_stop: Callable[[], bool] | None = None,
    **_: object,
) -> dict[str, tuple[str, str | None]]:
    """Backfill source_path from the summary, and observe each document.

    Every document-derived memory in this library carries its origin in
    its summary — ``filename.pdf [327/546]`` — and nowhere else.  Not one
    of 123,907 memories had source_path set, so file-level provenance was
    parseable by eye and unqueryable by anything.  That is an open
    commission the patron filed, not a defect this errand invented.

    The repair is mechanical.  The OBSERVATION is the point: once a
    document's chunks are queryable, the walk can see the document whole —
    how many chunks it holds, over what range, and which are missing.
    That is a fact about the library that nothing previously recorded, and
    a wanderer is the only thing positioned to notice it, because it is
    the only thing that reads everything.
    """
    verdicts: dict[str, tuple[str, str | None]] = {}
    touched: set[str] = set()

    for mid in memory_ids:
        if should_stop is not None and should_stop():
            break
        row = db.execute(
            "SELECT summary FROM memories WHERE id = ?", (mid,),
        ).fetchone()
        if row is None:
            verdicts[mid] = ("skipped", "memory gone")
            continue
        m = _CHUNK_RE.match(row["summary"] or "")
        if m is None:
            # A summary with no chunk prefix carries no derivable origin.
            # Recorded so it leaves the queue rather than being re-read
            # every lap for a provenance it will never have.
            verdicts[mid] = ("skipped", "no chunk prefix in summary")
            continue
        doc = m.group("doc").strip()
        try:
            db.execute(
                "UPDATE memories SET source_path = ? WHERE id = ?", (doc, mid),
            )
        except Exception as exc:  # noqa: BLE001
            verdicts[mid] = ("failed", str(exc)[:200])
            continue
        touched.add(doc)
        verdicts[mid] = ("acted", f"source_path={doc[:60]}")

    for doc in touched:
        _observe_document(db, patrol, doc, state.lap)
    return verdicts


def _observe_document(
    db: DatabaseEngine, patrol: "Patrol", doc: str, lap: int,
) -> None:
    """Record how completely this library holds one source document."""
    rows = db.execute(
        "SELECT summary FROM memories WHERE source_path = ? AND is_archived = 0",
        (doc,),
    ).fetchall()
    seen, total = set(), 0
    for r in rows:
        m = _CHUNK_RE.match(r["summary"] or "")
        if m is None:
            continue
        seen.add(int(m.group("n")))
        total = max(total, int(m.group("total")))
    if not seen or total <= 0:
        return
    missing = sorted(set(range(1, total + 1)) - seen)
    coverage = len(seen) / total
    # "Located", not "held".  Coverage is computed over memories this walk
    # has already stamped with a source_path, so mid-lap a document looks
    # incomplete when the rest of it simply has not been reached yet.  The
    # upsert corrects the record as the walk proceeds, and only a gap that
    # survives a completed lap is a real gap — so the wording must not
    # claim more than the walk has actually established.
    detail = f"{len(seen)}/{total} chunks located"
    if not missing:
        detail += " — complete"
    else:
        shown = ", ".join(str(x) for x in missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        detail += f"; not yet located: {shown}{more}"
    patrol.observe(
        errand="source_provenance",
        kind="document_coverage",
        subject=doc,
        detail=detail,
        # Weight is coverage, and the index orders ASCENDING: the least
        # complete document is the one worth a person's attention.
        weight=round(coverage, 4),
        lap=lap,
    )


register(Errand(
    name="source_provenance",
    description="Backfill source_path from summaries; observe document coverage",
    run=_source_provenance,
    population=(
        "(m.source_path IS NULL OR m.source_path = '') "
        "AND m.summary LIKE '%[%/%]%'"
    ),
    when="always",
))


# ── Errand: the concepts this library discusses but cannot find ─────────

_CONCEPT_RE = re.compile(r"\b([A-Z][a-zA-Z]{4,})\b")

# Ingestion stamps and format labels: true of the memory, not of the
# concept, and identical across everything imported in the same minute.
# Numeric fields that ACCUMULATE across passes.  Everything else takes the
# newer reading.
#
# This was the other way round — sum any number — and it was silently
# corrupting every observation with a numeric field.  `tag_id` was being
# added to itself on each re-deposit, so a seam recorded against tag 69
# eventually pointed at tag 207, an unrelated tag that happened to sit at
# the running total; `held` became a sum of every measurement ever taken,
# which is how `example` came to claim 16,130 memories while holding 2.
# Forty of forty seams and 406 of 406 twins were affected.
#
# An identifier is not a quantity and a measurement is not a tally.  Only
# a field that genuinely counts occurrences over the whole corpus belongs
# here, because a walk sees one batch at a time and cannot total such a
# thing any other way.
_ACCUMULATING_KEYS = frozenset({"mentions"})

# Keys that stamp WHICH definition a measurement was taken under, rather
# than carrying a measurement.  These always take the newer reading: a
# union of two stamps describes a population neither pass ever looked at,
# and the stamp exists precisely so a reader can tell those apart.  The
# hazard is real because a row is reused across a retraction — a withdrawn
# observation keeps its data until the same subject is seen again — so the
# old membership is still sitting there when the new one is written.
_STAMP_KEYS = frozenset({
    "left_documents", "right_documents", "duplicates_excluded",
    "unresolved_registry_entries", "basis",
})

# Observation kinds anchored on a tag, and therefore invalidated when that
# tag is merged or deactivated.  The audit covered only `tag_seam` and kept
# it spotless while three other kinds rotted beside it: merging five
# duplicate tags left ten stale twin rows, four tag_content rows and one
# variant candidate, all naming tags that no longer exist.
_TAG_ANCHORED_KINDS = ("tag_seam", "tag_twin", "tag_gap_variant", "tag_content")

# Kinds that record a fact about something the producing errand may never
# look at again, and which are therefore exempt from the lap test.
#
# `source_provenance` is a REPAIR errand: it empties its own population as
# it fixes it, and `_observe_document` re-observes only documents a batch
# actually touched.  So once the last unattributed chunk is repaired the
# errand can never re-report any document, and its lap still advances —
# `_lap_complete` takes the empty-queue branch.  Measured on the live
# library: population 0, wander_state.lap 1, and all 122,532 visits plus
# all 1,979 observations at lap 0.  One newly ingested chunked document
# stamps a visit at lap 1, the next tick advances the lap to 2, and the
# lap test then reads `0 < 2 - 2 + 1` as true for every one of those
# 1,979 rows and retracts them as "the walk no longer reports it" — which
# is false.  The walk has nothing left to walk.  Nothing re-creates them.
#
# A lap that advanced over an empty queue is not evidence about anything.
_TERMINAL_KINDS = ("document_coverage", "revision_shadow")

_STRUCTURAL_TAG_PREFIXES = ("month-", "day-", "minute-", "hour-",
                            "week-", "year-")

# Capitalised common words, which a capitalisation heuristic cannot tell
# from proper nouns.  Sentence-initial "Great" and "Lord" look exactly
# like "Malchut" to a regex.
_CONCEPT_STOPWORDS = frozenset("""
    the this that these those when where which while what whom whose
    their there they then thus here however because since although also
    and but for not from with without into onto upon each every some
    such only very just like both other another same over under above
    below after before how why who one two three four five first second
    third next last chapter section figure table page note see may might
    must shall will would could should can cannot have has had been being
    were was are its his her him she you your our they them all any more
    most much many few less least own said say says according great good
    better best long high low new old well even still yet lord supreme
    thou thee hath unto come came therefore whereas within among between
    during through against about across part parts case cases form forms
    kind kinds way ways time times thing things world worlds life man men
    people place places point points work works right left style guide
    release english
""".split())


def _concept_gaps(
    *,
    db: DatabaseEngine,
    config: ChicoryConfig,
    memory_ids: list[str],
    state: WanderRecord,
    patrol: "Patrol",
    should_stop: Callable[[], bool] | None = None,
    **_: object,
) -> dict[str, tuple[str, str | None]]:
    """Build a dictionary of what the library discusses but cannot find.

    A retrieval score tells you a question was answered badly.  It cannot
    tell you why, and there are two why's that call for opposite responses:
    the shelves do not hold the material, or they hold it and nothing can
    reach it.  On this library `zohar` carries 15,205 memories while
    `tiferet` carries none — the Zohar is here, and Tiferet is unfindable
    inside it.

    Only the walk can notice that, because it is the only thing that reads
    the CONTENT rather than the index.  A term a memory returns to four or
    more times is something that memory is about; if no tag holds it, the
    library is discussing a concept it cannot retrieve by.

    The heuristic is capitalisation, which is why the stop-list above is
    long and the threshold is four: sentence-initial "Great" is
    indistinguishable from "Malchut" to a regex, and only recurrence
    separates them.
    """
    from collections import Counter

    verdicts: dict[str, tuple[str, str | None]] = {}
    seen: Counter = Counter()
    docs: dict[str, set] = {}
    # A concept's REGION: what shares its memories.  Co-occurrence within
    # one memory is the honest unit — two terms in the same chunk are
    # related, two terms in the same 8,969-chunk book are barely
    # acquainted, which is why a document-level net left the whole
    # Kabbalah vocabulary with no edges at all.
    co: dict[str, Counter] = {}
    around: dict[str, Counter] = {}

    tags_by_memory = patrol.tags_for(memory_ids) if memory_ids else {}

    for mid in memory_ids:
        if should_stop is not None and should_stop():
            break
        row = db.execute(
            "SELECT content, source_path FROM memories WHERE id = ?", (mid,),
        ).fetchone()
        if row is None:
            verdicts[mid] = ("skipped", "memory gone")
            continue
        hits = Counter(
            term.lower()
            for term in _CONCEPT_RE.findall(row["content"] or "")
            if term.lower() not in _CONCEPT_STOPWORDS
        )
        about = [t for t, n in hits.items() if n >= config.concept_min_mentions]
        if len(about) > config.concept_max_terms_per_memory:
            # A chunk about this many terms is a table, not prose: its
            # terms co-occur by layout rather than by meaning.  One CLDR
            # locale file put 870 language names in a single chunk and its
            # clique swamped the whole net.
            verdicts[mid] = ("skipped", f"tabular: {len(about)} terms")
            continue
        memory_tags = tags_by_memory.get(mid, [])
        for term in about:
            seen[term] += 1
            docs.setdefault(term, set()).add(row["source_path"] or "(unfiled)")
            partners = co.setdefault(term, Counter())
            for other in about:
                if other != term:
                    partners[other] += 1
            # The tags on the memories that discuss it: the bridge from an
            # unfindable concept back into the index that cannot find it.
            near = around.setdefault(term, Counter())
            for tag in memory_tags:
                # Letter tags carry no subject, and neither does the clock.
                # This counter is capped at twelve keys on deposit, and on
                # the live library six of chochmah's fourteen were stamps
                # like `day-2026-07-17` and `minute-2026-07-17t19-30` — so
                # half the width was spent recording an import batch.
                # Worse, similarity computed over them measures WHEN two
                # concepts were ingested rather than what they are about.
                if len(tag) > 1 and not tag.startswith(_STRUCTURAL_TAG_PREFIXES):
                    near[tag] += 1
        verdicts[mid] = ("ok", f"concepts={len(about)}")

    deposited = 0
    for term, mentions in seen.items():
        row = db.execute(
            "SELECT t.id, COUNT(mt.memory_id) AS n FROM tags t "
            "LEFT JOIN memory_tags mt ON mt.tag_id = t.id "
            "WHERE t.name = ? GROUP BY t.id",
            (term,),
        ).fetchone()
        held = row["n"] if row else 0
        if held >= config.faq_gap_thin_tag_memories:
            # Findable now — so retract the standing observation instead of
            # merely declining to refresh it.  Skipping leaves the old row
            # at retracted=0 with frozen counts, which is how `malchut`
            # ended up still listed as a gap at mentions=3,948 while its
            # tag held 993 memories: a live node in the net, carrying stale
            # numbers, re-proposed forever and never leaving the desk.
            db.execute(
                "UPDATE patrol_observations SET retracted = 1, "
                "detail = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') "
                "WHERE kind = 'concept_gap' AND subject = ? AND retracted = 0",
                (f"closed: a tag now holds {held} memories", term),
            )
            continue
        term_docs = sorted(docs.get(term, set()))
        # Spread, not recurrence.  Recurrence within a memory cannot tell a
        # concept from a citation convention: on this corpus "Press",
        # "University" and "Cambridge" recur four-plus times per chunk in
        # every bibliography and topped the first dictionary built that
        # way.  What separates them is how far they range — vocabulary
        # appears across the whole library, a concept concentrates in the
        # few documents that are about it.  Low is concentrated, so the
        # ASC ordering surfaces concepts and buries furniture.
        spread = round(len(term_docs) / max(mentions, 1), 4)
        patrol.observe(
            errand="concept_gaps",
            kind="concept_gap",
            subject=term,
            detail=(
                f"discussed in {mentions} memories across {len(term_docs)} "
                f"document(s) this pass; "
                + ("no tag holds it" if row is None else f"its tag holds {held}")
                + " — the library talks about this and cannot retrieve by it"
            ),
            weight=spread,
            data={
                "documents": term_docs,
                "mentions": mentions,
                # Capped: a region is a neighbourhood, not a census, and
                # an uncapped counter on 9,551 concepts is a megabyte of
                # JSON nobody reads.
                "co": dict(co.get(term, Counter()).most_common(24)),
                "around": dict(around.get(term, Counter()).most_common(12)),
            },
            lap=state.lap,
        )
        deposited += 1
    if deposited:
        _log.info("Concept gaps: %d recorded this pass", deposited)
    return verdicts


def _meta_shelf_proposals(
    *,
    db: DatabaseEngine,
    config: ChicoryConfig,
    memory_ids: list[str],
    state: WanderRecord,
    patrol: "Patrol",
    should_stop: Callable[[], bool] | None = None,
    **_: object,
) -> dict[str, tuple[str, str | None]]:
    """Assemble the gap net into candidate subjects, once per lap.

    A whole-net pass rather than a per-memory one, so it ignores the batch
    it was handed and reports the walk's own footsteps as visited.  It runs
    at most once per lap: the net only changes when concept_gaps has
    deposited more, and re-running it inside a lap re-derives the same
    groups at real cost.

    Nothing here decides anything.  Every group lands as a PROPOSAL for the
    meta_shelves desk, because the junk filter provably cannot finish the
    job — a game's config file scores docJ 0.948 against a real subject's
    0.55.
    """
    from chicory.layer4.meta_shelves import MetaShelfProposer

    marker = f"lap{state.lap}"
    if (state.cursor_hint or "") == marker:
        return {mid: ("skipped", "already proposed this lap")
                for mid in memory_ids}
    # Checked BEFORE the pass rather than inside it: this one is
    # all-or-nothing over the whole gap net (measured ~40s), and the patrol
    # phase holds the foreground lock, which every patron handler acquires
    # with a bare blocking `with` and no timeout.  cursor_hint is left
    # unset so a bail is re-attempted rather than recorded as done.
    if should_stop is not None and should_stop():
        return {mid: ("skipped", "a patron is waiting") for mid in memory_ids}
    result = MetaShelfProposer(config, db).propose()
    db.execute(
        "UPDATE wander_state SET cursor_hint = ?, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') WHERE errand = ?",
        (marker, "meta_shelf_proposals"),
    )
    db.connection.commit()
    _log.info("Meta-shelf proposals: %d accepted of %d groups",
              result.get("accepted", 0), result.get("groups", 0))
    return {mid: ("ok", f"proposed={result.get('proposed', 0)}")
            for mid in memory_ids}


register(Errand(
    name="meta_shelf_proposals",
    description="Assemble concept gaps into candidate subjects for the desk",
    run=_meta_shelf_proposals,
    # Reads content for the bibliography guard and walks the whole net, so
    # it waits for quiet like the errand whose output it consumes.
    when="idle",
))


# ── Metaspace errands ───────────────────────────────────────────────────
#
# The walk above reads the library's MATERIAL.  These read its INDEX.
#
# A survey found 844 tags with no edge (so the roster could not see them —
# `plato`, 1,262 memories, had no shelf), 392 with no centroid, and 520
# fragment tags duplicating a hyphenated parent.  None of it was visible
# from inside normal use: retrieval worked, the shelves rendered, nothing
# errored.  A one-off query finds that once.  A walk keeps finding it, as
# tags keep arriving.


def _tag_seams(
    *,
    db: DatabaseEngine,
    config: ChicoryConfig,
    memory_ids: list[str],
    state: WanderRecord,
    patrol: "Patrol",
    should_stop: Callable[[], bool] | None = None,
    **_: object,
) -> dict[str, tuple[str, str | None]]:
    """Notice where a tag has fallen out of the structure around it.

    Four things can be true of a tag that make it unreachable, and each is
    invisible from the outside:

      no centroid  _topical_tag_ids selects FROM tag_centroids, so the
                   roster cannot even consider it.
      no edge      components are built from centroid_edges, which form
                   through co-RETRIEVAL — so a tag nobody has queried is an
                   isolated node and gets dropped for being under
                   subject_min_cluster_size.
      no shelf     it derives nothing, so nothing routes to it.
      redundant    its memories are a subset of a hyphenated parent's, so
                   it is the same tag twice and inflates co-occurrence.

    Recording, not repairing.  Seeding an edge changes what the roster
    derives and merging a tag deletes rows; neither belongs in an errand
    that runs unattended.
    """
    ids = [int(x) for x in memory_ids if str(x).isdigit()]
    if not ids:
        return {}
    ph = ",".join("?" * len(ids))
    rows = db.execute(
        f"""SELECT t.id, t.name,
               (SELECT COUNT(*) FROM memory_tags mt WHERE mt.tag_id = t.id) AS held,
               EXISTS(SELECT 1 FROM tag_centroids c WHERE c.tag_id = t.id) AS has_centroid,
               EXISTS(SELECT 1 FROM centroid_edges e
                      WHERE (e.tag_a_id = t.id OR e.tag_b_id = t.id)
                        AND e.edge_strength >= ?
                        AND e.co_retrieval_count >= ?) AS has_edge
            FROM tags t WHERE t.id IN ({ph})""",
        (config.subject_min_edge_strength,
         config.subject_edge_merge_min_count, *ids),
    ).fetchall()

    shelved: set[int] = set()
    for r in db.execute(
        "SELECT member_tag_ids FROM subjects WHERE status = 'active'"
    ):
        try:
            shelved.update(json.loads(r["member_tag_ids"]))
        except (ValueError, TypeError):
            continue

    live = db.execute(
        "SELECT COUNT(*) AS n FROM memories WHERE is_archived = 0"
    ).fetchone()
    hub_cap = config.subject_hub_tag_max_fraction * max(
        (live["n"] if live else 0), 1)

    verdicts: dict[str, tuple[str, str | None]] = {}
    for r in rows:
        if should_stop is not None and should_stop():
            break
        key = str(r["id"])
        name, held = r["name"], r["held"]
        if name.startswith(_STRUCTURAL_TAG_PREFIXES):
            verdicts[key] = ("skipped", "a clock stamp, not a subject")
            continue
        if held == 0:
            verdicts[key] = ("ok", "holds nothing")
            continue
        if held > hub_cap:
            # `document` sits on 123,798 memories and `pdf` on 100,530.
            # Being off the shelf tree is what subject_hub_tag_max_fraction
            # is FOR — a tag on a fifth of the library connects everything
            # to everything — so this is the rule working, not a seam.
            verdicts[key] = ("ok", f"a hub tag ({held:,}), excluded by design")
            continue

        faults = []
        if not r["has_centroid"]:
            faults.append("no centroid")
        if not r["has_edge"]:
            # "No edge the ROSTER would walk", not "no edge".  The
            # hysteresis keeps an edge below subject_edge_merge_min_count
            # only when both ends shared a prior root, so a tag that has
            # never been shelved needs a qualifying edge to get in at all.
            # Measured: `archaeology` (1,659 memories) carries seven edges,
            # every one at co_retrieval_count 1 or 2, and is invisible to
            # the roster despite looking connected.
            faults.append("no qualifying edge")
        if r["id"] not in shelved:
            faults.append("on no shelf")
        parent = _redundant_parent(db, r["id"], name)
        if parent:
            faults.append(f"duplicates {parent}")

        if not faults:
            verdicts[key] = ("ok", f"placed, {held} memories")
            continue
        patrol.observe(
            errand="tag_seams",
            kind="tag_seam",
            subject=name,
            detail=f"{held} memories - " + "; ".join(faults),
            # Ascending order surfaces the least-affecting first, matching
            # every other observation kind; the desk sorts by held instead.
            weight=round(1.0 / (1.0 + held), 6),
            data={"tag_id": r["id"], "held": held, "faults": faults},
            lap=state.lap,
        )
        verdicts[key] = ("acted", "; ".join(faults))
    return verdicts


def _redundant_parent(db: DatabaseEngine, tag_id: int, name: str) -> str | None:
    """A hyphenated tag whose memories are a superset of this one's.

    Containment, not name shape.  A fragment that carries memories its
    parent lacks is a real tag that merely looks like a piece of one —
    `data` holds 4,103 memories outside `lab-data`, and `philosophy`
    17,636 outside `political-philosophy`.
    """
    row = db.execute(
        "SELECT id, name FROM tags WHERE is_active = 1 AND name LIKE ? "
        "AND name != ? AND instr('-' || name || '-', ?) > 0 "
        "ORDER BY LENGTH(name) DESC LIMIT 4",
        (f"%{name}%", name, f"-{name}-"),
    ).fetchall()
    for r in row:
        outside = db.execute(
            "SELECT COUNT(*) AS n FROM memory_tags a WHERE a.tag_id = ? "
            "AND NOT EXISTS(SELECT 1 FROM memory_tags b "
            "WHERE b.memory_id = a.memory_id AND b.tag_id = ?)",
            (tag_id, r["id"]),
        ).fetchone()
        if outside and outside["n"] == 0:
            return r["name"]
    return None


register(Errand(
    name="tag_seams",
    description="Notice tags that have fallen out of the structure around them",
    run=_tag_seams,
    domain="tags",
    # Cheap per tag and it touches no material, so it is safe beside a
    # conversation.
    when="always",
))


def _tag_twins(
    *,
    db: DatabaseEngine,
    config: ChicoryConfig,
    memory_ids: list[str],
    state: WanderRecord,
    patrol: "Patrol",
    should_stop: Callable[[], bool] | None = None,
    **_: object,
) -> dict[str, tuple[str, str | None]]:
    """Propose tags that may be one tag written twice.

    The concept-level version of this found `gevurah` and `gvurah`, which
    share zero memories and could only ever be joined by spelling.  At tag
    level the same trick finds transliteration pairs and plurals — and it
    CANNOT be trusted to act, because `device-dependent` and
    `device-independent` sit at ratio 0.941 while meaning opposite things,
    and `10-26-17-50mm-gdacca-pc-stir-2` and `-stir-3` are two experiments
    rather than one misspelt.

    So it records a candidate with both numbers a reader needs — how alike
    the names are, and how far the memories overlap — and decides nothing.
    An overlap near 1.0 is the same tag twice; near 0.0 with plural
    morphology is a judgment about vocabulary, not a typo.
    """
    ids = [int(x) for x in memory_ids if str(x).isdigit()]
    if not ids:
        return {}
    ratio_min = config.meta_shelf_variant_ratio
    ph = ",".join("?" * len(ids))
    # Every id the cursor handed over gets a verdict, whatever happens
    # below.  An object an errand examines and does not stamp returns to
    # the head of the queue forever: measured, a batch of short-named tags
    # kept this errand at 94% coverage after 72 runs, because the cursor
    # kept re-offering the same rows it had never recorded a visit for.
    verdicts: dict[str, tuple[str, str | None]] = {
        str(i): ("skipped", "too short to compare") for i in ids
    }
    mine = db.execute(
        f"SELECT id, name FROM tags WHERE id IN ({ph}) AND is_active = 1 "
        f"AND LENGTH(name) > 3", tuple(ids),
    ).fetchall()
    if not mine:
        return verdicts

    others = db.execute(
        "SELECT id, name FROM tags WHERE is_active = 1 AND LENGTH(name) > 3"
    ).fetchall()
    pool = [(r["id"], r["name"]) for r in others
            if not r["name"].startswith(_STRUCTURAL_TAG_PREFIXES)]

    matcher = SequenceMatcher(autojunk=False)
    for r in mine:
        if should_stop is not None and should_stop():
            break
        key, name = str(r["id"]), r["name"]
        if name.startswith(_STRUCTURAL_TAG_PREFIXES):
            verdicts[key] = ("skipped", "a clock stamp")
            continue
        matcher.set_seq2(name)
        found = []
        for oid, other in pool:
            if oid == r["id"] or abs(len(other) - len(name)) > 2:
                continue
            matcher.set_seq1(other)
            if matcher.real_quick_ratio() < ratio_min \
                    or matcher.quick_ratio() < ratio_min:
                continue
            ratio = matcher.ratio()
            if ratio < ratio_min:
                continue
            found.append((round(ratio, 4), oid, other))
        if not found:
            verdicts[key] = ("ok", "no near name")
            continue
        found.sort(reverse=True)
        twins = []
        for ratio, oid, other in found[:4]:
            counts = db.execute(
                "SELECT (SELECT COUNT(*) FROM memory_tags WHERE tag_id = ?) AS a, "
                "(SELECT COUNT(*) FROM memory_tags WHERE tag_id = ?) AS b, "
                "(SELECT COUNT(*) FROM memory_tags x JOIN memory_tags y "
                " ON y.memory_id = x.memory_id WHERE x.tag_id = ? AND y.tag_id = ?) AS s",
                (r["id"], oid, r["id"], oid),
            ).fetchone()
            union = counts["a"] + counts["b"] - counts["s"]
            twins.append({
                "other": other, "other_id": oid, "ratio": ratio,
                "overlap": round(counts["s"] / union, 4) if union else 0.0,
                "held": counts["a"], "other_held": counts["b"],
            })
        best = max(twins, key=lambda x: x["overlap"])
        patrol.observe(
            errand="tag_twins",
            kind="tag_twin",
            subject=name,
            detail=(
                f"resembles {best['other']} (names {best['ratio']}, memories "
                f"overlap {best['overlap']}) - "
                + ("almost certainly the same tag twice"
                   if best["overlap"] > 0.8 else
                   "same-looking names on different memories; a vocabulary "
                   "call, not a typo")
            ),
            # Highest overlap first under the ascending convention.
            weight=round(1.0 - best["overlap"], 6),
            data={"tag_id": r["id"], "twins": twins},
            lap=state.lap,
        )
        verdicts[key] = ("acted", f"{len(twins)} candidate(s)")
    return verdicts


register(Errand(
    name="tag_twins",
    description="Propose tags that may be one tag written two ways",
    run=_tag_twins,
    domain="tags",
    # Compares each tag against the whole vocabulary, so it waits for quiet.
    when="idle",
))


def _seam_audit(
    *,
    db: DatabaseEngine,
    config: ChicoryConfig,
    memory_ids: list[str],
    state: WanderRecord,
    patrol: "Patrol",
    should_stop: Callable[[], bool] | None = None,
    **kwargs: object,
) -> dict[str, tuple[str, str | None]]:
    """Walk the walk's own findings and keep them honest.

    An observation is a claim about the library made at a moment, and the
    library moves underneath it.  Nothing retracted them: `malchut` sat at
    retracted=0 claiming 3,948 mentions while its tag held 993, and within
    an hour of merging seven duplicate tags, 22 standing seams pointed at
    tags that no longer existed.  A record nobody audits becomes a record
    nobody can trust, and the desk that reads it inherits the rot.

    Three outcomes per seam, and only one of them changes the library:

      retract  the tag is gone, or every fault it named has cleared.
      repair   the fault is "no centroid" and the memories are embedded.
               A centroid is a pure function of what the tag already
               holds — recomputing it asserts nothing new, which is what
               makes it safe for an unattended errand, in the same way
               that backfilling a missing embedding is.
      restate  the fault list is re-measured and written back.

    It will not seed an edge, merge a tag, or place a shelf.  Those change
    what the roster DERIVES, and the survey that produced these seams is
    also the evidence for why: `implementation` and `implementing` look
    like one tag and share a co-tag context of 0.003.
    """
    import json as _json

    ids = [int(x) for x in memory_ids if str(x).isdigit()]
    if not ids:
        return {}
    verdicts: dict[str, tuple[str, str | None]] = {
        str(i): ("skipped", "not an auditable observation") for i in ids
    }
    ph = ",".join("?" * len(ids))
    rows = db.execute(
        f"SELECT id, kind, errand, subject, lap, data FROM patrol_observations "
        f"WHERE id IN ({ph}) AND retracted = 0",
        tuple(ids),
    ).fetchall()
    if not rows:
        return verdicts
    errand_lap = {r["errand"]: r["lap"] for r in db.execute(
        "SELECT errand, lap FROM wander_state")}
    # The latest lap at which each errand actually deposited something.
    # `wander_state.lap` says how far the cursor has gone, which is not the
    # same as the errand having run and reported: an errand whose queue is
    # empty advances its lap without producing anything, and reading that
    # as "it walked past this and stopped reporting it" is how the audit
    # came to threaten 1,979 records it had no evidence against.
    reported_lap = {r["errand"]: r["lap"] for r in db.execute(
        "SELECT errand, MAX(lap) AS lap FROM patrol_observations "
        "WHERE retracted = 0 GROUP BY errand")}

    live = db.execute(
        "SELECT COUNT(*) AS n FROM memories WHERE is_archived = 0").fetchone()
    hub_cap = config.subject_hub_tag_max_fraction * max(
        (live["n"] if live else 0), 1)
    shelved: set[int] = set()
    for r in db.execute(
        "SELECT member_tag_ids FROM subjects WHERE status = 'active'"
    ):
        try:
            shelved.update(_json.loads(r["member_tag_ids"]))
        except (ValueError, TypeError):
            continue

    centroids = None
    embedding = kwargs.get("embedding")
    if embedding is not None:
        try:
            from chicory.layer3.centroid_subgraph import CentroidSubgraph
            centroids = CentroidSubgraph(config, db, embedding)
        except Exception:          # pragma: no cover - optional repair path
            centroids = None

    def retract(obs_id: int, why: str) -> None:
        db.execute(
            "UPDATE patrol_observations SET retracted = 1, detail = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') WHERE id = ?",
            (f"resolved: {why}", obs_id),
        )

    for r in rows:
        if should_stop is not None and should_stop():
            break
        key = str(r["id"])
        try:
            data = _json.loads(r["data"] or "{}")
        except (ValueError, TypeError):
            data = {}
        tag_id = data.get("tag_id")
        # The one test available to every kind, including those with no
        # tag to look up: did the errand still find this on its last lap?
        current = errand_lap.get(r["errand"])
        reported = reported_lap.get(r["errand"])
        stale_by_lap = (
            current is not None
            and r["lap"] < current - config.audit_stale_laps + 1
            # Two things the bare lap number does not establish, and
            # without which "it stopped reporting this" is not a claim the
            # record supports.  The errand must be one that re-derives its
            # findings by walking (see _TERMINAL_KINDS), and it must have
            # reported SOMETHING since — an errand that has deposited
            # nothing at a later lap has not demonstrated it would have
            # re-reported this one.
            and r["kind"] not in _TERMINAL_KINDS
            and reported is not None and reported > r["lap"]
        )
        if stale_by_lap:
            retract(r["id"],
                    f"the walk no longer reports it (last seen on lap "
                    f"{r['lap']}, now on {current}, and it has reported "
                    f"as recently as lap {reported})")
            verdicts[key] = ("acted", "retracted: no longer found")
            continue

        if r["kind"] not in _TAG_ANCHORED_KINDS:
            # Nothing further is cheap to check.  Re-deriving a corpus
            # asymmetry or a bilingual alignment means scanning whole
            # populations, which is precisely what that errand's own lap
            # does — so the lap test above is the whole of the audit here.
            verdicts[key] = ("ok", "current as of the last lap")
            continue

        tag = db.execute(
            "SELECT id, name, is_active, merged_into FROM tags WHERE id = ?",
            (tag_id,),
        ).fetchone() if tag_id else None

        if tag is None:
            retract(r["id"], "the tag no longer exists")
            verdicts[key] = ("acted", "retracted: tag gone")
            continue
        # A pairwise finding dies when EITHER end does.  Merging five
        # duplicate tags left ten twin rows standing: five whose subject
        # was merged away, and five more still naming the merged tag as
        # the thing their subject resembles.
        partner = None
        if r["kind"] in ("tag_twin", "tag_gap_variant"):
            others = data.get("twins") or data.get("variants") or []
            partner = (others[0].get("other") or others[0].get("gap"))                 if others else None
        if partner is not None:
            alive = db.execute(
                "SELECT 1 FROM tags WHERE name = ? AND is_active = 1 "
                "UNION SELECT 1 FROM patrol_observations WHERE kind = "
                "'concept_gap' AND subject = ? AND retracted = 0",
                (partner, partner),
            ).fetchone()
            if alive is None:
                retract(r["id"], f"the other side, {partner!r}, is gone")
                verdicts[key] = ("acted", "retracted: partner gone")
                continue
        if not tag["is_active"]:
            into = tag["merged_into"]
            retract(r["id"], f"the tag was merged into {into}" if into
                    else "the tag was deactivated")
            verdicts[key] = ("acted", "retracted: tag merged away")
            continue

        held = db.execute(
            "SELECT COUNT(*) AS n FROM memory_tags WHERE tag_id = ?", (tag_id,)
        ).fetchone()["n"]
        if held == 0 or held > hub_cap:
            retract(r["id"], "holds nothing" if held == 0
                    else f"a hub tag ({held:,}), off the tree by design")
            verdicts[key] = ("acted", "retracted: no longer applicable")
            continue

        if r["kind"] != "tag_seam":
            # The other kinds have no fault list to re-measure; both ends
            # existing is the whole of their claim.
            verdicts[key] = ("ok", "both ends still exist")
            continue

        has_centroid = db.execute(
            "SELECT 1 FROM tag_centroids WHERE tag_id = ?", (tag_id,)
        ).fetchone() is not None
        repaired = False
        if not has_centroid and centroids is not None:
            try:
                if centroids.rebuild_centroid_for_tag(tag_id):
                    has_centroid, repaired = True, True
            except Exception:      # pragma: no cover - repair is best effort
                pass

        has_edge = db.execute(
            "SELECT 1 FROM centroid_edges WHERE (tag_a_id = ? OR tag_b_id = ?) "
            "AND edge_strength >= ? AND co_retrieval_count >= ?",
            (tag_id, tag_id, config.subject_min_edge_strength,
             config.subject_edge_merge_min_count),
        ).fetchone() is not None

        faults = []
        if not has_centroid:
            faults.append("no centroid")
        if not has_edge:
            faults.append("no qualifying edge")
        if tag_id not in shelved:
            faults.append("on no shelf")
        parent = _redundant_parent(db, tag_id, tag["name"])
        if parent:
            faults.append(f"duplicates {parent}")

        if not faults:
            retract(r["id"], "every fault it named has cleared")
            verdicts[key] = ("acted", "retracted: resolved")
            continue

        data["faults"] = faults
        data["held"] = held
        db.execute(
            "UPDATE patrol_observations SET data = ?, weight = ?, detail = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') WHERE id = ?",
            (_json.dumps(data, sort_keys=True), round(1.0 / (1.0 + held), 6),
             f"{held} memories - " + "; ".join(faults), r["id"]),
        )
        verdicts[key] = (
            ("acted", "repaired the centroid; " + "; ".join(faults))
            if repaired else ("ok", "; ".join(faults))
        )
    db.connection.commit()
    return verdicts


register(Errand(
    name="seam_audit",
    description="Re-check the walk's own findings; retract what no longer holds",
    run=_seam_audit,
    domain="observations",
    # Every kind now, not only the tag-anchored ones.
    population="1 = 1",
    when="always",
))


def _tag_gap_variants(
    *,
    db: DatabaseEngine,
    config: ChicoryConfig,
    memory_ids: list[str],
    state: WanderRecord,
    patrol: "Patrol",
    should_stop: Callable[[], bool] | None = None,
    **_: object,
) -> dict[str, tuple[str, str | None]]:
    """Find an indexed term's unindexed twin — the blind spot between two bridges.

    Two variant-detectors already exist and neither can see this.  The
    meta-shelf bridge compares GAP to GAP; tag_twins compares TAG to TAG.
    Nothing compares a tag to a gap — and the moment one spelling becomes
    a tag it is retracted from the gap net, so fixing one variant makes
    its twin permanently invisible to both.

    That is not hypothetical.  This library holds three English editions
    of the Zohar with different romanisation conventions: `gvurah`,
    `sfirot`, `malchut`, `chochmah` in one, `gevurah`, `sefirot`,
    `malkhut`, `chokhmah` in another, `hesed` in a third.  Counts are
    near-categorical — gvurah 707 to 2 in the first, gevurah 80 to 0 in
    the second — so the split is a convention rather than noise.  The
    first edition got tagged and the sefirot charter was built on it.
    Twelve volumes and 5,719 chunks of the second are unreachable by
    their own vocabulary, and the gap net recorded every one of those
    terms without anything ever joining them up.

    Spelling proposes and the material narrows: cosine between the tag's
    co-occurring tags and the gap's `around` counter, both being "the tags
    carried by the memories that mention this".

    That second test is a TRIAGE CUT and not a verdict, which the numbers
    insist on.  Over 717 candidates the scores run continuous — the widest
    break is 0.033 — with true and false interleaved the whole way down:
    `mazal`/`azael` at 0.884 and `power`/`tower` at 0.839 both outscore
    `shechinah`/`shekhinah` at 0.795, and `adsorption`/`absorption` sits
    at 0.801 while meaning the opposite process.

    So it records with both numbers and decides nothing, the way tag_twins
    does.  Reading a candidate is the only thing that settles it.
    """
    from difflib import SequenceMatcher

    ids = [int(x) for x in memory_ids if str(x).isdigit()]
    if not ids:
        return {}
    verdicts: dict[str, tuple[str, str | None]] = {
        str(i): ("skipped", "too short to compare") for i in ids
    }
    ph = ",".join("?" * len(ids))
    mine = db.execute(
        f"SELECT id, name FROM tags WHERE id IN ({ph}) AND is_active = 1 "
        f"AND LENGTH(name) > 4", tuple(ids),
    ).fetchall()
    if not mine:
        return verdicts

    # Names only: loading every gap's JSON per batch would re-read megabytes
    # to score a handful of candidates.  The `around` counter is fetched
    # below, for the few names that survive the spelling test.
    gap_names = [r["subject"] for r in db.execute(
        "SELECT subject FROM patrol_observations "
        "WHERE kind = 'concept_gap' AND retracted = 0"
    ).fetchall()]
    live_tags = {r["name"] for r in db.execute(
        "SELECT name FROM tags WHERE is_active = 1")}
    by_len: dict[int, list[str]] = {}
    for g in gap_names:
        if g not in live_tags and len(g) > 4:
            by_len.setdefault(len(g), []).append(g)

    ratio_min = config.meta_shelf_variant_ratio
    arc_min = config.variant_context_min
    matcher = SequenceMatcher(autojunk=False)

    for r in mine:
        if should_stop is not None and should_stop():
            break
        key, name = str(r["id"]), r["name"]
        if name.startswith(_STRUCTURAL_TAG_PREFIXES):
            verdicts[key] = ("skipped", "a clock stamp")
            continue
        matcher.set_seq2(name)
        cand = []
        for delta in (-2, -1, 0, 1, 2):
            for g in by_len.get(len(name) + delta, ()):
                if g == name:
                    continue
                matcher.set_seq1(g)
                if matcher.real_quick_ratio() < ratio_min \
                        or matcher.quick_ratio() < ratio_min:
                    continue
                score = matcher.ratio()
                if score >= ratio_min:
                    cand.append((round(score, 4), g))
        if not cand:
            verdicts[key] = ("ok", "no unindexed variant")
            continue

        near = _tag_context(db, r["id"])
        found = []
        for score, g in sorted(cand, reverse=True)[:6]:
            row = db.execute(
                "SELECT data FROM patrol_observations WHERE kind = 'concept_gap' "
                "AND subject = ? AND retracted = 0", (g,),
            ).fetchone()
            if row is None:
                continue
            try:
                data = json.loads(row["data"] or "{}")
            except (ValueError, TypeError):
                continue
            around = {k: v for k, v in (data.get("around") or {}).items()
                      if not k.startswith(_STRUCTURAL_TAG_PREFIXES) and len(k) > 1}
            arc = _profile_cosine(near, around)
            if arc >= arc_min:
                found.append({"gap": g, "ratio": score, "arc": round(arc, 4),
                              "mentions": data.get("mentions", 0),
                              "documents": (data.get("documents") or [])[:4]})
        if not found:
            verdicts[key] = ("ok", "spelling matched, material did not")
            continue
        best = found[0]
        patrol.observe(
            errand="tag_gap_variants",
            kind="tag_gap_variant",
            subject=name,
            detail=(
                f"{name!r} is indexed; {best['gap']!r} is not, and the two sit "
                f"in the same material (spelling {best['ratio']}, context "
                f"{best['arc']}). {best['mentions']} mentions are unreachable "
                f"by that spelling."
            ),
            weight=round(1.0 - best["arc"], 6),
            data={"tag_id": r["id"], "variants": found},
            lap=state.lap,
        )
        verdicts[key] = ("acted", f"{len(found)} unindexed variant(s)")
    return verdicts


def _tag_context(db: DatabaseEngine, tag_id: int) -> dict[str, int]:
    """The tags carried by the memories that carry this one.

    The same shape as a gap observation's `around` counter, deliberately:
    that is what makes the two sides comparable at all.
    """
    return {
        r["name"]: r["k"] for r in db.execute(
            "SELECT t2.name, COUNT(*) AS k FROM memory_tags a "
            "JOIN memory_tags b ON b.memory_id = a.memory_id "
            "JOIN tags t2 ON t2.id = b.tag_id "
            "WHERE a.tag_id = ? AND b.tag_id != ? AND LENGTH(t2.name) > 1 "
            "GROUP BY t2.id ORDER BY k DESC LIMIT 24",
            (tag_id, tag_id),
        ).fetchall()
        if not r["name"].startswith(_STRUCTURAL_TAG_PREFIXES)
    }


def _profile_cosine(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    shared = a.keys() & b.keys()
    if not shared:
        return 0.0
    num = sum(a[k] * b[k] for k in shared)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return num / (na * nb) if na and nb else 0.0


register(Errand(
    name="tag_gap_variants",
    description="Find an indexed term's unindexed spelling — editions the shelf misses",
    run=_tag_gap_variants,
    domain="tags",
    # Compares each tag against the whole standing gap net, so it waits.
    when="idle",
))


def _hebrew_lexicon(
    *,
    db: DatabaseEngine,
    config: ChicoryConfig,
    memory_ids: list[str],
    state: WanderRecord,
    patrol: "Patrol",
    should_stop: Callable[[], bool] | None = None,
    **_: object,
) -> dict[str, tuple[str, str | None]]:
    """Read the bilingual volumes in Hebrew, and let the English name it.

    A whole-corpus sweep once per lap, like the meta-shelf proposer: the
    alignment is a ratio over chunk-sets and means nothing until both
    sides are counted.

    Scoped to the Hebrew-bearing material by the population predicate, so
    it cannot disturb the English gap net that covers everything else.
    """
    from chicory.layer4.bilingual import BilingualLexicon

    marker = f"lap{state.lap}"
    if (state.cursor_hint or "") == marker:
        return {mid: ("skipped", "already built this lap") for mid in memory_ids}
    # As above: a whole-corpus build with no partial resume, so the only
    # useful place to yield is before starting one.
    if should_stop is not None and should_stop():
        return {mid: ("skipped", "a patron is waiting") for mid in memory_ids}
    result = BilingualLexicon(config, db).build(patrol=patrol, lap=state.lap)
    db.execute(
        "UPDATE wander_state SET cursor_hint = ?, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') WHERE errand = ?",
        (marker, "hebrew_lexicon"),
    )
    db.connection.commit()
    sk = result.get("skipped") or {}
    _log.info("Bilingual lexicon: %d pairs from %d chunks "
              "(%d below the Hebrew share, %d with no usable skeleton)",
              result.get("aligned", 0), result.get("chunks_scanned", 0),
              sk.get("below_share", 0), sk.get("no_skeletons", 0))
    return {mid: ("ok", f"aligned={result.get('aligned', 0)}")
            for mid in memory_ids}


register(Errand(
    name="hebrew_lexicon",
    description="Learn a Hebrew-English lexicon from the facing translation",
    run=_hebrew_lexicon,
    # The same predicate the sweep uses, deliberately.  An aleph LIKE is
    # eight times faster and finds 5,729 of the 5,765, but the two would
    # then disagree about what the population IS — and `משה` has no aleph,
    # so the cheap version is an approximation rather than an index hint.
    population="m.content GLOB '*[א-ת]*'",
    when="idle",
))


def _asymmetry_definition(config: ChicoryConfig) -> str:
    """A fingerprint of the DECLARED comparison — pairs and registry.

    Folded into the once-per-lap marker so that editing a role, a pair or
    a registry entry re-derives on the next batch instead of standing on
    the old definition until the lap turns.  A lap here is the whole
    memories domain: 148,654 stampings were outstanding when this was
    written, ~2,974 batches at the default fifty, and new material pushes
    the boundary out as it arrives.  Waiting that long to notice an edit
    the author made deliberately is the same defect in miniature as the
    one the registry exists to fix.

    Config only, so it costs no query.  A document joining or leaving a
    role *in the library* still waits for the lap, because detecting that
    means digesting every document; ``_withdraw_stale`` is what catches
    it when the comparison does run.
    """
    import hashlib

    payload = json.dumps(
        [config.asymmetry_pairs, config.corpus_documents],
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _unstamped_asymmetries(db: DatabaseEngine) -> bool:
    """Whether a standing finding carries no record of what it compared.

    One indexed lookup, and it opens the guard exactly once: a finding
    written before the registry existed says nothing about which
    documents were on either side, so it cannot be audited and must be
    re-derived rather than left to age out.  124 rows were in that state
    when this shipped, and the lap guard would have held them there.

    After the first pass every standing row is stamped, so this is a
    LIMIT 1 miss on an index for the rest of the errand's life.
    """
    return db.execute(
        "SELECT 1 FROM patrol_observations WHERE kind = 'corpus_asymmetry' "
        "AND retracted = 0 AND data NOT LIKE '%\"basis\"%' LIMIT 1"
    ).fetchone() is not None


def _corpus_asymmetry(
    *,
    db: DatabaseEngine,
    config: ChicoryConfig,
    memory_ids: list[str],
    state: WanderRecord,
    patrol: "Patrol",
    should_stop: Callable[[], bool] | None = None,
    **_: object,
) -> dict[str, tuple[str, str | None]]:
    """Diff two accounts of one thing, and record where they disagree.

    Not "find an asymmetry" — find the SAME thing represented twice.  A
    tag pair scoring 0.997 on centroid similarity is two different things
    whose vectors sit close, and 3,724 of those produced no question worth
    asking.  A concept in two languages, a text in three witnesses, a
    subject described from inside and outside: those diverge about
    something, which is what makes the divergence readable.

    A whole-corpus sweep once per lap, like the other comparative passes,
    because a rate over a population means nothing until the population is
    fully counted.
    """
    from chicory.layer4.asymmetry import CorpusAsymmetry

    marker = f"lap{state.lap}:{_asymmetry_definition(config)}"
    if (state.cursor_hint or "") == marker and not _unstamped_asymmetries(db):
        return {mid: ("skipped", "already compared this lap") for mid in memory_ids}
    # The pass digests every document and profiles both sides with no
    # partial resume, so before it is the only useful place to yield —
    # measured at 3.43s, held under the foreground lock.  cursor_hint stays
    # unset on a bail, so the comparison is re-attempted rather than
    # recorded as done for this lap.
    if should_stop is not None and should_stop():
        return {mid: ("skipped", "a patron is waiting") for mid in memory_ids}
    result = CorpusAsymmetry(config, db).run(patrol=patrol, lap=state.lap)
    db.execute(
        "UPDATE wander_state SET cursor_hint = ?, "
        "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') WHERE errand = ?",
        (marker, "corpus_asymmetry"),
    )
    db.connection.commit()
    total = sum(p.get("recorded", 0) for p in result.get("pairs", []))
    # Which documents each side resolved to, every lap.  The comparison
    # was wrong for a day because nothing said which files were in it,
    # and a count with no population behind it is what hid that.
    for p in result.get("pairs", []):
        _log.info(
            "Corpus asymmetry %s (%s): left %d chunks over %d documents, "
            "right %d over %d; %d recorded, %d withdrawn%s%s",
            p.get("pair"), p.get("skipped") or "compared",
            p.get("left_chunks", 0), len(p.get("left_documents", [])),
            p.get("right_chunks", 0), len(p.get("right_documents", [])),
            p.get("recorded", 0), p.get("withdrawn", 0),
            (f"; {len(p['duplicates_excluded'])} duplicate copies excluded"
             if p.get("duplicates_excluded") else ""),
            (f"; registry names {', '.join(p['unresolved_registry_entries'])}"
             f" but the library holds no such document"
             if p.get("unresolved_registry_entries") else ""),
        )
    _log.info("Corpus asymmetry: %d findings across %d pairs",
              total, len(result.get("pairs", [])))
    return {mid: ("ok", f"asymmetries={total}") for mid in memory_ids}


register(Errand(
    name="corpus_asymmetry",
    description="Diff two accounts of one thing and record where they disagree",
    run=_corpus_asymmetry,
    # Reads the content of whole populations, so it waits for quiet.
    when="idle",
))


def _tag_content(
    *,
    db: DatabaseEngine,
    config: ChicoryConfig,
    memory_ids: list[str],
    state: WanderRecord,
    patrol: "Patrol",
    should_stop: Callable[[], bool] | None = None,
    **_: object,
) -> dict[str, tuple[str, str | None]]:
    """Ask whether a tag's material is about what the tag says it is.

    Two representations of the same memories — the words they use, and the
    label they were filed under.  Comparing content against the tag NAME
    would say nothing: `mysticism` never appears in the material it
    labels.  What distinguishes a subject from a label is whether its
    memories share vocabulary the library at large does not.

    Measured: `plato` yields socrates, glaucon, republic.  `biosemiotics`
    yields pattee.  `whirly-art` and `essay`, 1,552 memories each, yield
    nothing whatever — a tag on a great deal of material that is not about
    anything in particular.

    Records both outcomes.  A tag with no distinctive vocabulary is not
    necessarily wrong — it may be a genuine catch-all — but it cannot be
    routed to, and nothing else in the system notices that.
    """
    from chicory.layer4.asymmetry import TagContentAsymmetry

    ids = [int(x) for x in memory_ids if str(x).isdigit()]
    if not ids:
        return {}
    verdicts: dict[str, tuple[str, str | None]] = {
        str(i): ("skipped", "not an eligible tag") for i in ids
    }
    ph = ",".join("?" * len(ids))
    rows = db.execute(
        f"SELECT id, name FROM tags WHERE id IN ({ph}) AND is_active = 1 "
        f"AND LENGTH(name) > 1", tuple(ids),
    ).fetchall()
    if not rows:
        return verdicts
    checker = TagContentAsymmetry(config, db)
    for r in rows:
        if should_stop is not None and should_stop():
            break
        if r["name"].startswith(_STRUCTURAL_TAG_PREFIXES):
            verdicts[str(r["id"])] = ("skipped", "a clock stamp")
            continue
        found = checker.examine(r["id"], r["name"])
        if found is None:
            verdicts[str(r["id"])] = ("skipped", "too few memories to profile")
            continue
        if found["distinctive"]:
            top = ", ".join(d["term"] for d in found["distinctive"][:5])
            patrol.observe(
                errand="tag_content", kind="tag_content", subject=r["name"],
                detail=f"its material is about: {top}",
                weight=round(1.0 / (1.0 + len(found["distinctive"])), 6),
                data=found, lap=state.lap,
            )
            verdicts[str(r["id"])] = ("ok", f"about: {top[:40]}")
        else:
            patrol.observe(
                errand="tag_content", kind="tag_content", subject=r["name"],
                detail=(
                    f"{found['sampled']} memories carry this tag and share no "
                    f"vocabulary the library does not — it labels material "
                    f"without naming a subject, so nothing can be routed to it"
                ),
                weight=1.0, data=found, lap=state.lap,
            )
            verdicts[str(r["id"])] = ("acted", "no distinctive vocabulary")
    db.connection.commit()
    return verdicts


register(Errand(
    name="tag_content",
    description="Ask whether a tag's material is about what the tag claims",
    run=_tag_content,
    domain="tags",
    # Reads the content behind every tag it visits, so it waits for quiet.
    when="idle",
))


register(Errand(
    name="concept_gaps",
    description="Record concepts the library discusses but holds no tag for",
    run=_concept_gaps,
    # A whole-library survey that reads content, so it waits for quiet.
    # This is the first user of the idle slot, which existed for exactly
    # this: work that should not compete with a patron's question.
    when="idle",
))


# ── Heartbeat ───────────────────────────────────────────────────────────


class Heartbeat:
    """A clock for the background pass.

    The tick calls whatever it is given — in practice the orchestrator's
    existing background pass, which already acquires and releases the
    foreground lock per phase and returns early when a request is waiting.
    That is the whole reason this class is so thin: the hard part
    (preemption) was already solved for the traffic-paced case, and
    borrowing it means idle work needs no separate concurrency story.

    Idleness is *observed*, not instrumented.  A tick that finds the
    foreground waiting counts as busy and resets the run; after
    ``wander_idle_ticks`` consecutive quiet ticks the library is idle, and
    idle-only errands become eligible.  This costs nothing and touches no
    existing call site — the alternative, stamping a timestamp at the eight
    places that set ``_foreground_waiting``, buys precision the patrol has
    no use for.
    """

    def __init__(
        self,
        config: ChicoryConfig,
        tick: Callable[[bool], None],
        busy: Callable[[], bool] | None = None,
    ) -> None:
        self._config = config
        self._tick = tick
        self._busy = busy
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._quiet_ticks = 0
        self._ticks = 0

    @property
    def idle(self) -> bool:
        return self._quiet_ticks >= self._config.wander_idle_ticks

    @property
    def ticks(self) -> int:
        return self._ticks

    def start(self) -> bool:
        """Start the heartbeat thread.  Idempotent; False if disabled."""
        if not self._config.wander_enabled:
            _log.info("Heartbeat disabled by config")
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="chicory-heartbeat", daemon=True,
        )
        self._thread.start()
        _log.info(
            "Heartbeat started (tick=%.0fs, idle after %d quiet ticks)",
            self._config.wander_tick_seconds, self._config.wander_idle_ticks,
        )
        return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)

    def run_once(self) -> bool:
        """One beat.  Returns whether the tick actually ran.

        Exposed for tests and for a manual poke, so the loop below has no
        logic of its own worth testing separately.
        """
        if self._is_busy():
            self._quiet_ticks = 0
            return False
        self._quiet_ticks += 1
        self._ticks += 1
        try:
            self._tick(self.idle)
        except Exception:
            _log.exception("Heartbeat tick failed")
        return True

    def _is_busy(self) -> bool:
        return self._busy is not None and self._busy()

    def _run(self) -> None:
        # wait() rather than sleep() so stop() is immediate rather than
        # taking up to a full tick to be noticed.
        while not self._stop.wait(self._config.wander_tick_seconds):
            self.run_once()
