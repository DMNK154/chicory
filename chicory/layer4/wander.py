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
class Errand:
    """One standing question the patrol carries around the library.

    ``population`` narrows which memories the errand cares about at all —
    a SQL predicate over alias ``m`` (the memories table).  The cursor then
    orders *within* that population, so a targeted errand still gets the
    least-recently-visited guarantee, and a memory it cannot fix rotates to
    the back instead of blocking the queue.

    ``when`` is "always" for work that is safe beside a conversation, or
    "idle" for work that should wait until nobody is talking.
    """

    name: str
    description: str
    run: Callable[..., dict[str, tuple[str, str | None]]]
    population: str | None = None
    when: str = "always"


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
        where = "m.is_archived = 0"
        if errand.population:
            where += f" AND ({errand.population})"
        rows = self._db.execute(
            f"""
            SELECT m.id, v.lap AS visit_lap FROM memories m
            LEFT JOIN patrol_visits v
                   ON v.memory_id = m.id AND v.errand = ?
            WHERE {where}
            ORDER BY v.visited_at ASC, m.created_at ASC
            LIMIT ?
            """,
            (errand.name, limit),
        ).fetchall()
        return [(r["id"], r["visit_lap"]) for r in rows]

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
        """How many memories this errand still cares about.

        This is a full scan of the memories table with the errand's
        predicate applied per row — 3.4s on a six-figure library — so it is
        NEVER on the patrol's hot path.  Status callers pass a cap, which
        short-circuits the scan and turns the answer into "at least cap".
        """
        where = "m.is_archived = 0"
        if errand.population:
            where += f" AND ({errand.population})"
        if cap is not None:
            row = self._db.execute(
                f"SELECT COUNT(*) AS n FROM "
                f"(SELECT 1 FROM memories m WHERE {where} LIMIT ?)",
                (cap,),
            ).fetchone()
        else:
            row = self._db.execute(
                f"SELECT COUNT(*) AS n FROM memories m WHERE {where}"
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
                "SELECT data FROM patrol_observations "
                "WHERE errand = ? AND kind = ? AND subject = ?",
                (errand, kind, subject),
            ).fetchone()
            if row and row["data"]:
                try:
                    prior = json.loads(row["data"])
                except (json.JSONDecodeError, TypeError, ValueError):
                    prior = {}
                for k, v in prior.items():
                    cur = merged.get(k)
                    if isinstance(v, list) and isinstance(cur, list):
                        merged[k] = sorted(set(v) | set(cur))
                    elif isinstance(v, dict) and isinstance(cur, dict):
                        # Counters sum: a region is built a batch at a
                        # time, and how often two concepts share a memory
                        # is only meaningful over the whole corpus.
                        totals = dict(v)
                        for kk, vv in cur.items():
                            totals[kk] = totals.get(kk, 0) + vv
                        merged[k] = totals
                    elif isinstance(v, (int, float)) and isinstance(cur, (int, float)):
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
                if len(tag) > 1:      # letter tags carry no subject
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
            continue  # findable; not a gap
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
