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

import logging
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
        self._last_run: dict[str, datetime] = {}

    @property
    def state(self) -> WanderState:
        return self._state

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
            _log.warning("Re-embed failed for %s: %s", mid, exc)
            verdicts[mid] = ("failed", str(exc)[:200])

    _resolve_findings(db, [m for m, (v, _) in verdicts.items() if v == "acted"])
    return verdicts


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
