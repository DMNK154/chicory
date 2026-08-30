"""The desk report: what the library would tell you if it could speak first.

Every other surface in this system is pull — findings, commissions, FAQ
candidates and custody flags accumulate quietly and are only ever seen by
someone who already knew to look.  A patrol that discovers 1,802
near-duplicates and cannot mention them is doing bookkeeping, not
librarianship.

This assembles one answer to three questions:

    what changed        since you were last here
    what needs you      judgments only a person can make
    what I learned      digests written, questions crystallized, tickets closed

plus a health line, so the state of the building is visible without
asking for it.

Two deliberate properties:

- Reading does not acknowledge.  The "since" watermark advances only on
  an explicit ack, so a report you glanced at and did not act on comes
  back next time.  Silent state changes are exactly what the custody
  layer exists to prevent; the desk holds itself to the same rule.
- Every number here is derived, never cached.  The report cannot drift
  from the library it describes.
"""

from __future__ import annotations

import logging
from datetime import datetime

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.layer4.librarian import CUSTODY_STATES

_log = logging.getLogger(__name__)

_WATERMARK_KEY = "desk_report_seen_at"


class DeskReport:
    """Assembles the librarian's standing report."""

    def __init__(self, config: ChicoryConfig, db: DatabaseEngine) -> None:
        self._config = config
        self._db = db

    # ── State ───────────────────────────────────────────────────────────

    def watermark(self) -> str | None:
        row = self._db.execute(
            "SELECT value FROM librarian_state WHERE key = ?", (_WATERMARK_KEY,),
        ).fetchone()
        return row["value"] if row else None

    def acknowledge(self, at: str | None = None) -> dict:
        """Mark the current moment as seen; deltas restart from here.

        The stamp comes from SQLite's clock in SQLite's format, because
        every timestamp it will be compared against does too.  Python's
        isoformat() carries microseconds where the schema defaults carry
        milliseconds, and the two compare lexicographically — a row
        written in the same millisecond as the ack would sort before it
        and vanish from the next report.
        """
        if at is not None:
            self._db.execute(
                "INSERT INTO librarian_state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')",
                (_WATERMARK_KEY, at),
            )
            return {"acknowledged_at": at}

        self._db.execute(
            "INSERT INTO librarian_state (key, value) "
            "VALUES (?, strftime('%Y-%m-%dT%H:%M:%f', 'now')) "
            "ON CONFLICT(key) DO UPDATE SET "
            "value = strftime('%Y-%m-%dT%H:%M:%f', 'now'), "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')",
            (_WATERMARK_KEY,),
        )
        return {"acknowledged_at": self.watermark()}

    # ── Report ──────────────────────────────────────────────────────────

    def report(self, limit: int = 5) -> dict:
        # A limit of 0 would show empty queues and read as an all-clear;
        # a negative one is an unbounded dump.  Neither is a greeting.
        limit = max(1, min(int(limit or 5), 50))
        since = self.watermark()
        # One pass over `memories` feeds both the queue and the health
        # line — this is a greeting, and five separate full scans of a
        # six-figure table is not a greeting.
        stats = self._memory_stats()
        changed = self._changed(since)
        needs = self._needs_you(limit, stats)
        learned = self._learned(since, limit)
        health = self._health(stats)
        return {
            "since": since,
            "first_report": since is None,
            "headline": self._headline(since, changed, needs, learned),
            "changed": changed,
            "needs_you": needs,
            "learned": learned,
            "health": health,
        }

    # ── Sections ────────────────────────────────────────────────────────

    def _changed(self, since: str | None) -> dict:
        if since is None:
            return {"note": "first report — nothing to compare against yet"}

        stored = self._scalar(
            "SELECT COUNT(*) AS n FROM memories WHERE created_at > ?", (since,),
        )
        # Only real transitions. Supersession journals a custody event with
        # from_status == to_status (the standing did not change, the belief
        # did), and counting those as transitions would inflate whichever
        # status the superseded memories happened to hold.
        transitions = {
            r["to_status"]: r["n"]
            for r in self._db.execute(
                "SELECT to_status, COUNT(*) AS n FROM custody_events "
                "WHERE created_at > ? AND from_status != to_status "
                "GROUP BY to_status", (since,),
            ).fetchall()
        }
        gate = {
            r["verdict"]: r["n"]
            for r in self._db.execute(
                "SELECT verdict, COUNT(*) AS n FROM write_gate_log "
                "WHERE created_at > ? GROUP BY verdict", (since,),
            ).fetchall()
        }
        new_subjects = [
            r["display_name"] or r["path"] or r["name"] for r in self._db.execute(
                "SELECT name, display_name, path FROM subjects "
                "WHERE status = 'active' AND created_at > ? "
                "ORDER BY memory_count DESC LIMIT 10", (since,),
            ).fetchall()
        ]
        new_subjects_total = self._scalar(
            "SELECT COUNT(*) AS n FROM subjects "
            "WHERE status = 'active' AND created_at > ?", (since,),
        )
        # A changed belief is not the same news as new work, and it is the
        # news a reader most needs: something the library previously
        # asserted has been overturned since they last looked.
        beliefs = [
            {
                "superseded": r["old_summary"],
                "by": r["new_summary"],
                "claim": r["claim"],
                "scope": r["scope"],
                "actor": r["actor"],
                "at": r["created_at"],
            }
            for r in self._db.execute(
                "SELECT s.scope, s.claim, s.actor, s.created_at, "
                "old.summary AS old_summary, new.summary AS new_summary "
                "FROM supersessions s "
                "LEFT JOIN memories old ON old.id = s.superseded_id "
                "LEFT JOIN memories new ON new.id = s.superseding_id "
                "WHERE s.retracted = 0 AND s.created_at > ? "
                "ORDER BY s.created_at DESC LIMIT 10", (since,),
            ).fetchall()
        ]
        beliefs_total = self._scalar(
            "SELECT COUNT(*) AS n FROM supersessions "
            "WHERE retracted = 0 AND created_at > ?", (since,),
        )
        retired = [
            r["path"] or r["name"] for r in self._db.execute(
                "SELECT name, path FROM subjects "
                "WHERE status = 'retired' AND updated_at > ? LIMIT 10", (since,),
            ).fetchall()
        ]
        return {
            "memories_stored": stored,
            "custody_transitions": transitions,
            "gate_verdicts": gate,
            "subjects_added": new_subjects,
            "subjects_added_total": new_subjects_total,
            "subjects_retired": retired,
            "beliefs_revised": beliefs,
            "beliefs_revised_total": beliefs_total,
        }

    def _memory_stats(self) -> dict:
        """Every whole-table memory count in a single scan.

        Custody is folded in as conditional sums rather than a second
        GROUP BY — the states are a known, closed set, so there is no
        reason to walk a six-figure table twice for a greeting.
        """
        custody_cols = ", ".join(
            f"SUM(CASE WHEN custody_status = '{s}' THEN 1 ELSE 0 END) AS c_{s}"
            for s in CUSTODY_STATES
        )
        row = self._db.execute(
            f"""
            SELECT COUNT(*) AS total,
                   SUM(CASE WHEN swept_at IS NOT NULL THEN 1 ELSE 0 END) AS swept,
                   SUM(CASE WHEN author IS NOT NULL THEN 1 ELSE 0 END) AS signed,
                   SUM(CASE WHEN ingestion_tier = 'quarantine' THEN 1 ELSE 0 END)
                       AS quarantined,
                   {custody_cols}
            FROM memories WHERE is_archived = 0
            """
        ).fetchone()
        custody = {}
        if row is not None:
            for s in CUSTODY_STATES:
                n = row[f"c_{s}"] or 0
                if n:
                    custody[s] = n
        return {
            "total": (row["total"] or 0) if row else 0,
            "swept": (row["swept"] or 0) if row else 0,
            "signed": (row["signed"] or 0) if row else 0,
            "quarantined": (row["quarantined"] or 0) if row else 0,
            "custody": custody,
        }

    def _needs_you(self, limit: int, stats: dict) -> dict:
        quarantined = stats["quarantined"]
        findings = {
            r["finding_type"]: r["n"]
            for r in self._db.execute(
                "SELECT finding_type, COUNT(*) AS n FROM sweep_findings "
                "WHERE resolved = 0 GROUP BY finding_type ORDER BY n DESC"
            ).fetchall()
        }
        # 'claimed' belongs here.  A specialist that claimed a ticket and
        # could not finish it (the teach cycle is told to leave one claimed
        # when its report gets quarantined) leaves work that is invisible
        # on every other surface — and CommissionDesk.file() dedups against
        # non-closed tickets, so re-filing the same gap silently points at
        # the stalled one.  Stalled work is exactly what a desk report is for.
        commissions = [
            {
                "commission_id": r["id"],
                "question": r["question_text"],
                "source": r["source"],
                "status": r["status"],
                "opened": r["created_at"],
                "last_touched": r["updated_at"],
                "stalled": r["status"] == "claimed",
            }
            for r in self._db.execute(
                "SELECT id, question_text, source, status, created_at, updated_at "
                "FROM commissions WHERE status IN ('open', 'claimed', 'reported') "
                "ORDER BY created_at ASC LIMIT ?", (limit,),
            ).fetchall()
        ]
        faq = [
            {"faq_id": r["id"], "question": r["question_text"], "status": r["status"]}
            for r in self._db.execute(
                "SELECT id, question_text, status FROM faq_entries "
                "WHERE status IN ('candidate', 'stale') "
                "ORDER BY created_at ASC LIMIT ?", (limit,),
            ).fetchall()
        ]
        ingress = [
            {"receipt": r["receipt"], "preview": (r["content"] or "")[:120]}
            for r in self._db.execute(
                "SELECT receipt, content FROM message_ingress "
                "WHERE status = 'flagged' ORDER BY arrived_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        ]

        # Totals, separately.  The lists above are capped for display; a
        # headline that counted them would report the page size and call
        # it the backlog — understating the queue by exactly the amount
        # that makes it worth mentioning.
        totals = {
            "commissions": self._scalar(
                "SELECT COUNT(*) AS n FROM commissions "
                "WHERE status IN ('open', 'claimed', 'reported')"
            ),
            "commissions_stalled": self._scalar(
                "SELECT COUNT(*) AS n FROM commissions WHERE status = 'claimed'"
            ),
            "faq": self._scalar(
                "SELECT COUNT(*) AS n FROM faq_entries "
                "WHERE status IN ('candidate', 'stale')"
            ),
            "ingress_flagged": self._scalar(
                "SELECT COUNT(*) AS n FROM message_ingress WHERE status = 'flagged'"
            ),
        }
        return {
            "quarantined_memories": quarantined,
            "open_findings": findings,
            "open_commissions": commissions,
            "faq_awaiting_judgment": faq,
            "ingress_flagged": ingress,
            "totals": totals,
            "lists_capped_at": limit,
        }

    def _learned(self, since: str | None, limit: int) -> dict:
        clause = " AND m.created_at > ?" if since else ""
        params = (since,) if since else ()

        # Resolve the tag once and filter on tag_id — joining through
        # tags.name instead makes both of these full scans of a
        # million-row junction table, for a greeting.
        tag_row = self._db.execute(
            "SELECT id FROM tags WHERE name = 'digest'"
        ).fetchone()
        if tag_row is None:
            return {
                "digests": [], "digests_total": 0,
                "commissions_closed": self._closed_commissions(since, params),
                "faq_serving": self._scalar(
                    "SELECT COUNT(*) AS n FROM faq_entries WHERE status = 'active'"
                ),
            }
        digest_tag = tag_row["id"]
        digests = [
            {
                "memory_id": r["id"],
                "summary": r["summary"],
                "author": r["author"],
                "created_at": r["created_at"],
            }
            for r in self._db.execute(
                "SELECT m.id, m.summary, m.author, m.created_at "
                "FROM memory_tags mt JOIN memories m ON m.id = mt.memory_id "
                f"WHERE mt.tag_id = ? AND m.is_archived = 0{clause} "
                "ORDER BY m.created_at DESC LIMIT ?",
                (digest_tag, *params, limit),
            ).fetchall()
        ]
        closed = self._closed_commissions(since, params)
        faq_active = self._scalar(
            "SELECT COUNT(*) AS n FROM faq_entries WHERE status = 'active'"
        )
        digests_total = self._scalar(
            "SELECT COUNT(*) AS n FROM memory_tags mt "
            "JOIN memories m ON m.id = mt.memory_id "
            f"WHERE mt.tag_id = ? AND m.is_archived = 0{clause}",
            (digest_tag, *params),
        )
        return {
            "digests": digests,
            "digests_total": digests_total,
            "commissions_closed": closed,
            "faq_serving": faq_active,
        }

    def _closed_commissions(self, since: str | None, params: tuple) -> int:
        return self._scalar(
            "SELECT COUNT(*) AS n FROM commissions WHERE status = 'closed'"
            + (" AND updated_at > ?" if since else ""),
            params,
        )

    def _health(self, stats: dict) -> dict:
        total = stats["total"]
        subjects = self._scalar(
            "SELECT COUNT(*) AS n FROM subjects WHERE status = 'active'"
        )
        queries = self._scalar("SELECT COUNT(*) AS n FROM query_history")
        return {
            "memories": total,
            "patrol_coverage": round(stats["swept"] / total, 3) if total else 0.0,
            "custody": stats["custody"],
            "subjects": subjects,
            "signed_memories": stats["signed"],
            "queries_recorded": queries,
            "gate_enforcing": self._config.librarian_gate_enforce,
        }

    # ── Headline ────────────────────────────────────────────────────────

    def _headline(
        self, since: str | None, changed: dict, needs: dict, learned: dict,
    ) -> str:
        """One plain sentence a person can act on.

        Composed mechanically — the desk has no model of its own, and a
        templated line that is always true beats a fluent one that drifts.
        """
        parts: list[str] = []

        # Counts come from totals, never from the capped display lists
        t = needs["totals"]
        pending = []
        if t["commissions"]:
            n = t["commissions"]
            clause = f"{n} commission{'s' if n != 1 else ''}"
            if t["commissions_stalled"]:
                clause += f" ({t['commissions_stalled']} stalled mid-research)"
            pending.append(clause)
        if t["faq"]:
            pending.append(f"{t['faq']} FAQ draft{'s' if t['faq'] != 1 else ''}")
        if needs["quarantined_memories"]:
            pending.append(f"{needs['quarantined_memories']} quarantined")
        if t["ingress_flagged"]:
            n = t["ingress_flagged"]
            pending.append(f"{n} flagged arrival{'s' if n != 1 else ''}")
        if pending:
            parts.append("awaiting your judgment: " + ", ".join(pending))

        open_findings = sum(needs["open_findings"].values())
        if open_findings:
            kinds = ", ".join(
                f"{n} {k.replace('_', '-')}"
                for k, n in list(needs["open_findings"].items())[:3]
            )
            parts.append(f"{open_findings:,} open findings ({kinds})")

        if since is None:
            parts.append("first report — no baseline yet")
        else:
            stored = changed.get("memories_stored", 0)
            if stored:
                parts.append(f"{stored:,} new memories since you were last here")
            added = changed.get("subjects_added_total", 0)
            if added:
                parts.append(f"{added} new shel{'ves' if added != 1 else 'f'}")

            revised = changed.get("beliefs_revised_total", 0)
            if revised:
                parts.append(
                    f"{revised} belief{'s' if revised != 1 else ''} revised"
                )

        written = learned.get("digests_total", 0)
        if written:
            parts.append(f"{written} digest{'s' if written != 1 else ''} written")

        return "; ".join(parts) if parts else "Nothing new — the library is quiet."

    # ── Helper ──────────────────────────────────────────────────────────

    def _scalar(self, sql: str, params: tuple = ()) -> int:
        row = self._db.execute(sql, params).fetchone()
        return (row["n"] or 0) if row else 0
