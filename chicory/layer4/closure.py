"""Specialist closure: unwinding the chain a piece of work opened.

Work descends through layers.  A patron's question opens a consult, the
consult opens a shelf, the shelf opens a commission, the commission opens
a specialist shift, and the shift produces a report.  Until now the
report simply landed at the bottom and stopped — the ticket closed, and
every layer above it kept believing whatever it believed before.

That is not hypothetical.  A shift proved a shelf's near-duplicate load
was an embedding artifact, and the parent digest went on asserting, under
a specialist byline, that the same findings were "not noise".  Both
memories remained equally retrievable and equally authoritative.  A later
reader — or a later specialist, following the protocol's instruction to
read its parent — could be handed the refuted claim.

Closure walks back up, in reverse order, and settles each layer:

    report written        ->  supersede the claims it overturned
    claims superseded     ->  stale any FAQ resting on them
    ticket closed         ->  tell whatever spawned it
    belief changed        ->  surface it on the desk

Three properties it must have, all of them inherited from the rest of
this system:

- Nothing is deleted.  A superseded memory stays exactly where it was and
  keeps its byline; it simply arrives wearing a pointer to what replaced
  it.  The library's history of being wrong is part of the record.
- Supersession is AUTHORED, never derived.  It lives in its own table
  rather than the relational tensor's supersession channel, because that
  channel is a derived edge cache with a decay-and-prune lifecycle, and a
  judgment that a claim was refuted must not quietly age out.
- Scope is honest.  Most corrections overturn a *claim*, not a whole
  memory.  A digest that got one section wrong is not thereby worthless,
  and marking it wholly superseded would throw away the parts that still
  stand.
"""

from __future__ import annotations

import logging

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.layer4.librarian import record_custody_transition

_log = logging.getLogger(__name__)

SCOPES = ("claim", "whole")


class ClosureLedger:
    """Records supersessions and unwinds the chain a report closes."""

    def __init__(self, config: ChicoryConfig, db: DatabaseEngine) -> None:
        self._config = config
        self._db = db

    # ── Supersession ────────────────────────────────────────────────────

    def supersede(
        self,
        superseded_id: str,
        superseding_id: str,
        actor: str,
        claim: str | None = None,
        reason: str | None = None,
        scope: str = "claim",
    ) -> dict:
        """Record that one memory overturns something in another.

        `claim` names the specific assertion that fell, so a partial
        correction stays partial.  Leave it None only for scope='whole'.
        """
        if scope not in SCOPES:
            return {"error": f"scope must be one of {SCOPES}"}
        if superseded_id == superseding_id:
            return {"error": "a memory cannot supersede itself"}

        rows = self._db.execute(
            "SELECT id FROM memories WHERE id IN (?, ?) AND is_archived = 0",
            (superseded_id, superseding_id),
        ).fetchall()
        found = {r["id"] for r in rows}
        missing = {superseded_id, superseding_id} - found
        if missing:
            return {"error": f"unknown or archived memory: {sorted(missing)}"}

        # '' is the storage sentinel for "no claim named" — see the schema
        # comment; a NULL here would defeat the UNIQUE constraint entirely.
        claim_key = claim or ""
        self._db.execute(
            "INSERT INTO supersessions "
            "(superseded_id, superseding_id, scope, claim, reason, actor) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(superseded_id, superseding_id, claim) DO UPDATE SET "
            "scope = excluded.scope, reason = excluded.reason, "
            "actor = excluded.actor, retracted = 0, "
            # a revived supersession is news again, so it must re-date or the
            # desk's since-window will never surface it
            "created_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')",
            (superseded_id, superseding_id, scope, claim_key, reason, actor),
        )
        # A custody event, so the change of belief is in the same journal
        # as every other change of standing.
        self._db.execute(
            "INSERT INTO custody_events "
            "(memory_id, from_status, to_status, actor, reason) "
            "SELECT id, custody_status, custody_status, ?, ? FROM memories "
            "WHERE id = ?",
            (actor, f"superseded ({scope}) by {superseding_id}"
                    + (f": {claim}" if claim else ""), superseded_id),
        )
        stale = self._stale_dependent_faq(superseded_id)
        return {
            "status": "recorded",
            "superseded_id": superseded_id,
            "superseding_id": superseding_id,
            "scope": scope,
            "claim": claim,
            "faq_marked_stale": stale,
        }

    def retract(
        self,
        superseded_id: str,
        superseding_id: str,
        claim: str | None = None,
        all_claims: bool = False,
    ) -> bool:
        """Withdraw a supersession — the correction was itself wrong.

        Withdraws the ONE named claim by default.  A pair can carry
        several independent corrections, and retracting a mistaken one
        must not silently reinstate the rest.  Pass all_claims=True to
        withdraw every correction between the pair deliberately.
        """
        if all_claims:
            cur = self._db.execute(
                "UPDATE supersessions SET retracted = 1 "
                "WHERE superseded_id = ? AND superseding_id = ? AND retracted = 0",
                (superseded_id, superseding_id),
            )
        else:
            cur = self._db.execute(
                "UPDATE supersessions SET retracted = 1 "
                "WHERE superseded_id = ? AND superseding_id = ? AND claim = ? "
                "AND retracted = 0",
                (superseded_id, superseding_id, claim or ""),
            )
        return cur.rowcount > 0

    def _stale_dependent_faq(self, superseded_id: str) -> int:
        """An FAQ answer resting on a refuted memory must not keep serving.

        The FAQ desk already re-verifies that supporting memories still
        exist; existence was never the question here — correctness is.
        """
        # Not just 'active': a candidate resting on a refuted memory can be
        # approved later, and a stale one can be refreshed and re-approved.
        # Both would then serve the refuted claim as canonical.
        rows = self._db.execute(
            "SELECT id, supporting_memory_ids, status FROM faq_entries "
            "WHERE status IN ('active', 'candidate', 'stale')"
        ).fetchall()
        hit = [
            r["id"] for r in rows
            if f'"{superseded_id}"' in (r["supporting_memory_ids"] or "")
        ]
        for faq_id in hit:
            self._db.execute(
                "UPDATE faq_entries SET status = 'stale' WHERE id = ?", (faq_id,),
            )
        return len(hit)

    def superseded_among(self, memory_ids: list[str]) -> set[str]:
        """Which of these memories currently stand refuted."""
        if not memory_ids:
            return set()
        ph = ",".join("?" * len(memory_ids))
        rows = self._db.execute(
            f"SELECT DISTINCT superseded_id FROM supersessions "
            f"WHERE retracted = 0 AND superseded_id IN ({ph})",
            tuple(memory_ids),
        ).fetchall()
        return {r["superseded_id"] for r in rows}

    # ── Reading ─────────────────────────────────────────────────────────

    def for_memories(self, memory_ids: list[str]) -> dict[str, list[dict]]:
        """Supersessions standing against each of these memories."""
        if not memory_ids:
            return {}
        ph = ",".join("?" * len(memory_ids))
        rows = self._db.execute(
            f"SELECT s.superseded_id, s.superseding_id, s.scope, s.claim, "
            f"s.reason, s.actor, s.created_at, m.summary AS by_summary "
            f"FROM supersessions s "
            f"LEFT JOIN memories m ON m.id = s.superseding_id "
            f"WHERE s.retracted = 0 AND s.superseded_id IN ({ph})",
            tuple(memory_ids),
        ).fetchall()
        out: dict[str, list[dict]] = {}
        for r in rows:
            out.setdefault(r["superseded_id"], []).append({
                "superseding_id": r["superseding_id"],
                "scope": r["scope"],
                "claim": r["claim"],
                "reason": r["reason"],
                "actor": r["actor"],
                "at": r["created_at"],
                "superseding_summary": r["by_summary"],
            })
        return out

    def recent(self, since: str | None = None, limit: int = 20) -> list[dict]:
        sql = (
            "SELECT s.superseded_id, s.superseding_id, s.scope, s.claim, "
            "s.actor, s.created_at, old.summary AS old_summary, "
            "new.summary AS new_summary FROM supersessions s "
            "LEFT JOIN memories old ON old.id = s.superseded_id "
            "LEFT JOIN memories new ON new.id = s.superseding_id "
            "WHERE s.retracted = 0"
        )
        params: list = []
        if since:
            sql += " AND s.created_at > ?"
            params.append(since)
        sql += " ORDER BY s.created_at DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self._db.execute(sql, tuple(params)).fetchall()]

    def count(self, since: str | None = None) -> int:
        sql = "SELECT COUNT(*) AS n FROM supersessions WHERE retracted = 0"
        params: tuple = ()
        if since:
            sql += " AND created_at > ?"
            params = (since,)
        row = self._db.execute(sql, params).fetchone()
        return (row["n"] or 0) if row else 0

    # ── The unwind ──────────────────────────────────────────────────────

    def close_chain(
        self,
        report_memory_id: str,
        actor: str,
        commission_id: int | None = None,
        supersedes: list[dict] | None = None,
        resolve_findings: list[int] | None = None,
    ) -> dict:
        """Settle every layer a completed piece of work passed through.

        Runs in reverse order — the report first, then what it overturned,
        then what depended on that, then the ticket, then the source that
        raised the ticket.  Each step is reported, so a partial unwind is
        visible rather than silent.
        """
        steps: list[dict] = []

        row = self._db.execute(
            "SELECT id FROM memories WHERE id = ? AND is_archived = 0",
            (report_memory_id,),
        ).fetchone()
        if row is None:
            return {"error": f"unknown or archived report: {report_memory_id}"}
        steps.append({"step": "report", "memory_id": report_memory_id})

        # 1. What the report overturned
        recorded, faq_stale, failed = [], 0, 0
        for item in supersedes or []:
            if not item.get("memory_id"):
                failed += 1
                steps.append({"step": "supersede",
                              "error": "entry has no memory_id", "entry": item})
                continue
            res = self.supersede(
                superseded_id=item["memory_id"],
                superseding_id=report_memory_id,
                actor=actor,
                claim=item.get("claim"),
                reason=item.get("reason"),
                scope=item.get("scope", "claim"),
            )
            if "error" in res:
                failed += 1
                steps.append({"step": "supersede", "error": res["error"],
                              "memory_id": item.get("memory_id")})
            else:
                recorded.append(item["memory_id"])
                faq_stale += res["faq_marked_stale"]
        if supersedes:
            steps.append({"step": "supersede", "recorded": recorded,
                          "failed": failed, "faq_marked_stale": faq_stale})
        if failed and not recorded:
            # The ticket was raised by a claim this report was supposed to
            # settle.  Closing it while nothing was actually superseded
            # would retire the question and leave the belief standing.
            return {
                "status": "incomplete",
                "error": "every supersession failed; commission left open",
                "report_memory_id": report_memory_id,
                "steps": steps,
            }

        # 2. Findings the report explains away
        resolved = 0
        for fid in resolve_findings or []:
            cur = self._db.execute(
                "UPDATE sweep_findings SET resolved = 1 "
                "WHERE id = ? AND resolved = 0", (fid,),
            )
            resolved += cur.rowcount
        if resolve_findings:
            steps.append({"step": "findings", "resolved": resolved})

        # 3. The ticket, and the source that raised it
        if commission_id is not None:
            crow = self._db.execute(
                "SELECT status, source FROM commissions WHERE id = ?",
                (commission_id,),
            ).fetchone()
            if crow is None:
                steps.append({"step": "commission",
                              "error": f"unknown commission {commission_id}"})
            else:
                self._db.execute(
                    "UPDATE commissions SET status = 'closed', "
                    "report_memory_id = ?, "
                    "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') "
                    "WHERE id = ?",
                    (report_memory_id, commission_id),
                )
                steps.append({"step": "commission", "id": commission_id,
                              "closed_from": crow["status"],
                              "raised_by": crow["source"]})

        return {
            "status": "closed",
            "report_memory_id": report_memory_id,
            "actor": actor,
            "steps": steps,
        }
