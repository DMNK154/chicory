"""Ingress dwell: hold incoming content under receipt before committing it.

The write gate decides in the moment, with one message in hand.  That
blinds it to every signature that is *temporal* — a poisoning burst whose
members each look fine alone, content that only reads as an attack in
light of what arrives next, a message that resembles something already
held out.  Those need multiple messages coexisting in one window.

So content is not committed on arrival.  It is HELD at a known address
with a RECEIPT (content hash, source, arrival time, dwell deadline), and
the hold itself is the detection window: every new arrival re-examines
everything still dwelling, so consecutive messages overlap in custody and
can be compared.  At the end of the dwell the receipt is SETTLED
(cleared / flagged) and only then is commitment to memory even considered.

Two properties matter as much as the detection:

- The receipt outlives the dwell.  Some poisoning is only visible in
  hindsight, when later content reveals an earlier plant; the receipt is
  what late detection has to grab — content hash, origin, timing, and
  every finding ever written against it.
- No verdict here blocks anything on its own.  Findings are advisory and
  provenance-anchored: statistics propose, custody and human judgment
  dispose.  A detector that grades its own history is not a detector.

This is a PARALLEL commit path, not a gate on the live response.  The
conversation proceeds at full speed; what dwells is the commitment to
memory, which nothing needs instantly.

What this does NOT do, stated plainly:

- It does not prevent anything.  Every finding is advisory; commitment
  is a separate, human-or-specialist decision.
- It is evadable, and cheaply.  Space a burst beyond the dwell window
  and the members never coexist; paraphrase below the cosine threshold
  and they never match; avoid the listed phrasings and the diction
  screen is silent.  Statistical detection always loses to an adversary
  who reads the detector.
- trajectory_deviation cannot distinguish an attack from a new idea.
  Novelty is off-trajectory by construction; the score means "look at
  this", never "this is hostile".
- resembles_quarantined is the only provenance-anchored check here, and
  it is INERT until quarantine has contents — which, with the write gate
  shipping in shadow mode, means it does nothing on a default install.
  It becomes useful only after gate enforcement is switched on.

The durable value is the receipt and the window, not the four passes:
content held at a known address, with its arrival provenance and every
finding ever written against it, available for judgment now and for
tracing later.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timedelta

import numpy as np

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.layer4.librarian import _array_to_blob, _blob_to_array, _unit

_log = logging.getLogger(__name__)

# Ingress statuses:
#   held       — dwelling; detection window open
#   cleared    — dwell elapsed, no findings
#   flagged    — dwell elapsed, findings written (awaiting judgment)
#   committed  — promoted into a memory (memory_id set)
#   discarded  — settled without commitment (judged not worth shelving)
INGRESS_STATES = ("held", "cleared", "flagged", "committed", "discarded")

# Diction screen: instruction-shaped, authority-claiming, or
# context-overriding phrasing.  This is the WEAKEST signal in the set —
# purely lexical, trivially paraphrased around, and prone to firing on
# legitimate quoted material or discussion *about* injection (this
# docstring would trip it).  It never decides anything by itself; it only
# raises a finding a human or specialist reads alongside provenance.
_IMPERATIVE_PATTERNS = [
    (re.compile(r"\bignore (?:all |any )?(?:previous|prior|above|earlier)\b", re.I), "override-previous"),
    (re.compile(r"\bdisregard (?:all |any )?(?:previous|prior|above|earlier|your)\b", re.I), "override-previous"),
    (re.compile(r"\byou are now\b|\bfrom now on,? you\b|\bact as (?:if|though)?\b", re.I), "role-reassignment"),
    (re.compile(r"\b(?:new|updated|revised) (?:instructions|system prompt|directive)\b", re.I), "instruction-claim"),
    (re.compile(r"\b(?:system|admin(?:istrator)?|developer|anthropic) (?:override|mode|message|says)\b", re.I), "authority-claim"),
    (re.compile(r"\bthe user (?:has )?(?:already )?(?:approved|authorized|consented)\b", re.I), "consent-claim"),
    (re.compile(r"\bdo not (?:tell|inform|mention to|alert) the user\b", re.I), "concealment"),
    (re.compile(r"\breveal (?:your|the) (?:system prompt|instructions|rules)\b", re.I), "exfiltration"),
]


def _now() -> datetime:
    return datetime.utcnow()


class IngressBuffer:
    """Receipted hold-and-examine window in front of memory commitment."""

    def __init__(self, config: ChicoryConfig, db: DatabaseEngine) -> None:
        self._config = config
        self._db = db

    # ── Hold ────────────────────────────────────────────────────────────

    def hold(
        self,
        content: str,
        embedding: np.ndarray | None = None,
        source: str = "user",
        origin: str | None = None,
        author: str | None = None,
    ) -> dict:
        """Take content into custody and issue a receipt.

        Returns the receipt record.  Nothing is committed to memory here.
        """
        receipt = f"rcpt-{uuid.uuid4().hex[:16]}"
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        arrived = _now()
        dwell_until = arrived + timedelta(
            seconds=self._config.ingress_dwell_seconds
        )
        blob = (
            _array_to_blob(_unit(embedding.astype(np.float32)))
            if embedding is not None else None
        )
        self._db.execute(
            "INSERT INTO message_ingress "
            "(receipt, content, content_hash, embedding, source, origin, "
            " author, arrived_at, dwell_until, status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'held')",
            (receipt, content, content_hash, blob, source, origin, author,
             arrived.isoformat(), dwell_until.isoformat()),
        )
        return {
            "receipt": receipt,
            "content_hash": content_hash,
            "source": source,
            "origin": origin,
            "author": author,
            "arrived_at": arrived.isoformat(),
            "dwell_until": dwell_until.isoformat(),
            "status": "held",
        }

    # ── Examine (the dwell's whole point) ───────────────────────────────

    def examine(self, receipt: str | None = None) -> list[dict]:
        """Run detection over held content.

        With no receipt, examines everything currently held — which is how
        the overlap works: a new arrival re-examines its predecessors while
        they are still in the window, so cross-message signatures become
        visible that neither message shows alone.

        Findings are written against the receipt and returned.  Nothing is
        blocked, moved, or rejected here.
        """
        held = self._load_held()
        if not held:
            return []
        targets = (
            [h for h in held if h["receipt"] == receipt] if receipt else held
        )
        if not targets:
            return []

        # The comparison window: everything held, plus recently settled
        # receipts — a burst that straddles a settle boundary is still a burst.
        window = self._load_window()
        # Load quarantine vectors ONCE per pass, not once per held item —
        # this runs under the foreground lock, and a per-item rescan of the
        # quarantine tier turns one arrival into N full table scans.
        quarantine = self._load_quarantine_vectors()
        findings: list[dict] = []
        for item in targets:
            findings.extend(self._burst_repetition(item, window))
            findings.extend(self._trajectory_deviation(item, window))
            findings.extend(self._imperative_register(item))
            findings.extend(self._resembles_quarantined(item, quarantine))

        # Both members of a pair discover the same normalized finding when
        # the whole window is examined; collapse before writing or returning
        # so the result matches the one row the UNIQUE constraint keeps.
        deduped: dict[tuple, dict] = {}
        for f in findings:
            # '' is the storage sentinel for "no peer"; None is the surface
            f["peer_receipt"] = f.get("peer_receipt") or None
            deduped.setdefault(
                (f["receipt"], f["finding_type"], f["peer_receipt"] or ""), f,
            )
        findings = list(deduped.values())

        if findings:
            # Upsert, not INSERT OR IGNORE: re-examination is the whole point
            # of a dwell, and the passes produce values that legitimately
            # change (the quarantine set grows, the window centroid shifts).
            # A stronger later observation must replace the first sighting,
            # or the stored record permanently under-reports.
            self._db.executemany(
                "INSERT INTO ingress_findings "
                "(receipt, finding_type, detail, severity, peer_receipt) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(receipt, finding_type, peer_receipt) DO UPDATE SET "
                "detail = excluded.detail, severity = excluded.severity "
                "WHERE excluded.severity > ingress_findings.severity",
                [
                    (f["receipt"], f["finding_type"], f["detail"],
                     f["severity"], f.get("peer_receipt") or "")
                    for f in findings
                ],
            )
        self._db.executemany(
            "UPDATE message_ingress SET examined_count = examined_count + 1 "
            "WHERE receipt = ?",
            [(t["receipt"],) for t in targets],
        )
        return findings

    def _burst_repetition(self, item: dict, window: list[dict]) -> list[dict]:
        """Near-identical content coexisting in the window.

        The signature only a dwell can see: each message of a poisoning
        burst looks unremarkable alone; together they are a campaign.
        """
        if item["vec"] is None:
            return []
        threshold = self._config.ingress_burst_threshold
        out = []
        for peer in window:
            if peer["receipt"] == item["receipt"] or peer["vec"] is None:
                continue
            # A model change mid-buffer leaves mixed dimensions; a raw dot
            # would raise and wedge every subsequent examination
            if peer["vec"].shape != item["vec"].shape:
                continue
            sim = float(np.dot(item["vec"], peer["vec"]))
            if sim >= threshold:
                # File under the lexically-first receipt so A~B and B~A
                # collapse to one finding for the pair
                first, second = sorted((item["receipt"], peer["receipt"]))
                out.append({
                    "receipt": first,
                    "finding_type": "burst_repetition",
                    "detail": f"cosine={sim:.3f} hash={peer['content_hash']}",
                    "severity": round(sim, 3),
                    "peer_receipt": second,
                })
        return out

    def _trajectory_deviation(self, item: dict, window: list[dict]) -> list[dict]:
        """Distance from the running centroid of recent arrivals.

        ADVISORY ONLY, and the weakest inference in the module: novelty
        deviates from trajectory exactly as attacks do.  A high score here
        means "examine", never "attack".
        """
        if item["vec"] is None:
            return []
        peers = [
            p["vec"] for p in window
            if p["receipt"] != item["receipt"] and p["vec"] is not None
            and p["vec"].shape == item["vec"].shape
        ]
        if len(peers) < self._config.ingress_trajectory_min_history:
            return []
        centroid = _unit(np.mean(np.stack(peers), axis=0))
        sim = float(np.dot(item["vec"], centroid))
        if sim < self._config.ingress_trajectory_deviation_threshold:
            return [{
                "receipt": item["receipt"],
                "finding_type": "trajectory_deviation",
                "detail": (
                    f"cosine_to_recent={sim:.3f} over {len(peers)} messages "
                    f"(advisory: novelty deviates too)"
                ),
                "severity": round(1.0 - max(sim, 0.0), 3),
                "peer_receipt": "",
            }]
        return []

    def _imperative_register(self, item: dict) -> list[dict]:
        """Lexical screen for instruction/authority diction.

        Weak and evadable by design — see _IMPERATIVE_PATTERNS.  Fires
        openly on discussion *about* injection, which is why it is a
        finding for a reader, not a verdict.
        """
        hits = [
            label for pattern, label in _IMPERATIVE_PATTERNS
            if pattern.search(item["content"])
        ]
        if not hits:
            return []
        unique = sorted(set(hits))
        return [{
            "receipt": item["receipt"],
            "finding_type": "imperative_register",
            "detail": "patterns=" + ",".join(unique),
            "severity": round(min(0.2 * len(unique), 1.0), 3),
            "peer_receipt": None,
        }]

    def _load_quarantine_vectors(self) -> list[tuple[str, np.ndarray]]:
        """Unit vectors of quarantine-tier memories, loaded once per pass."""
        rows = self._db.execute(
            """
            SELECT e.memory_id, e.embedding FROM embeddings e
            JOIN memories m ON m.id = e.memory_id
            WHERE e.chunk_index = 0 AND m.ingestion_tier = 'quarantine'
              AND m.is_archived = 0
            """
        ).fetchall()
        return [
            (r["memory_id"],
             _unit(np.frombuffer(r["embedding"], dtype=np.float32).copy()))
            for r in rows
        ]

    def _resembles_quarantined(
        self, item: dict, quarantine: list[tuple[str, np.ndarray]],
    ) -> list[dict]:
        """Content closely matching a memory already held out.

        The provenance-anchored check in this set: it appeals to a prior
        custody decision rather than to a statistical notion of normal.
        Inert until quarantine has contents — see the module docstring.
        """
        if item["vec"] is None or not quarantine:
            return []
        threshold = self._config.ingress_quarantine_resemblance_threshold
        best_id, best_sim = None, threshold
        for memory_id, vec in quarantine:
            if vec.shape != item["vec"].shape:
                continue
            sim = float(np.dot(item["vec"], vec))
            if sim >= best_sim:
                best_id, best_sim = memory_id, sim
        if best_id is None:
            return []
        return [{
            "receipt": item["receipt"],
            "finding_type": "resembles_quarantined",
            "detail": f"memory_id={best_id} cosine={best_sim:.3f}",
            "severity": round(best_sim, 3),
            "peer_receipt": None,
        }]

    # ── Settle ──────────────────────────────────────────────────────────

    def due(self) -> list[str]:
        """Receipts whose dwell has elapsed and are still held."""
        rows = self._db.execute(
            "SELECT receipt FROM message_ingress "
            "WHERE status = 'held' AND dwell_until <= ? ORDER BY arrived_at ASC",
            (_now().isoformat(),),
        ).fetchall()
        return [r["receipt"] for r in rows]

    def settle(self, receipt: str) -> dict:
        """Close the dwell: cleared if no findings, flagged if any.

        Settling never destroys anything — the receipt and its findings
        persist for hindsight, and commitment remains a separate decision.
        """
        row = self._db.execute(
            "SELECT status FROM message_ingress WHERE receipt = ?", (receipt,)
        ).fetchone()
        if row is None:
            return {"error": f"Unknown receipt: {receipt}"}
        if row["status"] != "held":
            return {"receipt": receipt, "status": row["status"], "note": "already settled"}

        # Final examination at the boundary, then verdict
        self.examine(receipt)
        findings = self.findings_for(receipt)
        status = "flagged" if findings else "cleared"
        self._db.execute(
            "UPDATE message_ingress SET status = ?, settled_at = ? WHERE receipt = ?",
            (status, _now().isoformat(), receipt),
        )
        return {
            "receipt": receipt,
            "status": status,
            "findings": findings,
        }

    def settle_due(self, limit: int | None = None) -> dict:
        """Settle receipts whose dwell has elapsed.

        Bounded by default: this runs under the foreground DB lock, and an
        unbounded backlog (a burst, or a long server outage) would hold
        that lock through hundreds of examinations.  Whatever is left
        settles on the next background pass.
        """
        cleared = flagged = 0
        due = self.due()
        if limit is not None:
            due = due[:limit]
        for receipt in due:
            result = self.settle(receipt)
            if result.get("status") == "flagged":
                flagged += 1
            elif result.get("status") == "cleared":
                cleared += 1
        return {
            "settled": cleared + flagged,
            "cleared": cleared,
            "flagged": flagged,
            # due() re-reads live state, so the settled ones are already gone
            "remaining": len(self.due()),
        }

    def mark_committed(self, receipt: str, memory_id: str) -> bool:
        cur = self._db.execute(
            "UPDATE message_ingress SET status = 'committed', memory_id = ?, "
            "settled_at = COALESCE(settled_at, ?) WHERE receipt = ?",
            (memory_id, _now().isoformat(), receipt),
        )
        return cur.rowcount > 0

    def mark_discarded(self, receipt: str) -> bool:
        """Discard a receipt without shelving it.

        Refuses a committed receipt: the memory it produced still exists,
        so overwriting the status would erase the shelving record while
        leaving the shelved content in place.
        """
        cur = self._db.execute(
            "UPDATE message_ingress SET status = 'discarded', "
            "settled_at = COALESCE(settled_at, ?) "
            "WHERE receipt = ? AND status != 'committed'",
            (_now().isoformat(), receipt),
        )
        return cur.rowcount > 0

    # ── Reading ─────────────────────────────────────────────────────────

    def get(self, receipt: str) -> dict | None:
        row = self._db.execute(
            "SELECT receipt, content, content_hash, source, origin, author, "
            "arrived_at, dwell_until, status, examined_count, memory_id, "
            "settled_at FROM message_ingress WHERE receipt = ?",
            (receipt,),
        ).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["findings"] = self.findings_for(receipt)
        return record

    def findings_for(self, receipt: str) -> list[dict]:
        """Findings naming this receipt on either side of a pair."""
        rows = self._db.execute(
            "SELECT finding_type, detail, severity, peer_receipt, created_at "
            "FROM ingress_findings WHERE receipt = ? OR peer_receipt = ? "
            "ORDER BY severity DESC",
            (receipt, receipt),
        ).fetchall()
        out = []
        for r in rows:
            rec = dict(r)
            # '' is the storage sentinel for "no peer"; present it as None
            rec["peer_receipt"] = rec["peer_receipt"] or None
            out.append(rec)
        return out

    def list_receipts(self, status: str = "held", limit: int = 25) -> list[dict]:
        rows = self._db.execute(
            "SELECT receipt, content_hash, source, origin, author, arrived_at, "
            "dwell_until, status, examined_count, memory_id, "
            "substr(content, 1, 200) AS preview "
            "FROM message_ingress WHERE status = ? "
            "ORDER BY arrived_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
        out = []
        for r in rows:
            rec = dict(r)
            rec["findings"] = self.findings_for(r["receipt"])
            out.append(rec)
        return out

    def stats(self) -> dict:
        rows = self._db.execute(
            "SELECT status, COUNT(*) AS n FROM message_ingress GROUP BY status"
        ).fetchall()
        finding_rows = self._db.execute(
            "SELECT finding_type, COUNT(*) AS n FROM ingress_findings "
            "GROUP BY finding_type"
        ).fetchall()
        return {
            "enabled": self._config.ingress_enabled,
            "dwell_seconds": self._config.ingress_dwell_seconds,
            "by_status": {r["status"]: r["n"] for r in rows},
            "findings": {r["finding_type"]: r["n"] for r in finding_rows},
        }

    def prune(self) -> int:
        """Drop adjudicated receipts past the retention window.

        Two things retention must never do:

        - Empty the judgment queue.  'flagged' receipts are the ones
          awaiting a decision; they are kept regardless of age, so
          retention can never quietly resolve a question by outliving it.
          Only cleared / committed / discarded receipts age out.
        - Strip evidence from a receipt that is still live.  A burst
          finding is a shared PAIR record; deleting it because one member
          aged out would erase the other member's only evidence and could
          flip its settle verdict to 'cleared'.  So findings are deleted
          only once BOTH sides are gone — i.e. orphans.
        """
        days = self._config.ingress_retention_days
        if days <= 0:
            return 0
        cutoff = (_now() - timedelta(days=days)).isoformat()
        stale = self._db.execute(
            "SELECT receipt FROM message_ingress "
            "WHERE status IN ('cleared', 'committed', 'discarded') "
            "AND settled_at IS NOT NULL AND settled_at < ?",
            (cutoff,),
        ).fetchall()
        if not stale:
            return 0
        receipts = [(r["receipt"],) for r in stale]
        self._db.executemany(
            "DELETE FROM message_ingress WHERE receipt = ?", receipts,
        )
        # Sweep orphans: a finding survives while either side still exists
        self._db.execute(
            "DELETE FROM ingress_findings WHERE "
            "receipt NOT IN (SELECT receipt FROM message_ingress) AND "
            "(peer_receipt = '' OR "
            " peer_receipt NOT IN (SELECT receipt FROM message_ingress))"
        )
        return len(receipts)

    # ── Internals ───────────────────────────────────────────────────────

    def _load_held(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT receipt, content, content_hash, embedding FROM message_ingress "
            "WHERE status = 'held' ORDER BY arrived_at ASC"
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def _load_window(self) -> list[dict]:
        """Held content plus the most recent settled receipts.

        The window is what any held item is compared against — a burst
        that straddles a settle boundary must still read as a burst.
        """
        rows = self._db.execute(
            "SELECT receipt, content, content_hash, embedding FROM message_ingress "
            "WHERE status = 'held' "
            "UNION ALL "
            "SELECT receipt, content, content_hash, embedding FROM ("
            "  SELECT receipt, content, content_hash, embedding, arrived_at "
            "  FROM message_ingress WHERE status != 'held' "
            "  ORDER BY arrived_at DESC LIMIT ?"
            ")",
            (self._config.ingress_window_lookback,),
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    @staticmethod
    def _row_to_item(row) -> dict:
        vec = None
        if row["embedding"] is not None:
            vec = _blob_to_array(row["embedding"])
        return {
            "receipt": row["receipt"],
            "content": row["content"],
            "content_hash": row["content_hash"],
            "vec": vec,
        }
