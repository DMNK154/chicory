"""Co-occurrence optimizer: normalized co-occurrence edges and local block formation.

Asks: which memories/tags repeatedly activate together in the same context?
Uses positive PMI (PPMI) for normalization so common tags don't dominate.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chicory.config import ChicoryConfig
    from chicory.db.engine import DatabaseEngine


class CooccurrenceOptimizer:

    def __init__(self, db: DatabaseEngine, config: ChicoryConfig) -> None:
        self._db = db
        self._cfg = config
        self._resonant_set: set[tuple[int, int]] | None = None

    def _get_resonant_set(self) -> set[tuple[int, int]]:
        if self._resonant_set is None:
            rows = self._db.execute(
                "SELECT tag_a_id, tag_b_id FROM glyph_resonances"
            ).fetchall()
            self._resonant_set = {(r["tag_a_id"], r["tag_b_id"]) for r in rows}
        return self._resonant_set

    def invalidate_resonance_cache(self) -> None:
        self._resonant_set = None

    def update_from_scope(
        self,
        scope_type: str,
        activated_tag_ids: list[int],
        activated_memory_ids: list[str],
    ) -> list[str]:
        """Update co-occurrence edges for a single activation scope.

        Tag pairs are gated by glyph lattice resonance.
        Returns block_keys of blocks whose membership changed.
        """
        if len(activated_tag_ids) < 2 and len(activated_memory_ids) < 2:
            return []

        total_scopes = self._get_total_scope_count(scope_type)
        changed_keys: list[str] = []

        pairs = self._generate_tag_pairs(activated_tag_ids, scope_type)
        pairs += self._generate_pairs("memory", activated_memory_ids, scope_type)

        if not pairs:
            return []

        now = "strftime('%Y-%m-%dT%H:%M:%f', 'now')"

        for left_type, left_id, right_type, right_id in pairs:
            left_count = self._get_item_scope_count(left_type, str(left_id), scope_type)
            right_count = self._get_item_scope_count(right_type, str(right_id), scope_type)

            row = self._db.execute(
                """SELECT raw_count FROM cooccurrence_edges
                   WHERE left_type=? AND left_id=? AND right_type=? AND right_id=? AND scope_type=?""",
                (left_type, str(left_id), right_type, str(right_id), scope_type),
            ).fetchone()

            new_count = (row["raw_count"] if row else 0.0) + 1.0
            n = max(total_scopes, 1)
            p_ab = new_count / n
            p_a = max(left_count, 1) / n
            p_b = max(right_count, 1) / n
            denom = p_a * p_b

            lift = p_ab / denom if denom > 0 else 0.0
            pmi = math.log(lift) if lift > 0 else 0.0
            ppmi = max(0.0, pmi)

            self._db.execute(
                f"""INSERT INTO cooccurrence_edges
                    (left_type, left_id, right_type, right_id, scope_type,
                     raw_count, expected_count, lift, pmi, co_strength, evidence_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    ON CONFLICT(left_type, left_id, right_type, right_id, scope_type)
                    DO UPDATE SET
                        raw_count = ?,
                        expected_count = ?,
                        lift = ?,
                        pmi = ?,
                        co_strength = ?,
                        evidence_count = evidence_count + 1,
                        last_seen_at = {now}""",
                (
                    left_type, str(left_id), right_type, str(right_id), scope_type,
                    new_count, denom * n, lift, pmi, ppmi,
                    new_count, denom * n, lift, pmi, ppmi,
                ),
            )

        changed_keys += self._update_blocks(scope_type, activated_tag_ids, activated_memory_ids)
        return changed_keys

    def get_co_strength(self, left_type: str, left_id: str, right_type: str, right_id: str) -> float:
        """Get the PPMI co-occurrence strength for a pair across all scopes."""
        row = self._db.execute(
            """SELECT MAX(co_strength) as s FROM cooccurrence_edges
               WHERE left_type=? AND left_id=? AND right_type=? AND right_id=?""",
            (left_type, left_id, right_type, right_id),
        ).fetchone()
        return row["s"] if row and row["s"] else 0.0

    def get_block_density(self, block_key: str) -> float:
        row = self._db.execute(
            "SELECT internal_density FROM forest_blocks WHERE block_key=?",
            (block_key,),
        ).fetchone()
        return row["internal_density"] if row else 0.0

    def _generate_tag_pairs(
        self, tag_ids: list[int], scope_type: str
    ) -> list[tuple[str, str, str, str]]:
        """Generate tag pairs gated by glyph lattice resonance."""
        if len(tag_ids) < 2:
            return []
        resonant = self._get_resonant_set()
        pairs = []
        for i in range(len(tag_ids)):
            for j in range(i + 1, len(tag_ids)):
                a, b = min(tag_ids[i], tag_ids[j]), max(tag_ids[i], tag_ids[j])
                if (a, b) in resonant:
                    pairs.append(("tag", str(a), "tag", str(b)))
        return pairs

    def _generate_pairs(
        self, item_type: str, ids: list, scope_type: str
    ) -> list[tuple[str, str, str, str]]:
        if len(ids) < 2:
            return []
        sorted_ids = sorted(str(i) for i in ids)
        pairs = []
        for i in range(len(sorted_ids)):
            for j in range(i + 1, len(sorted_ids)):
                pairs.append((item_type, sorted_ids[i], item_type, sorted_ids[j]))
        return pairs

    def _get_total_scope_count(self, scope_type: str) -> int:
        if scope_type == "retrieval":
            row = self._db.execute("SELECT COUNT(*) as c FROM retrieval_events").fetchone()
        elif scope_type == "sync_event":
            row = self._db.execute("SELECT COUNT(*) as c FROM synchronicity_events").fetchone()
        else:
            row = self._db.execute("SELECT COUNT(*) as c FROM memories").fetchone()
        return row["c"] if row else 1

    def _get_item_scope_count(self, item_type: str, item_id: str, scope_type: str) -> int:
        if item_type == "tag" and scope_type == "retrieval":
            row = self._db.execute(
                "SELECT COUNT(DISTINCT retrieval_id) as c FROM retrieval_tag_hits WHERE tag_id=?",
                (int(item_id),),
            ).fetchone()
        elif item_type == "tag":
            row = self._db.execute(
                "SELECT COUNT(*) as c FROM memory_tags WHERE tag_id=?",
                (int(item_id),),
            ).fetchone()
        elif item_type == "memory" and scope_type == "retrieval":
            row = self._db.execute(
                "SELECT COUNT(*) as c FROM retrieval_results WHERE memory_id=?",
                (item_id,),
            ).fetchone()
        else:
            row = {"c": 1}
        return row["c"] if row else 1

    def _batch_co_strengths(self, tag_ids: list[int]) -> dict[tuple[int, int], float]:
        """Fetch co-occurrence strengths for all resonant pairs in one query."""
        resonant = self._get_resonant_set()
        pairs = [
            (min(tag_ids[i], tag_ids[j]), max(tag_ids[i], tag_ids[j]))
            for i in range(len(tag_ids))
            for j in range(i + 1, len(tag_ids))
            if (min(tag_ids[i], tag_ids[j]), max(tag_ids[i], tag_ids[j])) in resonant
        ]
        if not pairs:
            return {}

        result: dict[tuple[int, int], float] = {}
        for chunk_start in range(0, len(pairs), 450):
            chunk = pairs[chunk_start:chunk_start + 450]
            conditions = " OR ".join(
                "(left_id = ? AND right_id = ?)" for _ in chunk
            )
            params = [v for a, b in chunk for v in (str(a), str(b))]
            rows = self._db.execute(
                f"""SELECT left_id, right_id, MAX(co_strength) as s
                    FROM cooccurrence_edges
                    WHERE left_type = 'tag' AND right_type = 'tag'
                      AND ({conditions})
                    GROUP BY left_id, right_id""",
                params,
            ).fetchall()
            for r in rows:
                result[(int(r["left_id"]), int(r["right_id"]))] = r["s"] or 0.0
        return result

    def _update_blocks(
        self,
        scope_type: str,
        tag_ids: list[int],
        memory_ids: list[str],
    ) -> list[str]:
        """Form or update a forest block from high-PPMI local neighborhood."""
        if not tag_ids:
            return []

        sorted_tag_strs = sorted(str(t) for t in tag_ids)
        block_key = _block_key("cooccurrence", "tag_set", sorted_tag_strs)

        co_map = self._batch_co_strengths(tag_ids)

        internal_strengths = list(co_map.values())
        density = sum(internal_strengths) / len(internal_strengths) if internal_strengths else 0.0

        self._db.execute(
            """INSERT INTO forest_blocks (block_key, block_type, forest_type, internal_density, evidence_count)
               VALUES (?, 'tag_set', 'cooccurrence', ?, 1)
               ON CONFLICT(block_key) DO UPDATE SET
                   internal_density = ?,
                   evidence_count = evidence_count + 1,
                   last_observed_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')""",
            (block_key, density, density),
        )

        block_row = self._db.execute(
            "SELECT id FROM forest_blocks WHERE block_key=?", (block_key,)
        ).fetchone()
        block_id = block_row["id"]

        for tid in tag_ids:
            co_sum = sum(
                co_map.get((min(tid, other), max(tid, other)), 0.0)
                for other in tag_ids if other != tid
            )
            strength = co_sum / max(len(tag_ids) - 1, 1)

            self._db.execute(
                """INSERT INTO block_memberships (block_id, target_type, target_id, membership_strength, evidence_count)
                   VALUES (?, 'tag', ?, ?, 1)
                   ON CONFLICT(block_id, target_type, target_id) DO UPDATE SET
                       membership_strength = ?,
                       evidence_count = evidence_count + 1,
                       last_seen_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')""",
                (block_id, str(tid), strength, strength),
            )

        for mid in memory_ids:
            self._db.execute(
                """INSERT INTO block_memberships (block_id, target_type, target_id, membership_strength, evidence_count)
                   VALUES (?, 'memory', ?, 1.0, 1)
                   ON CONFLICT(block_id, target_type, target_id) DO UPDATE SET
                       evidence_count = evidence_count + 1,
                       last_seen_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')""",
                (block_id, mid),
            )

        return [block_key]


def _block_key(forest_type: str, block_type: str, sorted_ids: list[str]) -> str:
    raw = f"{forest_type}:{block_type}:{','.join(sorted_ids)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]
