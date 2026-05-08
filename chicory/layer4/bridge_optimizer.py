"""Bridge optimizer: sparse high-leverage bridge edges between forest blocks.

Asks: which far-apart regions should remain reachable?
A bridge is strong when two blocks are semantically distant, cluster-distant,
and repeatedly or meaningfully connected. Generic hubs are penalized.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chicory.config import ChicoryConfig
    from chicory.db.engine import DatabaseEngine


class BridgeOptimizer:

    def __init__(self, db: DatabaseEngine, config: ChicoryConfig) -> None:
        self._db = db
        self._cfg = config

    def update_bridges(self, touched_block_keys: list[str]) -> int:
        """Compute bridge edges between touched blocks and all other blocks.

        Returns count of bridge edges upserted.
        """
        if not touched_block_keys:
            return 0

        touched_blocks = []
        for key in touched_block_keys:
            row = self._db.execute(
                "SELECT id, block_key, internal_density FROM forest_blocks WHERE block_key=?",
                (key,),
            ).fetchone()
            if row:
                touched_blocks.append(row)

        if not touched_blocks:
            return 0

        all_blocks = self._db.execute(
            "SELECT id, block_key, internal_density FROM forest_blocks"
        ).fetchall()

        block_members: dict[int, set[str]] = {}
        for block in all_blocks:
            rows = self._db.execute(
                "SELECT target_type, target_id FROM block_memberships WHERE block_id=? AND target_type='tag'",
                (block["id"],),
            ).fetchall()
            block_members[block["id"]] = {r["target_id"] for r in rows}

        block_degree: dict[int, int] = {}
        for block in all_blocks:
            block_degree[block["id"]] = len(block_members.get(block["id"], set()))

        median_degree = sorted(block_degree.values())[len(block_degree) // 2] if block_degree else 1

        upserted = 0
        max_per_block = self._cfg.canopy_bridge_max_per_block

        for tblock in touched_blocks:
            tid = tblock["id"]
            t_members = block_members.get(tid, set())
            if not t_members:
                continue

            candidates: list[tuple[int, float]] = []

            for oblock in all_blocks:
                oid = oblock["id"]
                if oid == tid:
                    continue

                o_members = block_members.get(oid, set())
                if not o_members:
                    continue

                conn = self._connection_strength(t_members, o_members)
                if conn <= 0:
                    continue

                cluster_dist = self._cluster_distance(t_members, o_members)
                rarity = self._rarity_bonus(t_members, o_members)

                hub_pen = self._hub_penalty(
                    block_degree.get(tid, 0),
                    block_degree.get(oid, 0),
                    median_degree,
                )

                bridge = conn * cluster_dist * rarity * hub_pen
                if bridge > 0:
                    candidates.append((oid, bridge))

            candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = candidates[:max_per_block]

            for oid, bridge in candidates:
                left_id, right_id = (min(tid, oid), max(tid, oid))

                conn = self._connection_strength(
                    block_members.get(left_id, set()),
                    block_members.get(right_id, set()),
                )
                cdist = self._cluster_distance(
                    block_members.get(left_id, set()),
                    block_members.get(right_id, set()),
                )
                rarity = self._rarity_bonus(
                    block_members.get(left_id, set()),
                    block_members.get(right_id, set()),
                )

                self._db.execute(
                    """INSERT INTO bridge_edges
                       (left_block_id, right_block_id, connection_strength, cluster_distance,
                        rarity_bonus, bridge_strength, evidence_count)
                       VALUES (?, ?, ?, ?, ?, ?, 1)
                       ON CONFLICT(left_block_id, right_block_id) DO UPDATE SET
                           connection_strength = ?,
                           cluster_distance = ?,
                           rarity_bonus = ?,
                           bridge_strength = ?,
                           evidence_count = evidence_count + 1,
                           last_seen_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')""",
                    (
                        left_id, right_id, conn, cdist, rarity, bridge,
                        conn, cdist, rarity, bridge,
                    ),
                )

                self._db.execute(
                    """INSERT INTO block_adjacency
                       (left_block_id, right_block_id, adjacency_type, bridge_weight, evidence_count)
                       VALUES (?, ?, 'bridge', ?, 1)
                       ON CONFLICT(left_block_id, right_block_id, adjacency_type) DO UPDATE SET
                           bridge_weight = ?,
                           evidence_count = evidence_count + 1,
                           last_observed_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')""",
                    (left_id, right_id, bridge, bridge),
                )

                upserted += 1

        for tblock in touched_blocks:
            tid = tblock["id"]
            bridge_sum_row = self._db.query_one(
                """SELECT COALESCE(SUM(bridge_strength), 0.0) as s FROM bridge_edges
                   WHERE left_block_id=? OR right_block_id=?""",
                (tid, tid),
            )
            self._db.execute(
                "UPDATE forest_blocks SET external_bridge_strength=? WHERE id=?",
                (bridge_sum_row["s"], tid),
            )

        return upserted

    def get_bridge_strength(self, block_a_id: int, block_b_id: int) -> float:
        left_id, right_id = min(block_a_id, block_b_id), max(block_a_id, block_b_id)
        row = self._db.query_one(
            "SELECT bridge_strength FROM bridge_edges WHERE left_block_id=? AND right_block_id=?",
            (left_id, right_id),
        )
        return row["bridge_strength"] if row else 0.0

    def _connection_strength(self, members_a: set[str], members_b: set[str]) -> float:
        """Retrieval co-activation and shared tag overlap."""
        members_a = {m for m in members_a if m is not None}
        members_b = {m for m in members_b if m is not None}
        shared = members_a & members_b
        union = members_a | members_b
        if not union:
            return 0.0

        jaccard = len(shared) / len(union)

        tag_ids_a = [int(t) for t in members_a]
        tag_ids_b = [int(t) for t in members_b]

        co_retrieval = 0.0
        if tag_ids_a and tag_ids_b:
            placeholders_a = ",".join("?" * len(tag_ids_a))
            placeholders_b = ",".join("?" * len(tag_ids_b))
            row = self._db.query_one(
                f"""SELECT COUNT(DISTINCT a.retrieval_id) as c
                    FROM retrieval_tag_hits a
                    JOIN retrieval_tag_hits b ON a.retrieval_id = b.retrieval_id
                    WHERE a.tag_id IN ({placeholders_a}) AND b.tag_id IN ({placeholders_b})""",
                tuple(tag_ids_a + tag_ids_b),
            )
            total_row = self._db.query_one("SELECT COUNT(*) as c FROM retrieval_events")
            total = total_row["c"] if total_row else 1
            co_retrieval = (row["c"] / max(total, 1)) if row else 0.0

        return 0.5 * jaccard + 0.5 * co_retrieval

    def _cluster_distance(self, members_a: set[str], members_b: set[str]) -> float:
        """1 - Jaccard: higher when neighborhoods are more separated."""
        union = members_a | members_b
        if not union:
            return 0.0
        shared = members_a & members_b
        return 1.0 - (len(shared) / len(union))

    def _rarity_bonus(self, members_a: set[str], members_b: set[str]) -> float:
        """Reward non-generic crossings. Inverse of average tag frequency."""
        all_tags = list(members_a | members_b)
        if not all_tags:
            return 1.0

        total_row = self._db.query_one("SELECT COUNT(*) as c FROM memory_tags")
        total = total_row["c"] if total_row else 1

        tag_ids = [int(t) for t in all_tags]
        placeholders = ",".join("?" * len(tag_ids))
        rows = self._db.query(
            f"SELECT tag_id, COUNT(*) as c FROM memory_tags WHERE tag_id IN ({placeholders}) GROUP BY tag_id",
            tuple(tag_ids),
        )
        freq_map = {r["tag_id"]: r["c"] for r in rows}

        freqs = [freq_map.get(tid, 0) / max(total, 1) for tid in tag_ids]
        avg_freq = sum(freqs) / len(freqs) if freqs else 0.5
        return 1.0 / (1.0 + avg_freq * 10)

    def _hub_penalty(self, degree_a: int, degree_b: int, median_degree: int) -> float:
        """Penalize generic hub nodes. Penalty increases with degree above median."""
        if median_degree <= 0:
            median_degree = 1
        max_degree = max(degree_a, degree_b)
        if max_degree <= median_degree:
            return 1.0
        excess = (max_degree - median_degree) / median_degree
        return 1.0 / (1.0 + self._cfg.canopy_bridge_hub_penalty * excess)
