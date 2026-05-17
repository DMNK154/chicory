"""Forest reorganizer: coordinates co-occurrence and bridge optimizers.

The forest is the base layer — co-occurrence compresses local neighborhoods,
bridge preserves global traversability. The canopy grows on top of the forest.
Raw memories stay fixed; the forest reorganizes maps, not terrain.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from chicory.layer4.bridge_optimizer import BridgeOptimizer
from chicory.layer4.cooccurrence_optimizer import CooccurrenceOptimizer

if TYPE_CHECKING:
    from chicory.config import ChicoryConfig
    from chicory.db.engine import DatabaseEngine


class ForestReorganizer:

    def __init__(self, db: DatabaseEngine, config: ChicoryConfig) -> None:
        self._db = db
        self._cfg = config
        self.co_optimizer = CooccurrenceOptimizer(db, config)
        self.bridge_optimizer = BridgeOptimizer(db, config)

    def update_on_store(
        self,
        memory_id: str,
        tag_ids: list[int],
    ) -> list[str]:
        """Run forest reorganization after a memory store.

        Returns block_keys of all touched/created forest blocks.
        """
        changed = self.co_optimizer.update_from_scope(
            scope_type="store",
            activated_tag_ids=tag_ids,
            activated_memory_ids=[memory_id],
        )

        if changed:
            self.bridge_optimizer.update_bridges(changed)
            self._write_snapshot("store", memory_id, [memory_id], tag_ids, changed)

        return changed

    def update_on_retrieval(
        self,
        retrieval_id: int,
        result_memory_ids: list[str],
        activated_tag_ids: list[int],
    ) -> list[str]:
        """Run forest reorganization after a retrieval.

        Returns block_keys of all touched/created forest blocks.
        """
        changed = self.co_optimizer.update_from_scope(
            scope_type="retrieval",
            activated_tag_ids=activated_tag_ids,
            activated_memory_ids=result_memory_ids,
        )

        if changed:
            self.bridge_optimizer.update_bridges(changed)
            self._write_snapshot(
                "retrieval", str(retrieval_id),
                result_memory_ids, activated_tag_ids, changed,
            )

        return changed

    def update_on_sync_event(
        self,
        sync_event_id: int,
        involved_tag_ids: list[int],
        involved_memory_ids: list[str],
    ) -> list[str]:
        """Run forest reorganization after a synchronicity event."""
        changed = self.co_optimizer.update_from_scope(
            scope_type="sync_event",
            activated_tag_ids=involved_tag_ids,
            activated_memory_ids=involved_memory_ids,
        )

        if changed:
            self.bridge_optimizer.update_bridges(changed)
            self._write_snapshot(
                "sync_event", str(sync_event_id),
                involved_memory_ids, involved_tag_ids, changed,
            )

        return changed

    def get_block_bridge_strength(self, block_key: str) -> float:
        """Get the aggregate bridge strength for a forest block."""
        row = self._db.execute(
            "SELECT external_bridge_strength FROM forest_blocks WHERE block_key=?",
            (block_key,),
        ).fetchone()
        return row["external_bridge_strength"] if row else 0.0

    def get_block_co_density(self, block_key: str) -> float:
        return self.co_optimizer.get_block_density(block_key)

    def _write_snapshot(
        self,
        trigger_type: str,
        trigger_id: str,
        memory_ids: list[str],
        tag_ids: list[int],
        block_keys: list[str],
    ) -> None:
        co_count = self._db.query_one("SELECT COUNT(*) as c FROM cooccurrence_edges")["c"]
        bridge_count = self._db.query_one("SELECT COUNT(*) as c FROM bridge_edges")["c"]
        block_count = self._db.query_one("SELECT COUNT(*) as c FROM forest_blocks")["c"]

        block_ids = []
        for key in block_keys:
            row = self._db.execute(
                "SELECT id FROM forest_blocks WHERE block_key=?", (key,)
            ).fetchone()
            if row:
                block_ids.append(row["id"])

        self._db.execute(
            """INSERT INTO forest_snapshots
               (trigger_type, trigger_id, touched_memory_ids, touched_tag_ids,
                touched_block_ids, co_edge_count, bridge_edge_count, block_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trigger_type,
                trigger_id,
                json.dumps(memory_ids),
                json.dumps(tag_ids),
                json.dumps(block_ids),
                co_count,
                bridge_count,
                block_count,
            ),
        )
