"""Temporal tag episodes: drift-detected, revisitable tag-space clusters.

Sits between the global tag centroid (CentroidSubgraph) and per-memory
episodic edges (EpisodicTensor).  Tracks contiguous periods of coherent
tag-space activity as "episodes."

Drift detection:  On each store/retrieval, compute the centroid of the
operation's tags.  If cosine distance from the current episode exceeds
an adaptive threshold (Welford online variance), snapshot the episode
and start a new one — or revisit a previous episode if closer.

Sync boundaries:  Strong synchronicity events force episode boundaries,
marking the transition with the sync event ID.

Episodes form a revisitable graph (via transitions), not a timeline.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from chicory.config import ChicoryConfig
    from chicory.db.engine import DatabaseEngine
    from chicory.layer3.centroid_subgraph import CentroidSubgraph

log = logging.getLogger(__name__)


def _array_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).tobytes()


def _blob_to_array(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


class TemporalEpisodeTracker:

    def __init__(
        self,
        db: DatabaseEngine,
        config: ChicoryConfig,
        centroid_subgraph: CentroidSubgraph,
    ) -> None:
        self._db = db
        self._cfg = config
        self._centroids = centroid_subgraph
        self._lock = threading.Lock()
        self._current_episode_id: int | None = self._load_current_episode_id()

    def _load_current_episode_id(self) -> int | None:
        row = self._db.execute(
            "SELECT id FROM temporal_episodes WHERE status = 'active' "
            "ORDER BY last_active_at DESC LIMIT 1"
        ).fetchone()
        return row["id"] if row else None

    # ------------------------------------------------------------------
    # Centroid computation
    # ------------------------------------------------------------------

    def _compute_operation_centroid(self, tag_ids: list[int]) -> np.ndarray | None:
        if not tag_ids:
            return None
        centroids = self._centroids.get_centroids_batch(tag_ids)
        if len(centroids) < max(1, len(tag_ids) // 2):
            return None
        vecs = list(centroids.values())
        centroid = np.mean(vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        return centroid

    # ------------------------------------------------------------------
    # Adaptive threshold (Welford online variance)
    # ------------------------------------------------------------------

    def _get_adaptive_threshold(self, episode_id: int) -> float:
        row = self._db.execute(
            "SELECT mean_distance, variance_sum, distance_samples "
            "FROM temporal_episodes WHERE id = ?",
            (episode_id,),
        ).fetchone()
        if not row:
            return 0.3 + self._cfg.episode_drift_sigma * math.sqrt(0.01)

        n = row["distance_samples"]
        if n < self._cfg.episode_min_samples_for_adaptive:
            mean = 0.3
            var = 0.01
        else:
            mean = row["mean_distance"]
            var = row["variance_sum"] / (n - 1) if n > 1 else 0.01

        return mean + self._cfg.episode_drift_sigma * math.sqrt(max(var, 1e-8))

    def _update_distance_stats(self, episode_id: int, distance: float) -> None:
        row = self._db.execute(
            "SELECT mean_distance, variance_sum, distance_samples "
            "FROM temporal_episodes WHERE id = ?",
            (episode_id,),
        ).fetchone()
        if not row:
            return

        n = row["distance_samples"] + 1
        old_mean = row["mean_distance"]
        delta = distance - old_mean
        new_mean = old_mean + delta / n
        delta2 = distance - new_mean
        new_m2 = row["variance_sum"] + delta * delta2

        self._db.execute(
            "UPDATE temporal_episodes SET mean_distance = ?, variance_sum = ?, "
            "distance_samples = ? WHERE id = ?",
            (new_mean, new_m2, n, episode_id),
        )

    # ------------------------------------------------------------------
    # Episode CRUD
    # ------------------------------------------------------------------

    def _create_episode(self, op_centroid: np.ndarray, tag_ids: list[int]) -> int:
        cursor = self._db.execute(
            "INSERT INTO temporal_episodes (centroid, tag_ids) VALUES (?, ?)",
            (_array_to_blob(op_centroid), json.dumps(sorted(set(tag_ids)))),
        )
        self._db.connection.commit()
        return cursor.lastrowid

    def _snapshot_episode(self, episode_id: int) -> None:
        self._db.execute(
            "UPDATE temporal_episodes SET status = 'dormant', "
            "snapshot_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') "
            "WHERE id = ?",
            (episode_id,),
        )

    def _revisit_episode(
        self, episode_id: int, op_centroid: np.ndarray, tag_ids: list[int],
    ) -> None:
        row = self._db.execute(
            "SELECT centroid, tag_ids FROM temporal_episodes WHERE id = ?",
            (episode_id,),
        ).fetchone()
        if not row:
            return

        old_centroid = _blob_to_array(row["centroid"])
        alpha = self._cfg.episode_ema_alpha
        new_centroid = alpha * op_centroid + (1 - alpha) * old_centroid
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid /= norm

        old_tags = set(json.loads(row["tag_ids"]))
        merged_tags = sorted(old_tags | set(tag_ids))

        self._db.execute(
            "UPDATE temporal_episodes SET status = 'active', "
            "centroid = ?, tag_ids = ?, visit_count = visit_count + 1, "
            "last_active_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') "
            "WHERE id = ?",
            (_array_to_blob(new_centroid), json.dumps(merged_tags), episode_id),
        )

    def _update_episode_centroid(
        self, episode_id: int, op_centroid: np.ndarray, tag_ids: list[int],
    ) -> None:
        row = self._db.execute(
            "SELECT centroid, tag_ids FROM temporal_episodes WHERE id = ?",
            (episode_id,),
        ).fetchone()
        if not row:
            return

        old_centroid = _blob_to_array(row["centroid"])
        alpha = self._cfg.episode_ema_alpha
        new_centroid = alpha * op_centroid + (1 - alpha) * old_centroid
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid /= norm

        old_tags = set(json.loads(row["tag_ids"]))
        merged_tags = sorted(old_tags | set(tag_ids))

        self._db.execute(
            "UPDATE temporal_episodes SET centroid = ?, tag_ids = ?, "
            "operation_count = operation_count + 1, "
            "last_active_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') "
            "WHERE id = ?",
            (_array_to_blob(new_centroid), json.dumps(merged_tags), episode_id),
        )

    def _record_transition(
        self,
        from_id: int,
        to_id: int,
        transition_type: str,
        metadata: dict | None = None,
    ) -> None:
        self._db.execute(
            "INSERT INTO episode_transitions "
            "(from_episode_id, to_episode_id, transition_type, metadata) "
            "VALUES (?, ?, ?, ?)",
            (from_id, to_id, transition_type,
             json.dumps(metadata) if metadata else None),
        )

    def assign_memory(
        self, memory_id: str, episode_id: int, assignment_type: str = "store",
    ) -> None:
        self._db.execute(
            "INSERT OR IGNORE INTO memory_episode_assignments "
            "(memory_id, episode_id, assignment_type) VALUES (?, ?, ?)",
            (memory_id, episode_id, assignment_type),
        )

    # ------------------------------------------------------------------
    # Revisitation search
    # ------------------------------------------------------------------

    def _find_revisitable_episode(
        self, op_centroid: np.ndarray, threshold: float,
    ) -> int | None:
        rows = self._db.execute(
            "SELECT id, centroid FROM temporal_episodes "
            "WHERE status = 'dormant' "
            "ORDER BY last_active_at DESC LIMIT ?",
            (self._cfg.episode_revisit_max_candidates,),
        ).fetchall()

        if not rows:
            return None

        best_id: int | None = None
        best_dist = threshold

        for r in rows:
            c = _blob_to_array(r["centroid"])
            dist = 1.0 - float(np.dot(op_centroid, c))
            if dist < best_dist:
                best_dist = dist
                best_id = r["id"]

        return best_id

    # ------------------------------------------------------------------
    # Core drift/revisit logic
    # ------------------------------------------------------------------

    def _process_operation(
        self,
        op_centroid: np.ndarray,
        tag_ids: list[int],
        memory_ids: list[str],
        assignment_type: str,
    ) -> int | None:
        with self._lock:
            if self._current_episode_id is None:
                ep_id = self._create_episode(op_centroid, tag_ids)
                self._current_episode_id = ep_id
                for mid in memory_ids:
                    self.assign_memory(mid, ep_id, assignment_type)
                self._db.connection.commit()
                log.info("Created first episode %d with %d tags", ep_id, len(tag_ids))
                return ep_id

            row = self._db.execute(
                "SELECT centroid FROM temporal_episodes WHERE id = ?",
                (self._current_episode_id,),
            ).fetchone()
            if not row:
                ep_id = self._create_episode(op_centroid, tag_ids)
                self._current_episode_id = ep_id
                for mid in memory_ids:
                    self.assign_memory(mid, ep_id, assignment_type)
                self._db.connection.commit()
                return ep_id

            current_centroid = _blob_to_array(row["centroid"])
            distance = 1.0 - float(np.dot(op_centroid, current_centroid))
            threshold = self._get_adaptive_threshold(self._current_episode_id)

            if distance < threshold:
                self._update_episode_centroid(
                    self._current_episode_id, op_centroid, tag_ids,
                )
                self._update_distance_stats(self._current_episode_id, distance)
                for mid in memory_ids:
                    self.assign_memory(mid, self._current_episode_id, assignment_type)
                self._db.connection.commit()
                return self._current_episode_id

            old_id = self._current_episode_id

            revisit_id = self._find_revisitable_episode(op_centroid, threshold)
            if revisit_id is not None:
                self._snapshot_episode(old_id)
                self._revisit_episode(revisit_id, op_centroid, tag_ids)
                self._record_transition(old_id, revisit_id, "revisit")
                self._current_episode_id = revisit_id
                for mid in memory_ids:
                    self.assign_memory(mid, revisit_id, assignment_type)
                self._db.connection.commit()
                log.info(
                    "Revisited episode %d from %d (distance=%.3f, threshold=%.3f)",
                    revisit_id, old_id, distance, threshold,
                )
                return revisit_id

            self._snapshot_episode(old_id)
            new_id = self._create_episode(op_centroid, tag_ids)
            self._record_transition(old_id, new_id, "drift")
            self._current_episode_id = new_id
            for mid in memory_ids:
                self.assign_memory(mid, new_id, assignment_type)
            self._db.connection.commit()
            log.info(
                "New episode %d (drift from %d, distance=%.3f, threshold=%.3f)",
                new_id, old_id, distance, threshold,
            )
            return new_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_on_store(self, memory_id: str, tag_ids: list[int]) -> int | None:
        op_centroid = self._compute_operation_centroid(tag_ids)
        if op_centroid is None:
            return None
        return self._process_operation(
            op_centroid, tag_ids, [memory_id], "store",
        )

    def update_on_retrieval(
        self, memory_ids: list[str], tag_ids: list[int],
    ) -> int | None:
        if not tag_ids:
            return None
        op_centroid = self._compute_operation_centroid(tag_ids)
        if op_centroid is None:
            return None
        return self._process_operation(
            op_centroid, tag_ids, memory_ids, "retrieval",
        )

    def force_sync_boundary(
        self,
        sync_event_id: int,
        involved_tag_ids: list[int],
        effective_strength: float,
    ) -> int | None:
        if effective_strength < self._cfg.episode_sync_boundary_strength:
            return None

        op_centroid = self._compute_operation_centroid(involved_tag_ids)
        if op_centroid is None:
            return None

        with self._lock:
            if self._current_episode_id is None:
                ep_id = self._create_episode(op_centroid, involved_tag_ids)
                self._current_episode_id = ep_id
                self._db.connection.commit()
                return ep_id

            old_id = self._current_episode_id
            threshold = self._get_adaptive_threshold(old_id)

            revisit_id = self._find_revisitable_episode(op_centroid, threshold)
            if revisit_id is not None:
                self._snapshot_episode(old_id)
                self._revisit_episode(revisit_id, op_centroid, involved_tag_ids)
                self._record_transition(
                    old_id, revisit_id, "sync_boundary",
                    {"sync_event_id": sync_event_id},
                )
                self._current_episode_id = revisit_id
                self._db.connection.commit()
                log.info(
                    "Sync boundary: revisited episode %d from %d (event=%d)",
                    revisit_id, old_id, sync_event_id,
                )
                return revisit_id

            self._snapshot_episode(old_id)
            new_id = self._create_episode(op_centroid, involved_tag_ids)
            self._record_transition(
                old_id, new_id, "sync_boundary",
                {"sync_event_id": sync_event_id},
            )
            self._current_episode_id = new_id
            self._db.connection.commit()
            log.info(
                "Sync boundary: new episode %d from %d (event=%d)",
                new_id, old_id, sync_event_id,
            )
            return new_id

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def lifecycle_pass(self) -> dict[str, int]:
        dormant_hours = self._cfg.episode_dormant_after_hours
        archive_hours = self._cfg.episode_archive_after_hours

        cursor = self._db.execute(
            "UPDATE temporal_episodes SET status = 'dormant', "
            "snapshot_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') "
            "WHERE status = 'active' AND id != ? "
            "AND last_active_at < strftime('%Y-%m-%dT%H:%M:%f', 'now', ? || ' hours')",
            (self._current_episode_id or -1, f"-{dormant_hours}"),
        )
        dormant = cursor.rowcount

        cursor = self._db.execute(
            "UPDATE temporal_episodes SET status = 'archived' "
            "WHERE status = 'dormant' "
            "AND last_active_at < strftime('%Y-%m-%dT%H:%M:%f', 'now', ? || ' hours')",
            (f"-{archive_hours}",),
        )
        archived = cursor.rowcount

        if dormant or archived:
            self._db.connection.commit()
            log.info("Episode lifecycle: %d dormant, %d archived", dormant, archived)

        return {"dormant": dormant, "archived": archived}

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def bootstrap(self, batch_size: int = 500) -> int:
        log.info("Bootstrapping temporal episodes from existing memories...")

        rows = self._db.execute(
            "SELECT m.id, m.created_at FROM memories m "
            "WHERE m.is_archived = 0 "
            "ORDER BY m.created_at ASC"
        ).fetchall()
        if not rows:
            return 0

        all_mem_tags: dict[str, list[int]] = {}
        tag_rows = self._db.execute(
            "SELECT memory_id, tag_id FROM memory_tags ORDER BY memory_id"
        ).fetchall()
        for r in tag_rows:
            all_mem_tags.setdefault(r["memory_id"], []).append(r["tag_id"])

        count = 0
        for i, r in enumerate(rows):
            mid = r["id"]
            tag_ids = all_mem_tags.get(mid, [])
            if not tag_ids:
                continue
            result = self.update_on_store(mid, tag_ids)
            if result is not None:
                count += 1
            if (i + 1) % batch_size == 0:
                log.info("  bootstrapped %d / %d memories", i + 1, len(rows))

        episode_count = self._db.execute(
            "SELECT COUNT(*) as c FROM temporal_episodes"
        ).fetchone()["c"]
        log.info(
            "Episode bootstrap complete: %d memories processed, %d episodes created",
            count, episode_count,
        )
        return episode_count

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get_current_episode(self) -> dict | None:
        if self._current_episode_id is None:
            return None
        row = self._db.execute(
            "SELECT * FROM temporal_episodes WHERE id = ?",
            (self._current_episode_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "tag_ids": json.loads(row["tag_ids"]),
            "status": row["status"],
            "visit_count": row["visit_count"],
            "operation_count": row["operation_count"],
            "created_at": row["created_at"],
            "last_active_at": row["last_active_at"],
            "mean_distance": row["mean_distance"],
            "distance_samples": row["distance_samples"],
        }

    def get_episode_graph(self, limit: int = 50) -> dict:
        episodes = self._db.execute(
            "SELECT id, tag_ids, status, visit_count, operation_count, "
            "created_at, last_active_at, snapshot_at "
            "FROM temporal_episodes "
            "ORDER BY last_active_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

        ep_ids = [e["id"] for e in episodes]
        if not ep_ids:
            return {"episodes": [], "transitions": []}

        placeholders = ",".join("?" * len(ep_ids))
        transitions = self._db.execute(
            f"SELECT * FROM episode_transitions "
            f"WHERE from_episode_id IN ({placeholders}) "
            f"OR to_episode_id IN ({placeholders}) "
            f"ORDER BY transition_at DESC",
            ep_ids + ep_ids,
        ).fetchall()

        return {
            "episodes": [
                {
                    "id": e["id"],
                    "tag_ids": json.loads(e["tag_ids"]),
                    "status": e["status"],
                    "visit_count": e["visit_count"],
                    "operation_count": e["operation_count"],
                    "created_at": e["created_at"],
                    "last_active_at": e["last_active_at"],
                    "snapshot_at": e["snapshot_at"],
                }
                for e in episodes
            ],
            "transitions": [
                {
                    "from": t["from_episode_id"],
                    "to": t["to_episode_id"],
                    "type": t["transition_type"],
                    "at": t["transition_at"],
                    "metadata": json.loads(t["metadata"]) if t["metadata"] else None,
                }
                for t in transitions
            ],
        }

    def get_episode_for_memory(self, memory_id: str) -> int | None:
        row = self._db.execute(
            "SELECT episode_id FROM memory_episode_assignments "
            "WHERE memory_id = ? ORDER BY assigned_at DESC LIMIT 1",
            (memory_id,),
        ).fetchone()
        return row["episode_id"] if row else None

    def get_cross_episode_bridges(
        self, episode_a_id: int, episode_b_id: int, top_k: int = 20,
    ) -> list[dict]:
        rows = self._db.execute(
            """SELECT mrt.* FROM memory_relational_tensor mrt
               JOIN memory_episode_assignments mea_a
                 ON mrt.memory_a_id = mea_a.memory_id
               JOIN memory_episode_assignments mea_b
                 ON mrt.memory_b_id = mea_b.memory_id
               WHERE mea_a.episode_id = ? AND mea_b.episode_id = ?
                 AND mrt.edge_status != 'archived'
               ORDER BY mrt.semantic_strength + mrt.tag_projected_strength
                        + mrt.bridge_strength DESC
               LIMIT ?""",
            (episode_a_id, episode_b_id, top_k),
        ).fetchall()
        return [dict(r) for r in rows]
