"""Prime Ramsey lattice engine for synchronicity resonance detection."""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.layer1.embedding_engine import EmbeddingEngine
from chicory.layer1.tag_manager import TagManager
from chicory.layer3.poincare import PoincareProjection
from chicory.models.lattice import GlyphPosition, LatticePosition, Resonance, VoidProfile
from chicory.models.synchronicity import SynchronicityEvent


class SynchronicityEngine:
    """Organizes synchronicity events on a prime Ramsey lattice to detect resonance.

    Instead of clustering by tag overlap (Jaccard distance), this engine places
    events on a circle based on their semantic embedding and checks whether they
    occupy the same structural position across multiple incommensurate prime scales.

    Events that share slots at many primes are "resonant" — structurally entangled
    in a way that tag-overlap clustering cannot detect.
    """

    def __init__(
        self,
        config: ChicoryConfig,
        db: DatabaseEngine,
        embedding_engine: EmbeddingEngine,
        tag_manager: TagManager,
        glyph_encoder: object | None = None,
    ) -> None:
        self._config = config
        self._db = db
        self._embeddings = embedding_engine
        self._tags = tag_manager
        self._pca_basis: Optional[np.ndarray] = None
        self._glyph_encoder = glyph_encoder  # GlyphEncoder or None
        self._poincare = PoincareProjection(
            curvature=config.poincare_curvature,
            max_radius=config.poincare_max_radius,
        )

    # ── Lattice gate ───────────────────────────────────────────────

    def _load_resonant_tag_set(self) -> set[tuple[int, int]]:
        """Load glyph-resonant tag pairs as a set of (lo, hi) tuples.

        Used to gate tag×tag nested loops so only structurally resonant
        pairs are processed, bounding iteration to O(resonances).
        """
        rows = self._db.execute(
            "SELECT tag_a_id, tag_b_id FROM glyph_resonances"
        ).fetchall()
        return {(r["tag_a_id"], r["tag_b_id"]) for r in rows}

    def _load_co_retrieval_pairs(self) -> set[tuple[int, int]]:
        """Load tag pairs that have co-retrieval evidence from centroid_edges."""
        rows = self._db.execute(
            "SELECT tag_a_id, tag_b_id FROM centroid_edges "
            "WHERE co_retrieval_count > 0"
        ).fetchall()
        return {(r["tag_a_id"], r["tag_b_id"]) for r in rows}

    # ── Placement ────────────────────────────────────────────────────

    def place_event(self, event: SynchronicityEvent) -> Optional[LatticePosition]:
        """Place a synchronicity event on the prime Ramsey lattice.

        Computes angular position from tag embedding centroids, assigns slots
        at each prime scale, computes Poincaré disk coordinates with
        Ramsey-derived depth, and persists the position.  Idempotent — placing
        the same event twice returns the existing position.

        Does NOT detect resonances — use place_events_batch for that.
        For reseed(), rebuild_tensor() handles resonances after all placements.
        """
        if event.id is None:
            return None

        existing = self._db.execute(
            "SELECT id FROM lattice_positions WHERE sync_event_id = ?",
            (event.id,),
        ).fetchone()
        if existing:
            return self._load_position(event.id)

        result = self._compute_projection(event)
        if result is None:
            return None
        angle, projected_2d = result

        prime_slots = self._compute_prime_slots(angle)
        prime_slots_json = json.dumps(prime_slots)

        slot_pops, total = self._get_slot_populations()
        poincare = self._compute_poincare_coords(
            projected_2d, prime_slots, slot_pops, total,
        )
        px, py = float(poincare[0]), float(poincare[1])

        self._db.execute(
            """
            INSERT INTO lattice_positions
                (sync_event_id, angle, prime_slots, poincare_x, poincare_y)
            VALUES (?, ?, ?, ?, ?)
            """,
            (event.id, angle, prime_slots_json, px, py),
        )
        self._db.connection.commit()

        return LatticePosition(
            sync_event_id=event.id,
            angle=angle,
            prime_slots=prime_slots_json,
            poincare_x=px,
            poincare_y=py,
        )

    def place_events_batch(
        self, events: list[SynchronicityEvent],
    ) -> list[LatticePosition]:
        """Place multiple events and detect resonances in a single vectorized pass.

        Phase 1: Place each event (angle, prime slots, Poincaré coords, INSERT).
        Phase 2: Vectorized resonance detection for all newly placed events
                 using numpy broadcast, capped per event, batch SQL writes.
        """
        self._pca_basis = None  # Invalidate to pick up any new embeddings

        # Pre-fetch slot populations once for the whole batch
        slot_pops, total = self._get_slot_populations()

        placed: list[tuple[LatticePosition, SynchronicityEvent]] = []
        for event in events:
            if event.id is None:
                continue

            existing = self._db.execute(
                "SELECT id FROM lattice_positions WHERE sync_event_id = ?",
                (event.id,),
            ).fetchone()
            if existing:
                continue

            result = self._compute_projection(event)
            if result is None:
                continue
            angle, projected_2d = result

            prime_slots = self._compute_prime_slots(angle)
            prime_slots_json = json.dumps(prime_slots)

            poincare = self._compute_poincare_coords(
                projected_2d, prime_slots, slot_pops, total,
            )
            px, py = float(poincare[0]), float(poincare[1])

            self._db.execute(
                """
                INSERT INTO lattice_positions
                    (sync_event_id, angle, prime_slots, poincare_x, poincare_y)
                VALUES (?, ?, ?, ?, ?)
                """,
                (event.id, angle, prime_slots_json, px, py),
            )

            # Update running population counts for subsequent events in batch
            for p_str, slot in prime_slots.items():
                p = int(p_str) if isinstance(p_str, str) else p_str
                if p not in slot_pops:
                    slot_pops[p] = {}
                slot_pops[p][slot] = slot_pops[p].get(slot, 0) + 1
            total += 1

            pos = LatticePosition(
                sync_event_id=event.id,
                angle=angle,
                prime_slots=prime_slots_json,
                poincare_x=px,
                poincare_y=py,
            )
            placed.append((pos, event))

        if placed:
            self._db.connection.commit()
            self._find_and_persist_resonances(placed)

        return [pos for pos, _ in placed]

    def invalidate_pca_cache(self) -> None:
        """Force recomputation of PCA basis on next placement."""
        self._pca_basis = None

    def reseed(self) -> int:
        """Clear all lattice positions and re-place every synchronicity event.

        Use after changing lattice_primes to recompute all prime slots
        with the new configuration.  For Poincaré-only recomputation
        (preserving angles and slots), use reseed_poincare() instead.

        Returns the number of events placed.
        """
        self._pca_basis = None

        # Wipe existing positions
        self._db.execute("DELETE FROM lattice_positions")
        self._db.connection.commit()

        # Fetch all sync events
        rows = self._db.execute(
            "SELECT * FROM synchronicity_events ORDER BY id"
        ).fetchall()

        placed = 0
        for r in rows:
            event = SynchronicityEvent(
                id=r["id"],
                detected_at=datetime.fromisoformat(r["detected_at"]),
                event_type=r["event_type"],
                description=r["description"],
                strength=r["strength"],
                quadrant=r["quadrant"],
                involved_tags=r["involved_tags"],
                involved_memories=r["involved_memories"],
                trigger_retrieval_id=r["trigger_retrieval_id"],
                acknowledged=bool(r["acknowledged"]),
                last_reinforced=datetime.fromisoformat(r["last_reinforced"]) if r["last_reinforced"] else None,
                reinforcement_count=r["reinforcement_count"],
            )
            pos = self.place_event(event)
            if pos is not None:
                placed += 1

        self.rebuild_tensor()
        return placed

    def reseed_poincare(self) -> int:
        """Recompute Poincaré disk coordinates for all existing lattice positions.

        Re-projects each event's embedding centroid through the current PCA
        basis and applies Ramsey-depth adjustment using current slot populations.
        Angles and prime_slots are preserved — only poincare_x/y change.

        Use after migration backfill (which sets placeholder coords) or when
        the embedding model / PCA basis has changed.

        Returns the number of positions updated.
        """
        all_cached = self._embeddings.get_all_cached()
        if not all_cached:
            return 0

        self._pca_basis = None
        self._ensure_pca_basis(all_cached)

        rows = self._db.execute(
            """
            SELECT lp.sync_event_id, lp.prime_slots, se.involved_tags
            FROM lattice_positions lp
            JOIN synchronicity_events se ON se.id = lp.sync_event_id
            """
        ).fetchall()

        if not rows:
            return 0

        slot_pops, total = self._get_slot_populations()
        updated = 0

        for r in rows:
            tag_ids = json.loads(r["involved_tags"])
            if not tag_ids:
                continue

            tag_centroids: list[np.ndarray] = []
            for tag_id in tag_ids:
                mem_rows = self._db.query(
                    "SELECT memory_id FROM memory_tags WHERE tag_id = ?",
                    (tag_id,),
                )
                vecs = [
                    all_cached[mr["memory_id"]]
                    for mr in mem_rows if mr["memory_id"] in all_cached
                ]
                if vecs:
                    tag_centroids.append(np.mean(vecs, axis=0))

            if not tag_centroids:
                continue

            event_centroid = np.mean(tag_centroids, axis=0)

            if self._pca_basis is not None:
                projected = self._pca_basis @ event_centroid
            else:
                if len(event_centroid) < 2:
                    continue
                projected = event_centroid[:2]

            projected_2d = np.asarray(projected, dtype=np.float32)
            prime_slots = {
                int(k): v for k, v in json.loads(r["prime_slots"]).items()
            }

            poincare = self._compute_poincare_coords(
                projected_2d, prime_slots, slot_pops, total,
            )
            px, py = float(poincare[0]), float(poincare[1])

            self._db.execute(
                """
                UPDATE lattice_positions
                SET poincare_x = ?, poincare_y = ?
                WHERE sync_event_id = ?
                """,
                (px, py, r["sync_event_id"]),
            )
            updated += 1

        self._db.connection.commit()
        return updated

    # ── Resonance detection ──────────────────────────────────────────

    def find_resonances(
        self, min_shared_primes: Optional[int] = None
    ) -> list[Resonance]:
        """Find event pairs sharing slots across multiple prime scales.

        Strength = sum(log(p)) for shared primes — the information-theoretic
        surprise of the coincidence.  Returns resonances sorted strongest first.
        """
        if min_shared_primes is None:
            min_shared_primes = self._config.lattice_min_resonance_primes

        rows = self._db.execute(
            "SELECT sync_event_id, angle, prime_slots FROM lattice_positions"
        ).fetchall()

        if len(rows) < 2:
            return []

        positions: list[tuple[int, float, dict[int, int]]] = []
        for r in rows:
            slots = {int(k): v for k, v in json.loads(r["prime_slots"]).items()}
            positions.append((r["sync_event_id"], r["angle"], slots))

        primes = self._config.lattice_primes
        resonances: list[Resonance] = []

        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                id_a, _, slots_a = positions[i]
                id_b, _, slots_b = positions[j]

                shared = [p for p in primes if slots_a.get(p) == slots_b.get(p)]
                if len(shared) < min_shared_primes:
                    continue

                strength = sum(math.log(p) for p in shared)
                chance = math.exp(-strength)

                description = (
                    f"Events {id_a} and {id_b} resonate across {len(shared)} "
                    f"prime scales ({', '.join(str(p) for p in shared)}). "
                    f"Surprise: {strength:.1f} nats. "
                    f"Chance probability: {chance:.2e}"
                )

                resonances.append(Resonance(
                    event_ids=json.dumps([id_a, id_b]),
                    shared_primes=json.dumps(shared),
                    resonance_strength=strength,
                    description=description,
                ))

        resonances.sort(key=lambda r: r.resonance_strength, reverse=True)
        return resonances

    # ── Void profiling ───────────────────────────────────────────────

    def compute_void_profile(self) -> Optional[VoidProfile]:
        """Characterize the central void of the prime Ramsey lattice.

        Finds events nearest to the circular centroid and extracts their tags.
        The void is the thing all synchronicities orbit but never directly name.

        Returns None if fewer than 3 events are placed.
        """
        rows = self._db.execute(
            "SELECT sync_event_id, angle FROM lattice_positions ORDER BY angle"
        ).fetchall()

        if len(rows) < 3:
            return None

        angles = [r["angle"] for r in rows]
        event_ids = [r["sync_event_id"] for r in rows]

        # Circular mean
        sin_sum = sum(math.sin(a) for a in angles)
        cos_sum = sum(math.cos(a) for a in angles)
        centroid_angle = math.atan2(sin_sum, cos_sum)
        if centroid_angle < 0:
            centroid_angle += 2 * math.pi

        # Angular distance from centroid (handling wraparound)
        distances = []
        for a in angles:
            d = abs(a - centroid_angle)
            d = min(d, 2 * math.pi - d)
            distances.append(d)

        # Inner ring: closest void_radius fraction of events
        n_inner = max(1, int(len(angles) * self._config.lattice_void_radius))
        sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
        inner_indices = sorted_indices[:n_inner]

        # Gather tags from inner-ring events
        tag_counts: dict[str, int] = defaultdict(int)
        edge_angles_list: list[float] = []

        for idx in inner_indices:
            eid = event_ids[idx]
            edge_angles_list.append(angles[idx])

            row = self._db.execute(
                "SELECT involved_tags FROM synchronicity_events WHERE id = ?",
                (eid,),
            ).fetchone()
            if row and row["involved_tags"]:
                tag_ids = json.loads(row["involved_tags"])
                for tid in tag_ids:
                    try:
                        tag = self._tags.get_by_id(tid)
                        tag_counts[tag.name] += 1
                    except Exception:
                        continue

        sorted_tags = sorted(
            tag_counts.keys(), key=lambda t: tag_counts[t], reverse=True
        )
        void_radius = (
            max(distances[i] for i in inner_indices) if inner_indices else 0.0
        )

        description = (
            f"The lattice void is centered at angle {centroid_angle:.3f} rad. "
            f"{n_inner} events orbit within radius {void_radius:.3f} rad. "
            f"Edge themes: {', '.join(sorted_tags[:5]) or 'none'}. "
            f"These themes circle the void — present at its edge but never at "
            f"its center."
        )

        return VoidProfile(
            edge_tags=json.dumps(sorted_tags[:10]),
            edge_angles=json.dumps([round(a, 4) for a in edge_angles_list]),
            void_radius=void_radius,
            description=description,
        )

    # ── State / tool handler ─────────────────────────────────────────

    def get_lattice_state(self, max_positions: int = 50) -> dict:
        """Return the lattice state for the tool handler.

        Returns a summary with counts plus the top positions by strength
        (capped at *max_positions*) to avoid OOM on large lattices.
        """
        position_count = self._db.execute(
            "SELECT COUNT(*) as cnt FROM lattice_positions"
        ).fetchone()["cnt"]

        rows = self._db.execute(
            """
            SELECT lp.sync_event_id, lp.angle, lp.prime_slots,
                   lp.poincare_x, lp.poincare_y, lp.placed_at,
                   se.event_type, se.description AS event_description, se.strength
            FROM lattice_positions lp
            JOIN synchronicity_events se ON se.id = lp.sync_event_id
            ORDER BY se.strength DESC
            LIMIT ?
            """,
            (max_positions,),
        ).fetchall()

        positions = []
        for r in rows:
            entry = {
                "sync_event_id": r["sync_event_id"],
                "angle": round(r["angle"], 4),
                "prime_slots": json.loads(r["prime_slots"]),
                "placed_at": r["placed_at"],
                "event_type": r["event_type"],
                "event_description": r["event_description"],
                "event_strength": round(r["strength"], 3),
            }
            if r["poincare_x"] is not None:
                entry["poincare"] = {
                    "x": round(r["poincare_x"], 6),
                    "y": round(r["poincare_y"], 6),
                    "radius": round(
                        math.sqrt(r["poincare_x"] ** 2 + r["poincare_y"] ** 2), 6
                    ),
                }
            positions.append(entry)

        resonance_rows = self._db.execute(
            """
            SELECT event_ids, shared_primes, resonance_strength, description
            FROM resonances
            ORDER BY resonance_strength DESC
            LIMIT 1000
            """
        ).fetchall()

        resonance_count = self._db.execute(
            "SELECT COUNT(*) as cnt FROM resonances"
        ).fetchone()["cnt"]

        void = self.compute_void_profile()

        # Glyph lattice state
        glyph_rows = self._db.execute(
            """
            SELECT gp.tag_id, gp.angle, gp.prime_slots, gp.glyph_vector,
                   gp.placed_at, t.name AS tag_name
            FROM glyph_positions gp
            JOIN tags t ON t.id = gp.tag_id
            ORDER BY gp.angle
            """
        ).fetchall()

        glyph_res_rows = self._db.execute(
            """
            SELECT gr.tag_a_id, gr.tag_b_id, gr.shared_primes,
                   gr.resonance_strength,
                   ta.name AS tag_a_name, tb.name AS tag_b_name
            FROM glyph_resonances gr
            JOIN tags ta ON ta.id = gr.tag_a_id
            JOIN tags tb ON tb.id = gr.tag_b_id
            ORDER BY gr.resonance_strength DESC
            LIMIT 500
            """
        ).fetchall()

        glyph_res_count = self._db.execute(
            "SELECT COUNT(*) as cnt FROM glyph_resonances"
        ).fetchone()["cnt"]

        return {
            "positions": positions,
            "position_count": position_count,
            "positions_returned": len(positions),
            "resonances": [
                {
                    "event_ids": json.loads(r["event_ids"]),
                    "shared_primes": json.loads(r["shared_primes"]),
                    "strength": round(r["resonance_strength"], 3),
                    "description": r["description"],
                }
                for r in resonance_rows
            ],
            "resonance_count": resonance_count,
            "void_profile": {
                "edge_tags": json.loads(void.edge_tags),
                "void_radius": round(void.void_radius, 4),
                "description": void.description,
            }
            if void
            else None,
            "primes_used": self._config.lattice_primes,
            "glyph_lattice": {
                "positions": [
                    {
                        "tag_id": r["tag_id"],
                        "tag_name": r["tag_name"],
                        "angle": round(r["angle"], 4),
                        "prime_slots": json.loads(r["prime_slots"]),
                        "glyph_vector": np.frombuffer(r["glyph_vector"], dtype=np.float32).tolist(),
                        "placed_at": r["placed_at"],
                    }
                    for r in glyph_rows
                ],
                "position_count": len(glyph_rows),
                "resonances": [
                    {
                        "tag_a": r["tag_a_name"],
                        "tag_b": r["tag_b_name"],
                        "shared_primes": json.loads(r["shared_primes"]),
                        "strength": round(r["resonance_strength"], 3),
                    }
                    for r in glyph_res_rows
                ],
                "resonance_count": glyph_res_count,
            },
        }

    # ── Tag relational tensor ──────────────────────────────────────

    def _find_and_persist_resonances(
        self,
        new_positions: list[tuple[LatticePosition, SynchronicityEvent]],
    ) -> None:
        """Vectorized resonance detection for newly placed events.

        1. Load all lattice positions into a numpy slot matrix (one-time)
        2. For each new event, broadcast-compare against all positions
        3. Cap resonances per event (top-K strongest) to prevent table explosion
        4. Aggregate tensor updates per resonant tag pairs only (glyph-gated)
        """
        resonant_set = self._load_resonant_tag_set()
        min_shared = self._config.lattice_min_resonance_primes
        primes = self._config.lattice_primes
        n_primes = len(primes)
        max_per_event = n_primes // 2
        min_angular_dist = self._compute_proximity_threshold(primes, min_shared)

        new_event_id_set = {event.id for _, event in new_positions}
        rows = self._db.query(
            """
            SELECT lp.sync_event_id, lp.angle, lp.prime_slots,
                   lp.poincare_x, lp.poincare_y, se.involved_tags
            FROM lattice_positions lp
            JOIN synchronicity_events se ON se.id = lp.sync_event_id
            """,
        )

        if len(rows) < 2:
            return

        # Build slot matrix: (n_positions, n_primes) + Poincaré coords
        all_ids: list[int] = []
        all_angles = np.zeros(len(rows), dtype=np.float64)
        all_poincare = np.zeros((len(rows), 2), dtype=np.float32)
        all_tags: dict[int, set[int]] = {}
        id_to_idx: dict[int, int] = {}
        slot_matrix = np.zeros((len(rows), n_primes), dtype=np.int32)
        has_poincare = False

        for i, r in enumerate(rows):
            eid = r["sync_event_id"]
            all_ids.append(eid)
            all_angles[i] = r["angle"]
            if r["poincare_x"] is not None:
                all_poincare[i, 0] = r["poincare_x"]
                all_poincare[i, 1] = r["poincare_y"]
                has_poincare = True
            id_to_idx[eid] = i
            all_tags[eid] = set(json.loads(r["involved_tags"]))
            slots = json.loads(r["prime_slots"])
            for j, p in enumerate(primes):
                slot_matrix[i, j] = slots.get(str(p), slots.get(p, -1))

        log_primes = np.array([math.log(p) for p in primes], dtype=np.float64)
        new_event_ids = new_event_id_set

        resonance_rows: list[tuple] = []
        tensor_agg: dict[tuple[int, int], float] = {}

        for _pos, event in new_positions:
            idx = id_to_idx.get(event.id)
            if idx is None:
                continue

            new_vec = slot_matrix[idx]  # (n_primes,)

            # Column-wise accumulation: avoids materializing (n_all, n_primes)
            # bool matrix.  Each column compare is (n_all,) bool → sum into
            # int32 counts one prime at a time.  Memory: O(n_all) not O(n_all × n_primes).
            shared_counts = np.zeros(len(all_ids), dtype=np.int32)
            for j in range(n_primes):
                shared_counts += (slot_matrix[:, j] == new_vec[j])
            shared_counts[idx] = 0  # exclude self

            # Filter by threshold
            candidate_indices = np.where(shared_counts >= min_shared)[0]
            if len(candidate_indices) == 0:
                continue

            # Filter out angular-proximity artifacts
            angular_diffs = np.abs(all_angles[candidate_indices] - all_angles[idx])
            angular_diffs = np.minimum(angular_diffs, 2.0 * math.pi - angular_diffs)
            far_enough = angular_diffs >= min_angular_dist
            candidate_indices = candidate_indices[far_enough]
            if len(candidate_indices) == 0:
                continue

            # Compute match matrix only for candidates (small subset)
            cand_matches = slot_matrix[candidate_indices] == new_vec  # (n_cand, n_primes)

            # Compute strengths: sum of log(p) for shared primes
            strengths = (
                cand_matches.astype(np.float64) * log_primes
            ).sum(axis=1)

            # Cap: keep only top-K strongest
            if len(candidate_indices) > max_per_event:
                top_k = np.argsort(strengths)[-max_per_event:]
                candidate_indices = candidate_indices[top_k]
                strengths = strengths[top_k]
                cand_matches = cand_matches[top_k]

            new_tag_ids = all_tags.get(event.id, set())

            for ci_pos, ci in enumerate(candidate_indices):
                other_id = all_ids[ci]

                # Avoid double-counting for new-to-new pairs
                if other_id in new_event_ids and event.id > other_id:
                    continue

                strength = float(strengths[ci_pos])
                shared = [
                    primes[j] for j in range(n_primes)
                    if cand_matches[ci_pos, j]
                ]

                id_a = min(event.id, other_id)
                id_b = max(event.id, other_id)
                chance = math.exp(-strength)
                description = (
                    f"Events {id_a} and {id_b} resonate across {len(shared)} "
                    f"prime scales ({', '.join(str(p) for p in shared)}). "
                    f"Surprise: {strength:.1f} nats. "
                    f"Chance probability: {chance:.2e}"
                )

                if has_poincare:
                    hdist = self._poincare.distance(
                        all_poincare[idx], all_poincare[ci],
                    )
                    description += f" Hyperbolic distance: {hdist:.3f}."

                resonance_rows.append((
                    id_a, id_b, json.dumps([id_a, id_b]),
                    json.dumps(shared), strength, description,
                ))

                # Aggregate tensor: only for glyph-resonant tag pairs
                other_tag_ids = all_tags.get(other_id, set())
                for ta in new_tag_ids:
                    for tb in other_tag_ids:
                        key = (min(ta, tb), max(ta, tb))
                        if key[0] == key[1] or key not in resonant_set:
                            continue
                        tensor_agg[key] = max(
                            tensor_agg.get(key, 0.0), strength,
                        )

        # Batch persist resonances
        if resonance_rows:
            self._db.executemany(
                """
                INSERT INTO resonances
                    (event_a_id, event_b_id, event_ids,
                     shared_primes, resonance_strength, description)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(event_a_id, event_b_id) DO UPDATE SET
                    shared_primes = excluded.shared_primes,
                    resonance_strength = excluded.resonance_strength,
                    description = excluded.description,
                    detected_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                """,
                resonance_rows,
            )

        # Batch persist tensor updates (aggregated — one row per tag pair)
        if tensor_agg:
            tensor_rows = [
                (key_a, key_b, strength, "[]")
                for (key_a, key_b), strength in tensor_agg.items()
            ]
            self._db.executemany(
                """
                INSERT INTO tag_relational_tensor
                    (tag_a_id, tag_b_id, synchronicity_strength, memory_ids)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tag_a_id, tag_b_id) DO UPDATE SET
                    synchronicity_strength = MAX(
                        tag_relational_tensor.synchronicity_strength,
                        excluded.synchronicity_strength
                    ),
                    updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                """,
                tensor_rows,
            )

        if resonance_rows or tensor_agg:
            self._db.connection.commit()

    def update_cooccurrence_tensor(self) -> int:
        """Populate the co-occurrence network from memory + retrieval PMI.

        Combines ingest-time memory co-occurrence with retrieval-time
        co-occurrence.  Only glyph-resonant tag pairs are considered,
        keeping the tensor bounded by lattice structure.  Memory pairs
        are further gated by a retention score combining PMI with forest
        bridge strength.

        Returns the number of tensor entries created/updated.
        """
        resonant_set = self._load_resonant_tag_set()
        if not resonant_set:
            return 0

        # ── Memory co-occurrence PMI (ingest-time) ─────────────────
        total_row = self._db.execute(
            "SELECT COUNT(DISTINCT memory_id) as cnt FROM memory_tags"
        ).fetchone()
        total_memories = total_row["cnt"] if total_row else 0

        mem_tag_counts: dict[int, int] = {}
        rows = self._db.execute(
            "SELECT tag_id, COUNT(DISTINCT memory_id) as cnt FROM memory_tags GROUP BY tag_id"
        ).fetchall()
        for r in rows:
            mem_tag_counts[r["tag_id"]] = r["cnt"]

        cooccurrences = self._tags.get_all_co_occurrences(min_count=2)
        tag_blocks, bridge_lookup = self._build_bridge_lookups()
        threshold = self._config.tensor_retention_threshold
        bridge_bonus = self._config.tensor_bridge_bonus

        mem_pmi: dict[tuple[int, int], float] = {}
        mem_memory_ids: dict[tuple[int, int], list[str]] = {}
        if cooccurrences and total_memories >= 2:
            for tag_a, tag_b, count in cooccurrences:
                key = (min(tag_a, tag_b), max(tag_a, tag_b))
                if key not in resonant_set:
                    continue

                p_a = mem_tag_counts.get(tag_a, 0) / total_memories
                p_b = mem_tag_counts.get(tag_b, 0) / total_memories
                p_ab = count / total_memories
                if p_a <= 0 or p_b <= 0 or p_ab <= 0:
                    continue

                pmi = math.log(p_ab / (p_a * p_b))
                if pmi <= 0:
                    continue

                bridge = self._tag_pair_bridge(
                    key[0], key[1], tag_blocks, bridge_lookup,
                )
                if pmi + bridge_bonus * bridge < threshold:
                    continue

                mem_pmi[key] = pmi

        # ── Retrieval co-occurrence PMI (query-time) ───────────────
        total_ret_row = self._db.execute(
            "SELECT COUNT(DISTINCT retrieval_id) as cnt FROM retrieval_tag_hits"
        ).fetchone()
        total_retrievals = total_ret_row["cnt"] if total_ret_row else 0

        ret_pmi: dict[tuple[int, int], float] = {}
        if total_retrievals >= 2:
            ret_tag_counts: dict[int, int] = {}
            rows = self._db.execute(
                "SELECT tag_id, COUNT(DISTINCT retrieval_id) as cnt "
                "FROM retrieval_tag_hits GROUP BY tag_id"
            ).fetchall()
            for r in rows:
                ret_tag_counts[r["tag_id"]] = r["cnt"]

            rows = self._db.execute(
                """
                SELECT a.tag_id AS tag_a, b.tag_id AS tag_b,
                       COUNT(DISTINCT a.retrieval_id) AS cnt
                FROM retrieval_tag_hits a
                JOIN retrieval_tag_hits b
                  ON a.retrieval_id = b.retrieval_id
                 AND a.tag_id < b.tag_id
                GROUP BY a.tag_id, b.tag_id
                HAVING cnt >= 2
                """
            ).fetchall()
            for r in rows:
                key = (r["tag_a"], r["tag_b"])
                if key not in resonant_set:
                    continue
                p_a = ret_tag_counts.get(key[0], 0) / total_retrievals
                p_b = ret_tag_counts.get(key[1], 0) / total_retrievals
                p_ab = r["cnt"] / total_retrievals
                if p_a <= 0 or p_b <= 0 or p_ab <= 0:
                    continue
                pmi = math.log(p_ab / (p_a * p_b))
                if pmi > 0:
                    ret_pmi[key] = pmi

        # ── Combine and write ──────────────────────────────────────
        all_keys = set(mem_pmi) | set(ret_pmi)
        if not all_keys:
            return 0

        updated = 0
        for key in all_keys:
            combined = mem_pmi.get(key, 0.0) + ret_pmi.get(key, 0.0)

            if key in mem_pmi:
                memory_rows = self._db.execute(
                    """
                    SELECT a.memory_id FROM memory_tags a
                    JOIN memory_tags b ON a.memory_id = b.memory_id
                    WHERE a.tag_id = ? AND b.tag_id = ?
                    """,
                    (key[0], key[1]),
                ).fetchall()
                memory_ids = sorted({r["memory_id"] for r in memory_rows})
            else:
                memory_ids = []

            self._db.execute(
                """
                INSERT INTO tag_relational_tensor
                    (tag_a_id, tag_b_id, cooccurrence_strength, memory_ids)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tag_a_id, tag_b_id) DO UPDATE SET
                    cooccurrence_strength = excluded.cooccurrence_strength,
                    memory_ids = excluded.memory_ids,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                """,
                (key[0], key[1], combined, json.dumps(memory_ids)),
            )
            updated += 1

        logger.info(
            "Tensor cooccurrence: %d written (%d from memory, %d from retrieval)",
            updated, len(mem_pmi), len(ret_pmi),
        )
        self._db.connection.commit()
        return updated

    def _build_bridge_lookups(
        self,
    ) -> tuple[dict[int, set[int]], dict[tuple[int, int], float]]:
        """Build tag→block and block-pair→bridge_strength lookups."""
        tag_blocks: dict[int, set[int]] = {}
        rows = self._db.execute(
            "SELECT target_id, block_id FROM block_memberships WHERE target_type='tag'"
        ).fetchall()
        for r in rows:
            try:
                tid = int(r["target_id"])
            except (ValueError, TypeError):
                continue
            tag_blocks.setdefault(tid, set()).add(r["block_id"])

        bridge_lookup: dict[tuple[int, int], float] = {}
        rows = self._db.execute(
            "SELECT left_block_id, right_block_id, bridge_strength FROM bridge_edges"
        ).fetchall()
        for r in rows:
            bridge_lookup[(r["left_block_id"], r["right_block_id"])] = r["bridge_strength"]

        return tag_blocks, bridge_lookup

    @staticmethod
    def _tag_pair_bridge(
        tag_a: int,
        tag_b: int,
        tag_blocks: dict[int, set[int]],
        bridge_lookup: dict[tuple[int, int], float],
    ) -> float:
        """Max bridge strength between any block pair containing these tags."""
        blocks_a = tag_blocks.get(tag_a)
        blocks_b = tag_blocks.get(tag_b)
        if not blocks_a or not blocks_b:
            return 0.0
        best = 0.0
        for ba in blocks_a:
            for bb in blocks_b:
                if ba == bb:
                    continue
                key = (min(ba, bb), max(ba, bb))
                strength = bridge_lookup.get(key, 0.0)
                if strength > best:
                    best = strength
        return best

    def update_semantic_tensor(self) -> int:
        """Populate the semantic network from Poincaré distance between tag centroids.

        Projects tag centroids (mean embedding of associated memories) into the
        Poincaré disk via PCA + exponential map, then uses hyperbolic geodesic
        distance as the semantic relatedness signal: strength = exp(-distance).

        Returns the number of tensor entries created/updated.
        """
        all_cached = self._embeddings.get_all_cached()
        if not all_cached:
            return 0

        # Compute tag centroids in embedding space
        tag_centroids: dict[int, np.ndarray] = {}
        tag_memory_sets: dict[int, set[str]] = {}

        rows = self._db.execute(
            "SELECT tag_id, memory_id FROM memory_tags"
        ).fetchall()

        tag_memories: dict[int, list[str]] = defaultdict(list)
        for r in rows:
            tag_memories[r["tag_id"]].append(r["memory_id"])

        for tag_id, memory_ids in tag_memories.items():
            vecs = [all_cached[mid] for mid in memory_ids if mid in all_cached]
            if not vecs:
                continue
            centroid = np.mean(vecs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                tag_centroids[tag_id] = centroid / norm
                tag_memory_sets[tag_id] = {mid for mid in memory_ids if mid in all_cached}

        if not tag_centroids:
            return 0

        # Project tag centroids into the Poincaré disk
        self._ensure_pca_basis(all_cached)
        tag_poincare: dict[int, np.ndarray] = {}
        for tag_id, centroid in tag_centroids.items():
            if self._pca_basis is not None:
                projected = self._pca_basis @ centroid
            else:
                if len(centroid) < 2:
                    continue
                projected = centroid[:2]
            projected = np.asarray(projected, dtype=np.float32)
            tag_poincare[tag_id] = self._poincare.exp_map_origin(projected)

        resonant_set = self._load_resonant_tag_set()
        if not resonant_set:
            return 0

        co_retrieval_pairs = self._load_co_retrieval_pairs()

        updated = 0
        for key_a, key_b in resonant_set:
            if key_a not in tag_poincare or key_b not in tag_poincare:
                continue
            if (key_a, key_b) not in co_retrieval_pairs:
                continue

            hdist = self._poincare.distance(tag_poincare[key_a], tag_poincare[key_b])
            strength = math.exp(-hdist)
            if strength <= 1e-6:
                continue

            combined_memories = sorted(
                tag_memory_sets.get(key_a, set()) | tag_memory_sets.get(key_b, set())
            )

            self._db.execute(
                """
                INSERT INTO tag_relational_tensor
                    (tag_a_id, tag_b_id, semantic_strength, memory_ids)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tag_a_id, tag_b_id) DO UPDATE SET
                    semantic_strength = excluded.semantic_strength,
                    memory_ids = CASE
                        WHEN length(tag_relational_tensor.memory_ids) > length(excluded.memory_ids)
                        THEN tag_relational_tensor.memory_ids
                        ELSE excluded.memory_ids
                    END,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                """,
                (key_a, key_b, strength, json.dumps(combined_memories)),
            )
            updated += 1

        self._db.connection.commit()
        return updated

    def update_semiotic_tensor(self) -> int:
        """Populate the semiotic network from conditional probability asymmetry.

        Combines ingest-time memory co-occurrence with retrieval-time
        co-occurrence.  For resonant pair (A, B) with A < B:
        - base from memories:    P_mem(B|A), P_mem(A|B)
        - boost from retrievals: P_ret(B|A), P_ret(A|B)
        - final = base + boost

        Returns the number of tensor entries created/updated.
        """
        resonant_set = self._load_resonant_tag_set()
        if not resonant_set:
            return 0

        # Memory co-occurrence counts (ingest-time signal)
        mem_tag_counts: dict[int, int] = {}
        rows = self._db.execute(
            "SELECT tag_id, COUNT(DISTINCT memory_id) as cnt FROM memory_tags GROUP BY tag_id"
        ).fetchall()
        for r in rows:
            mem_tag_counts[r["tag_id"]] = r["cnt"]

        cooccurrences = self._tags.get_all_co_occurrences(min_count=1)
        mem_cooc_map: dict[tuple[int, int], int] = {
            (min(a, b), max(a, b)): c for a, b, c in cooccurrences
        } if cooccurrences else {}

        # Retrieval co-occurrence counts (query-time signal)
        ret_tag_counts: dict[int, int] = {}
        rows = self._db.execute(
            "SELECT tag_id, COUNT(DISTINCT retrieval_id) as cnt "
            "FROM retrieval_tag_hits GROUP BY tag_id"
        ).fetchall()
        for r in rows:
            ret_tag_counts[r["tag_id"]] = r["cnt"]

        ret_cooc_map: dict[tuple[int, int], int] = {}
        if ret_tag_counts:
            rows = self._db.execute(
                """
                SELECT a.tag_id AS tag_a, b.tag_id AS tag_b,
                       COUNT(DISTINCT a.retrieval_id) AS cnt
                FROM retrieval_tag_hits a
                JOIN retrieval_tag_hits b
                  ON a.retrieval_id = b.retrieval_id
                 AND a.tag_id < b.tag_id
                GROUP BY a.tag_id, b.tag_id
                HAVING cnt >= 2
                """
            ).fetchall()
            for r in rows:
                ret_cooc_map[(r["tag_a"], r["tag_b"])] = r["cnt"]

        if not mem_cooc_map and not ret_cooc_map:
            return 0

        updated = 0
        for tag_a, tag_b in resonant_set:
            mem_co = mem_cooc_map.get((tag_a, tag_b), 0)
            ret_co = ret_cooc_map.get((tag_a, tag_b), 0)
            if mem_co == 0 and ret_co == 0:
                continue

            forward = 0.0
            reverse = 0.0

            if mem_co > 0:
                ca = mem_tag_counts.get(tag_a, 0)
                cb = mem_tag_counts.get(tag_b, 0)
                if ca > 0:
                    forward += mem_co / ca
                if cb > 0:
                    reverse += mem_co / cb

            if ret_co > 0:
                ra = ret_tag_counts.get(tag_a, 0)
                rb = ret_tag_counts.get(tag_b, 0)
                if ra > 0:
                    forward += ret_co / ra
                if rb > 0:
                    reverse += ret_co / rb

            memory_rows = self._db.execute(
                """
                SELECT a.memory_id FROM memory_tags a
                JOIN memory_tags b ON a.memory_id = b.memory_id
                WHERE a.tag_id = ? AND b.tag_id = ?
                """,
                (tag_a, tag_b),
            ).fetchall()
            memory_ids = sorted({r["memory_id"] for r in memory_rows})

            self._db.execute(
                """
                INSERT INTO tag_relational_tensor
                    (tag_a_id, tag_b_id, semiotic_forward, semiotic_reverse, memory_ids)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(tag_a_id, tag_b_id) DO UPDATE SET
                    semiotic_forward = excluded.semiotic_forward,
                    semiotic_reverse = excluded.semiotic_reverse,
                    memory_ids = CASE
                        WHEN length(tag_relational_tensor.memory_ids) > length(excluded.memory_ids)
                        THEN tag_relational_tensor.memory_ids
                        ELSE excluded.memory_ids
                    END,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                """,
                (tag_a, tag_b, forward, reverse, json.dumps(memory_ids)),
            )
            updated += 1

        self._db.connection.commit()
        return updated

    def seed_tensor_from_associations(self) -> None:
        """Seed the tensor from the pre-existing association network.

        Populates R_cooccurrence, R_semantic, R_semiotic, and R_glyph.
        R_synchronicity is left at 0 (populated by lattice placement).
        """
        self.update_cooccurrence_tensor()
        self.update_semantic_tensor()
        self.update_semiotic_tensor()
        self.update_glyph_tensor()

    def rebuild_tensor(self) -> int:
        """Rebuild the entire tag relational tensor and resonances table from scratch.

        Clears all entries and repopulates all five networks:
        1. R_cooccurrence from memory_tags PMI
        2. R_semantic from embedding cosine similarities
        3. R_semiotic from conditional probability asymmetry
        4. R_synchronicity from lattice resonances (vectorized numpy, capped)
        5. R_glyph from glyph Ramsey lattice resonances

        Also clears and repopulates the resonances table.

        Returns total number of tensor entries.
        """
        self._db.execute("DELETE FROM tag_relational_tensor")
        self._db.execute("DELETE FROM resonances")
        self._db.connection.commit()

        # Seed co-occurrence, semantic, and semiotic networks
        self.update_cooccurrence_tensor()
        self.update_semantic_tensor()
        self.update_semiotic_tensor()

        # Populate synchronicity network — vectorized row-by-row
        resonant_set = self._load_resonant_tag_set()
        min_shared = self._config.lattice_min_resonance_primes
        primes = self._config.lattice_primes
        n_primes = len(primes)
        max_per_event = n_primes // 2
        min_angular_dist = self._compute_proximity_threshold(primes, min_shared)
        flush_every = 50_000  # Chunk writes to avoid WAL overflow

        rows = self._db.execute(
            """
            SELECT lp.sync_event_id, lp.angle, lp.prime_slots,
                   lp.poincare_x, lp.poincare_y, se.involved_tags
            FROM lattice_positions lp
            JOIN synchronicity_events se ON se.id = lp.sync_event_id
            """
        ).fetchall()

        n = len(rows)
        if n >= 2:
            # Build slot matrix: (n, n_primes) and parse tags + Poincaré coords
            all_ids: list[int] = []
            all_angles = np.zeros(n, dtype=np.float64)
            all_poincare_r = np.zeros((n, 2), dtype=np.float32)
            has_poincare_r = False
            all_tags: list[set[int]] = []
            slot_matrix = np.zeros((n, n_primes), dtype=np.int32)

            for i, r in enumerate(rows):
                all_ids.append(r["sync_event_id"])
                all_angles[i] = r["angle"]
                if r["poincare_x"] is not None:
                    all_poincare_r[i, 0] = r["poincare_x"]
                    all_poincare_r[i, 1] = r["poincare_y"]
                    has_poincare_r = True
                all_tags.append(set(json.loads(r["involved_tags"])))
                slots = json.loads(r["prime_slots"])
                for j, p in enumerate(primes):
                    slot_matrix[i, j] = slots.get(str(p), slots.get(p, -1))

            log_primes = np.array(
                [math.log(p) for p in primes], dtype=np.float64,
            )

            resonance_rows: list[tuple] = []
            tensor_agg: dict[tuple[int, int], float] = {}

            def _flush_resonances() -> None:
                """Write accumulated resonance rows to DB in chunks."""
                nonlocal resonance_rows
                if not resonance_rows:
                    return
                self._db.executemany(
                    """
                    INSERT OR IGNORE INTO resonances
                        (event_a_id, event_b_id, event_ids,
                         shared_primes, resonance_strength, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    resonance_rows,
                )
                self._db.connection.commit()
                resonance_rows = []

            # Row-by-row vectorized: compare position i against all j > i
            for i in range(n - 1):
                remaining = slot_matrix[i + 1:]  # (n-i-1, n_primes)

                # Broadcast compare against row i
                match_matrix = remaining == slot_matrix[i]  # bool
                shared_counts = match_matrix.sum(axis=1)

                candidate_offsets = np.where(shared_counts >= min_shared)[0]
                if len(candidate_offsets) == 0:
                    continue

                # Filter out angular-proximity artifacts
                remaining_angles = all_angles[i + 1:]
                angular_diffs = np.abs(
                    remaining_angles[candidate_offsets] - all_angles[i],
                )
                angular_diffs = np.minimum(
                    angular_diffs, 2.0 * math.pi - angular_diffs,
                )
                far_enough = angular_diffs >= min_angular_dist
                candidate_offsets = candidate_offsets[far_enough]
                if len(candidate_offsets) == 0:
                    continue

                # Strengths: sum of log(p) for shared primes
                strengths = (
                    match_matrix[candidate_offsets].astype(np.float64)
                    * log_primes
                ).sum(axis=1)

                # Cap: top-K strongest per event
                if len(candidate_offsets) > max_per_event:
                    top_k = np.argsort(strengths)[-max_per_event:]
                    candidate_offsets = candidate_offsets[top_k]
                    strengths = strengths[top_k]

                id_i = all_ids[i]
                tags_i = all_tags[i]

                for ci_pos, offset in enumerate(candidate_offsets):
                    j = i + 1 + int(offset)
                    id_j = all_ids[j]
                    strength = float(strengths[ci_pos])

                    shared = [
                        primes[k] for k in range(n_primes)
                        if match_matrix[offset, k]
                    ]

                    id_a, id_b = min(id_i, id_j), max(id_i, id_j)
                    chance = math.exp(-strength)
                    description = (
                        f"Events {id_a} and {id_b} resonate across "
                        f"{len(shared)} prime scales "
                        f"({', '.join(str(p) for p in shared)}). "
                        f"Surprise: {strength:.1f} nats. "
                        f"Chance probability: {chance:.2e}"
                    )

                    if has_poincare_r:
                        hdist = self._poincare.distance(
                            all_poincare_r[i], all_poincare_r[j],
                        )
                        description += f" Hyperbolic distance: {hdist:.3f}."

                    resonance_rows.append((
                        id_a, id_b, json.dumps([id_a, id_b]),
                        json.dumps(shared), strength, description,
                    ))

                    # Aggregate tensor: only for glyph-resonant tag pairs
                    tags_j = all_tags[j]
                    for ta in tags_i:
                        for tb in tags_j:
                            key = (min(ta, tb), max(ta, tb))
                            if key[0] == key[1] or key not in resonant_set:
                                continue
                            tensor_agg[key] = max(
                                tensor_agg.get(key, 0.0), strength,
                            )

                # Flush periodically to avoid WAL overflow
                if len(resonance_rows) >= flush_every:
                    _flush_resonances()

            # Final flush
            _flush_resonances()

            # Batch persist tensor updates
            if tensor_agg:
                tensor_rows = [
                    (key_a, key_b, strength, "[]")
                    for (key_a, key_b), strength in tensor_agg.items()
                ]
                for start in range(0, len(tensor_rows), flush_every):
                    chunk = tensor_rows[start:start + flush_every]
                    self._db.executemany(
                        """
                        INSERT INTO tag_relational_tensor
                            (tag_a_id, tag_b_id, synchronicity_strength,
                             memory_ids)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(tag_a_id, tag_b_id) DO UPDATE SET
                            synchronicity_strength = MAX(
                                tag_relational_tensor.synchronicity_strength,
                                excluded.synchronicity_strength
                            ),
                            updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                        """,
                        chunk,
                    )
                    self._db.connection.commit()

        # Populate glyph network from glyph lattice
        self.update_glyph_tensor()

        row = self._db.execute(
            "SELECT COUNT(*) as cnt FROM tag_relational_tensor"
        ).fetchone()
        return row["cnt"] if row else 0

    def get_resonant_memory_ids_fast(
        self,
        tag_ids: list[int],
    ) -> list[tuple[str, float]]:
        """Tensor-based lookup for resonant memories.

        Given query-context tag_ids, look up the tag relational tensor and
        combine the five network strengths plus the meta-resonance interaction
        term using configurable weights.

        The semiotic layer is direction-aware: when a query tag is tag_a_id,
        we use semiotic_forward (P(B|A)); when it's tag_b_id, we use
        semiotic_reverse (P(A|B)).

        The meta-resonance term (R_sync * R_glyph) fires only when a tag
        pair resonates on both the event lattice and the glyph lattice.

        Phase 1: compute a combined strength per *partner tag* (tags not in
        the query set that are connected via the tensor).  This avoids
        parsing the memory_ids JSON column entirely.

        Phase 2: resolve partner tags to memories via the indexed
        memory_tags table.
        """
        if not tag_ids:
            return []

        query_tag_set = set(tag_ids)
        placeholders = ",".join("?" * len(tag_ids))
        rows = self._db.execute(
            f"""
            SELECT tag_a_id, tag_b_id,
                   cooccurrence_strength, synchronicity_strength,
                   semantic_strength, semiotic_forward, semiotic_reverse,
                   glyph_strength, inhibition_strength, parallelness
            FROM tag_relational_tensor
            WHERE (tag_a_id IN ({placeholders}) OR tag_b_id IN ({placeholders}))
              AND (cooccurrence_strength > 0
                   OR synchronicity_strength > 0
                   OR semantic_strength > 0
                   OR semiotic_forward > 0
                   OR semiotic_reverse > 0
                   OR glyph_strength > 0
                   OR inhibition_strength > 0)
            """,
            (*tag_ids, *tag_ids),
        ).fetchall()

        if not rows:
            return []

        w_co = self._config.tensor_cooccurrence_weight
        w_sync = self._config.tensor_synchronicity_weight
        w_sem = self._config.tensor_semantic_weight
        w_semio = self._config.tensor_semiotic_weight
        w_glyph = self._config.tensor_glyph_weight
        w_meta = self._config.tensor_meta_resonance_weight
        w_inhib = self._config.tensor_inhibition_weight

        # Phase 1: aggregate combined strength per partner tag
        partner_scores: dict[int, float] = {}
        for row in rows:
            a_in_query = row["tag_a_id"] in query_tag_set
            b_in_query = row["tag_b_id"] in query_tag_set

            # Direction-aware semiotic contribution
            if a_in_query and b_in_query:
                # Both tags already in query — no new partner to surface
                continue
            elif a_in_query:
                semiotic = row["semiotic_forward"]
                partner = row["tag_b_id"]
            else:
                semiotic = row["semiotic_reverse"]
                partner = row["tag_a_id"]

            sync_val = row["synchronicity_strength"]
            glyph_val = row["glyph_strength"]

            excitatory = (
                w_co * row["cooccurrence_strength"]
                + w_sync * sync_val
                + w_sem * row["semantic_strength"]
                + w_semio * semiotic
                + w_glyph * glyph_val
            )

            # Meta-resonance: cross-lattice interaction term
            if sync_val > 0 and glyph_val > 0:
                excitatory += w_meta * sync_val * glyph_val

            # Lateral inhibition: suppresses antiparallel partners
            inhib = row["inhibition_strength"]
            par = row["parallelness"]
            suppression = w_inhib * inhib * max(0.0, -par) if inhib > 0 and par < 0 else 0.0

            combined = excitatory - suppression

            if combined > 0:
                partner_scores[partner] = max(
                    partner_scores.get(partner, 0.0), combined,
                )

        if not partner_scores:
            return []

        max_combined = max(partner_scores.values())
        if max_combined <= 0:
            return []

        # Phase 2: resolve partner tags → memories via memory_tags index
        partner_ids = list(partner_scores.keys())
        p2 = ",".join("?" * len(partner_ids))
        mem_rows = self._db.execute(
            f"SELECT memory_id, tag_id FROM memory_tags WHERE tag_id IN ({p2})",
            tuple(partner_ids),
        ).fetchall()

        memory_scores: dict[str, float] = {}
        for r in mem_rows:
            normalized = partner_scores[r["tag_id"]] / max_combined
            mid = r["memory_id"]
            memory_scores[mid] = max(memory_scores.get(mid, 0.0), normalized)

        return sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)

    # ── Glyph Ramsey Lattice ─────────────────────────────────────────

    def place_tags_on_glyph_lattice(
        self, tag_names: list[str],
    ) -> list[GlyphPosition]:
        """Place word tags on the glyph Ramsey lattice.

        For each word tag (len > 1) not already placed, computes an
        embedding vector (ByT5 encoder when available, else letter
        frequency), projects to an angle, assigns prime slots, and
        detects glyph resonances with existing placements.

        When the ByT5 encoder is available, all tag names are batch-
        embedded in a single forward pass for efficiency.
        """
        # Phase 0: filter to tags that need placement
        to_place: list[tuple[str, object]] = []  # (name, Tag)
        for name in tag_names:
            if not name:
                continue
            tag = self._tags.get_by_name(name)
            if tag is None:
                continue
            existing = self._db.execute(
                "SELECT tag_id FROM glyph_positions WHERE tag_id = ?",
                (tag.id,),
            ).fetchone()
            if existing:
                continue
            to_place.append((name, tag))

        if not to_place:
            return []

        # Phase 1: batch embed via ByT5 if available
        byt5_vecs: dict[str, np.ndarray] = {}
        use_byt5 = (
            self._glyph_encoder is not None
            and self._glyph_encoder.is_available
        )
        if use_byt5:
            names_to_embed = [name for name, _ in to_place]
            try:
                vecs = self._glyph_encoder.embed_batch(names_to_embed)
                for i, name in enumerate(names_to_embed):
                    byt5_vecs[name] = vecs[i]
            except Exception:
                byt5_vecs.clear()  # Fall through to letter-count

        # Phase 2: compute angles and store
        #
        # Angles use SHA256 of the embedding vector for uniform
        # distribution on the circle — PCA angles cluster semantically
        # similar tags, which trivially inflates prime-slot sharing.
        # The full ByT5 embedding is stored for cosine filtering.
        import hashlib

        new_positions: list[GlyphPosition] = []

        for name, tag in to_place:
            if name in byt5_vecs:
                vec = byt5_vecs[name]
                glyph_blob = vec.astype(np.float32).tobytes()
                h = hashlib.sha256(glyph_blob).digest()
                val = int.from_bytes(h[:8], "big")
                angle = (val / (2**64)) * 2 * math.pi
                glyph_dim = self._config.glyph_embedding_dimension
            else:
                result = self._compute_glyph_angle_letter_count(name)
                if result is None:
                    continue
                angle, glyph_blob, glyph_dim = result

            prime_slots = self._compute_prime_slots(
                angle, primes=self._config.glyph_lattice_primes,
            )

            self._db.execute(
                """
                INSERT INTO glyph_positions
                    (tag_id, angle, prime_slots, glyph_vector, glyph_dimension)
                VALUES (?, ?, ?, ?, ?)
                """,
                (tag.id, angle, json.dumps(prime_slots), glyph_blob, glyph_dim),
            )

            pos = GlyphPosition(
                tag_id=tag.id,
                tag_name=name,
                angle=angle,
                prime_slots=json.dumps(prime_slots),
                glyph_vector=glyph_blob,
                glyph_dimension=glyph_dim,
            )
            new_positions.append(pos)

        if new_positions:
            self._db.connection.commit()
            self._find_and_persist_glyph_resonances(new_positions)
            self._apply_symbolic_glyph_bonus(
                [p.tag_name for p in new_positions],
            )

        return new_positions

    def _compute_glyph_angle_letter_count(
        self, tag_name: str,
    ) -> Optional[tuple[float, bytes, int]]:
        """Fallback glyph angle via letter-count SHA256 hash.

        Returns (angle, glyph_vector_blob, dimension) or None.
        """
        import hashlib

        letter_counts = self._tags.decompose_to_letters([tag_name])
        if not letter_counts:
            return None

        vec = np.zeros(26, dtype=np.float32)
        for ch, count in letter_counts.items():
            idx = ord(ch) - ord("a")
            if 0 <= idx < 26:
                vec[idx] = count

        norm = np.linalg.norm(vec)
        if norm == 0:
            h = hashlib.sha256(tag_name.encode("utf-8")).digest()
            val = int.from_bytes(h[:8], "big")
            angle = (val / (2**64)) * 2 * math.pi
            blob = vec.astype(np.float32).tobytes()
            return angle, blob, 26

        vec_normalized = vec / norm
        h = hashlib.sha256(vec_normalized.tobytes()).digest()
        val = int.from_bytes(h[:8], "big")
        angle = (val / (2**64)) * 2 * math.pi

        blob = vec_normalized.astype(np.float32).tobytes()
        return angle, blob, 26

    def _find_and_persist_glyph_resonances(
        self, new_positions: list[GlyphPosition],
    ) -> None:
        """Vectorized glyph resonance detection for newly placed tags.

        Same algorithm as _find_and_persist_resonances but operates on
        tag pairs via glyph_positions instead of event pairs.

        Filters out trivial resonances where tag pairs have high glyph
        vector similarity (cosine > 0.95) — those share prime slots
        simply because they have similar compositions, not because of
        structural entanglement.
        """
        min_shared = self._config.glyph_min_resonance_primes
        primes = self._config.glyph_lattice_primes
        n_primes = len(primes)
        max_per_tag = n_primes // 2
        min_angular_dist = self._compute_proximity_threshold(primes, min_shared)

        rows = self._db.execute(
            "SELECT tag_id, angle, prime_slots, glyph_vector, glyph_dimension"
            " FROM glyph_positions"
        ).fetchall()

        if len(rows) < 2:
            return

        # Determine vector dimension — use max dimension present.
        # Mixed dimensions (e.g. after model change) trigger a rebuild.
        dimensions = {r["glyph_dimension"] for r in rows}
        if len(dimensions) > 1:
            self.rebuild_glyph_lattice()
            return
        dim = max(dimensions)

        # ByT5 embeddings cluster tightly (cos > 0.93 for most pairs),
        # so the angular proximity filter handles false positives.
        # Only filter near-duplicates for high-dim; use 0.95 for letter-count.
        trivial_threshold = 0.998 if dim > 26 else 0.95

        all_ids: list[int] = []
        all_angles = np.zeros(len(rows), dtype=np.float64)
        slot_matrix = np.zeros((len(rows), n_primes), dtype=np.int32)
        glyph_vecs = np.zeros((len(rows), dim), dtype=np.float32)

        for i, r in enumerate(rows):
            all_ids.append(r["tag_id"])
            all_angles[i] = r["angle"]
            slots = json.loads(r["prime_slots"])
            for j, p in enumerate(primes):
                slot_matrix[i, j] = slots.get(str(p), slots.get(p, -1))
            vec = np.frombuffer(r["glyph_vector"], dtype=np.float32).copy()
            glyph_vecs[i, :len(vec)] = vec

        # Normalize glyph vectors for cosine similarity
        norms = np.linalg.norm(glyph_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        glyph_unit = glyph_vecs / norms

        id_to_idx = {tid: i for i, tid in enumerate(all_ids)}
        log_primes = np.array(
            [math.log(p) for p in primes], dtype=np.float64,
        )

        new_tag_ids = {pos.tag_id for pos in new_positions}
        resonance_rows: list[tuple] = []
        tensor_agg: dict[tuple[int, int], float] = {}

        for pos in new_positions:
            idx = id_to_idx.get(pos.tag_id)
            if idx is None:
                continue

            new_vec = slot_matrix[idx]  # (n_primes,)

            # Broadcast compare against all positions
            match_matrix = slot_matrix == new_vec  # (n_all, n_primes) bool
            shared_counts = match_matrix.sum(axis=1)
            shared_counts[idx] = 0  # exclude self

            candidate_indices = np.where(shared_counts >= min_shared)[0]
            if len(candidate_indices) == 0:
                continue

            # Filter out angular-proximity artifacts: pairs closer than one
            # slot-width of p_max share all slots trivially
            angular_diffs = np.abs(all_angles[candidate_indices] - all_angles[idx])
            # Handle wraparound (circle topology)
            angular_diffs = np.minimum(angular_diffs, 2.0 * math.pi - angular_diffs)
            far_enough = angular_diffs >= min_angular_dist
            candidate_indices = candidate_indices[far_enough]
            if len(candidate_indices) == 0:
                continue

            # Filter out trivially similar glyph vectors
            cosine_sims = glyph_unit[candidate_indices] @ glyph_unit[idx]
            surprising = cosine_sims < trivial_threshold
            candidate_indices = candidate_indices[surprising]
            if len(candidate_indices) == 0:
                continue

            strengths = (
                match_matrix[candidate_indices].astype(np.float64)
                * log_primes
            ).sum(axis=1)

            # Cap per tag
            if len(candidate_indices) > max_per_tag:
                top_k = np.argsort(strengths)[-max_per_tag:]
                candidate_indices = candidate_indices[top_k]
                strengths = strengths[top_k]

            for ci_pos, ci in enumerate(candidate_indices):
                other_id = all_ids[ci]

                # Avoid double-counting for new-to-new pairs
                if other_id in new_tag_ids and pos.tag_id > other_id:
                    continue

                strength = float(strengths[ci_pos])
                shared = [
                    primes[k] for k in range(n_primes)
                    if match_matrix[ci, k]
                ]

                tag_a, tag_b = min(pos.tag_id, other_id), max(pos.tag_id, other_id)

                resonance_rows.append((
                    tag_a, tag_b, json.dumps(shared), strength,
                ))

                tensor_agg[(tag_a, tag_b)] = max(
                    tensor_agg.get((tag_a, tag_b), 0.0), strength,
                )

        # Batch persist glyph resonances
        if resonance_rows:
            self._db.executemany(
                """
                INSERT INTO glyph_resonances
                    (tag_a_id, tag_b_id, shared_primes, resonance_strength)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tag_a_id, tag_b_id) DO UPDATE SET
                    shared_primes = excluded.shared_primes,
                    resonance_strength = excluded.resonance_strength,
                    detected_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                """,
                resonance_rows,
            )

        # Batch persist tensor updates
        if tensor_agg:
            tensor_rows = [
                (key_a, key_b, strength, "[]")
                for (key_a, key_b), strength in tensor_agg.items()
            ]
            self._db.executemany(
                """
                INSERT INTO tag_relational_tensor
                    (tag_a_id, tag_b_id, glyph_strength, memory_ids)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(tag_a_id, tag_b_id) DO UPDATE SET
                    glyph_strength = MAX(
                        tag_relational_tensor.glyph_strength,
                        excluded.glyph_strength
                    ),
                    updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                """,
                tensor_rows,
            )

        if resonance_rows or tensor_agg:
            self._db.connection.commit()

    def _apply_symbolic_glyph_bonus(
        self, new_tag_names: list[str],
    ) -> None:
        """Apply bonus glyph_strength for symbolic equivalences and known pairs.

        When the glyph encoder is a GlyphBridge (GPT-GU integration):
        - Tags mapping to the SAME glyph get a multiplicative bonus
        - Tags whose glyphs form a known PAIR_TO_TEXT entry get a bonus
        """
        from chicory.layer1.glyph_bridge import GlyphBridge

        if not isinstance(self._glyph_encoder, GlyphBridge):
            return

        bridge: GlyphBridge = self._glyph_encoder

        # Build tag_name -> (tag_id, glyph) map for all placed tags
        all_mappings = bridge.get_all_glyph_mappings()
        if not all_mappings:
            return

        # Resolve tag_name -> tag_id for placed tags
        name_to_id: dict[str, int] = {}
        rows = self._db.execute(
            "SELECT gp.tag_id, t.name FROM glyph_positions gp"
            " JOIN tags t ON t.id = gp.tag_id"
        ).fetchall()
        for r in rows:
            name_to_id[r["name"]] = r["tag_id"]

        # Build glyph -> list[tag_id] index
        glyph_to_tag_ids: dict[str, list[int]] = {}
        for tag_name, glyph_symbol in all_mappings.items():
            tid = name_to_id.get(tag_name)
            if tid is not None:
                glyph_to_tag_ids.setdefault(glyph_symbol, []).append(tid)

        new_tag_ids = set()
        for name in new_tag_names:
            tid = name_to_id.get(name)
            if tid is not None:
                new_tag_ids.add(tid)

        bonus_rows: list[tuple[int, int, float]] = []
        symbolic_bonus = self._config.glyph_symbolic_bonus
        pair_bonus = self._config.glyph_pair_bonus

        # Same-glyph pairs: tags sharing the same glyph symbol
        for glyph_symbol, tag_ids in glyph_to_tag_ids.items():
            if len(tag_ids) < 2:
                continue
            for i in range(len(tag_ids)):
                for j in range(i + 1, len(tag_ids)):
                    a, b = min(tag_ids[i], tag_ids[j]), max(tag_ids[i], tag_ids[j])
                    # Only process pairs involving new tags
                    if a not in new_tag_ids and b not in new_tag_ids:
                        continue
                    bonus_rows.append((a, b, symbolic_bonus))

        # Known-pair matches: tags whose glyphs form a PAIR_TO_TEXT entry
        pair_rels = bridge.get_pair_relationships()
        if pair_rels:
            # Build tag_id -> glyph reverse lookup
            tid_to_glyph: dict[int, str] = {}
            for tag_name, glyph_symbol in all_mappings.items():
                tid = name_to_id.get(tag_name)
                if tid is not None:
                    tid_to_glyph[tid] = glyph_symbol

            placed_tids = list(tid_to_glyph.keys())
            for i in range(len(placed_tids)):
                if placed_tids[i] not in new_tag_ids:
                    continue
                for j in range(len(placed_tids)):
                    if i == j:
                        continue
                    ga = tid_to_glyph[placed_tids[i]]
                    gb = tid_to_glyph[placed_tids[j]]
                    pair_key = frozenset((ga, gb))
                    if pair_key in pair_rels:
                        a = min(placed_tids[i], placed_tids[j])
                        b = max(placed_tids[i], placed_tids[j])
                        bonus_rows.append((a, b, pair_bonus))

        if not bonus_rows:
            return

        # Deduplicate: keep max bonus per pair
        pair_max: dict[tuple[int, int], float] = {}
        for a, b, bonus in bonus_rows:
            pair_max[(a, b)] = max(pair_max.get((a, b), 0.0), bonus)

        upsert_rows = [
            (a, b, bonus, "[]")
            for (a, b), bonus in pair_max.items()
        ]

        self._db.executemany(
            """
            INSERT INTO tag_relational_tensor
                (tag_a_id, tag_b_id, glyph_strength, memory_ids)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(tag_a_id, tag_b_id) DO UPDATE SET
                glyph_strength = MAX(
                    tag_relational_tensor.glyph_strength,
                    excluded.glyph_strength
                ),
                updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
            """,
            upsert_rows,
        )
        self._db.connection.commit()

        logger.info(
            "Symbolic glyph bonus applied: %d pairs (%d same-glyph, %d pair-match)",
            len(pair_max),
            sum(1 for (_, _), b in pair_max.items() if b == symbolic_bonus),
            sum(1 for (_, _), b in pair_max.items() if b == pair_bonus),
        )

    def update_glyph_tensor(self) -> int:
        """Rebuild glyph_strength in the tensor from the glyph_resonances table."""
        # Reset all glyph_strength to 0
        self._db.execute(
            "UPDATE tag_relational_tensor SET glyph_strength = 0.0"
        )

        rows = self._db.execute(
            "SELECT tag_a_id, tag_b_id, resonance_strength FROM glyph_resonances"
        ).fetchall()

        if not rows:
            self._db.connection.commit()
            return 0

        tensor_rows = [
            (r["tag_a_id"], r["tag_b_id"], r["resonance_strength"], "[]")
            for r in rows
        ]
        self._db.executemany(
            """
            INSERT INTO tag_relational_tensor
                (tag_a_id, tag_b_id, glyph_strength, memory_ids)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(tag_a_id, tag_b_id) DO UPDATE SET
                glyph_strength = MAX(
                    tag_relational_tensor.glyph_strength,
                    excluded.glyph_strength
                ),
                updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
            """,
            tensor_rows,
        )

        self._db.connection.commit()
        return len(rows)

    def rebuild_glyph_lattice(self) -> int:
        """Clear and rebuild the entire glyph lattice from active word tags."""
        self._db.execute("DELETE FROM glyph_positions")
        self._db.execute("DELETE FROM glyph_resonances")
        self._db.connection.commit()

        # Invalidate PCA cache so it's recomputed from fresh embeddings
        if self._glyph_encoder is not None:
            self._glyph_encoder.invalidate_pca_cache()

        tag_names = self._tags.list_active_names()
        word_tags = [n for n in tag_names if len(n) > 1]
        if not word_tags:
            return 0

        positions = self.place_tags_on_glyph_lattice(word_tags)
        return len(positions)

    # ── Lateral Inhibition & Cross-Reference ─────────────────────────

    def compute_all_parallelness(self) -> int:
        """Compute parallelness for all tensor edges from glyph lattice angles.

        parallelness = cos(angle_a - angle_b): +1 parallel, -1 antiparallel.
        Vectorized with numpy; writes via temp table + single joined UPDATE.
        """
        import numpy as np

        pos_rows = self._db.execute(
            "SELECT tag_id, angle FROM glyph_positions"
        ).fetchall()
        if not pos_rows:
            return 0

        tag_angles: dict[int, float] = {r["tag_id"]: r["angle"] for r in pos_rows}

        edges = self._db.execute(
            "SELECT tag_a_id, tag_b_id FROM tag_relational_tensor"
        ).fetchall()

        a_ids = []
        b_ids = []
        deltas = []
        for row in edges:
            a, b = row["tag_a_id"], row["tag_b_id"]
            if a in tag_angles and b in tag_angles:
                a_ids.append(a)
                b_ids.append(b)
                deltas.append(tag_angles[a] - tag_angles[b])

        if not deltas:
            return 0

        par_values = np.cos(np.array(deltas, dtype=np.float64))

        self._db.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _par_batch "
            "(tag_a_id INTEGER, tag_b_id INTEGER, par REAL)"
        )
        self._db.execute("DELETE FROM _par_batch")
        self._db.executemany(
            "INSERT INTO _par_batch VALUES (?, ?, ?)",
            [(int(a_ids[i]), int(b_ids[i]), float(par_values[i]))
             for i in range(len(a_ids))],
        )
        self._db.execute(
            """
            UPDATE tag_relational_tensor
            SET parallelness = (
                    SELECT par FROM _par_batch p
                    WHERE p.tag_a_id = tag_relational_tensor.tag_a_id
                      AND p.tag_b_id = tag_relational_tensor.tag_b_id
                ),
                updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
            WHERE EXISTS (
                SELECT 1 FROM _par_batch p
                WHERE p.tag_a_id = tag_relational_tensor.tag_a_id
                  AND p.tag_b_id = tag_relational_tensor.tag_b_id
            )
            """
        )
        self._db.execute("DROP TABLE IF EXISTS _par_batch")
        self._db.connection.commit()
        return len(a_ids)

    def seed_from_cross_reference(self) -> int:
        """Import GPT-GU cross-reference data into the tag relational tensor.

        Loads oppositions (→ inhibition_strength), transformations (→ semiotic),
        and co-occurrences (→ cooccurrence boost) from the GlyphBridge.
        Maps glyph symbols back to Chicory tag IDs via glyph_bridge_cache.

        Returns the number of tensor edges updated.
        """
        from chicory.layer1.glyph_bridge import GlyphBridge

        if not isinstance(self._glyph_encoder, GlyphBridge):
            return 0

        bridge: GlyphBridge = self._glyph_encoder
        xref_data = bridge.get_cross_reference_data()
        if not xref_data:
            return 0

        # Build glyph_symbol -> [tag_id] mapping from bridge cache + tags
        glyph_to_tids = self._build_glyph_to_tag_ids(bridge)
        if not glyph_to_tids:
            return 0

        resonant_set = self._load_resonant_tag_set()
        updates: dict[tuple[int, int], dict[str, float]] = {}

        def _resonant_pair(a: int, b: int) -> tuple[int, int] | None:
            if a == b:
                return None
            key = (min(a, b), max(a, b))
            return key if key in resonant_set else None

        # 1. Oppositions → inhibition_strength + semantic_strength
        for glyph_a, opp_set in xref_data.get("oppositions", {}).items():
            tids_a = glyph_to_tids.get(glyph_a, [])
            for glyph_b in opp_set:
                tids_b = glyph_to_tids.get(glyph_b, [])
                for ta in tids_a:
                    for tb in tids_b:
                        pair = _resonant_pair(ta, tb)
                        if pair is None:
                            continue
                        d = updates.setdefault(pair, {})
                        d["inhibition_strength"] = max(
                            d.get("inhibition_strength", 0.0), 0.5)
                        d["semantic_strength"] = max(
                            d.get("semantic_strength", 0.0), 0.6)

        # 2. Transformations → semiotic_forward / semiotic_reverse
        for glyph_a, target_set in xref_data.get("transformations", {}).items():
            tids_a = glyph_to_tids.get(glyph_a, [])
            for glyph_b in target_set:
                tids_b = glyph_to_tids.get(glyph_b, [])
                for ta in tids_a:
                    for tb in tids_b:
                        pair = _resonant_pair(ta, tb)
                        if pair is None:
                            continue
                        d = updates.setdefault(pair, {})
                        if ta == pair[0]:
                            d["semiotic_forward"] = max(
                                d.get("semiotic_forward", 0.0), 0.7)
                        else:
                            d["semiotic_reverse"] = max(
                                d.get("semiotic_reverse", 0.0), 0.7)

        # 3. Co-occurrences → cooccurrence boost
        seen_pairs: set[tuple[str, str]] = set()
        for glyph_a, counter in xref_data.get("co_occurrences", {}).items():
            tids_a = glyph_to_tids.get(glyph_a, [])
            for glyph_b, count in counter.items():
                gpair = tuple(sorted([glyph_a, glyph_b]))
                if gpair in seen_pairs:
                    continue
                seen_pairs.add(gpair)
                tids_b = glyph_to_tids.get(glyph_b, [])
                cooc_val = 1.0 / (1.0 + math.exp(-(math.log(1 + count) - 2.0)))
                for ta in tids_a:
                    for tb in tids_b:
                        pair = _resonant_pair(ta, tb)
                        if pair is None:
                            continue
                        d = updates.setdefault(pair, {})
                        d["cooccurrence_strength"] = max(
                            d.get("cooccurrence_strength", 0.0), cooc_val)

        if not updates:
            return 0

        # Batch upsert into tensor
        for (a, b), signals in updates.items():
            cols = list(signals.keys())
            vals = [signals[k] for k in cols]
            set_clause = ", ".join(
                f"{c} = MAX(tag_relational_tensor.{c}, ?)" for c in cols
            )
            val_placeholders = ", ".join("?" for _ in cols)
            col_names = ", ".join(cols)

            self._db.execute(
                f"""
                INSERT INTO tag_relational_tensor
                    (tag_a_id, tag_b_id, {col_names}, memory_ids)
                VALUES (?, ?, {val_placeholders}, '[]')
                ON CONFLICT(tag_a_id, tag_b_id) DO UPDATE SET
                    {set_clause},
                    updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                """,
                (a, b, *vals, *vals),
            )

        self._db.connection.commit()

        logger.info(
            "Cross-reference seeded: %d tensor edges "
            "(%d oppositions, %d transformations, %d co-occurrences)",
            len(updates),
            sum(1 for s in updates.values() if "inhibition_strength" in s),
            sum(1 for s in updates.values()
                if "semiotic_forward" in s or "semiotic_reverse" in s),
            sum(1 for s in updates.values() if "cooccurrence_strength" in s),
        )

        return len(updates)

    def seed_semiotic_from_pairs(self) -> int:
        """Seed semiotic_forward/reverse from GPT-GU's PAIR_TO_TEXT directionality.

        Each pair has an inherent ordering (concept_a vs concept_b). Tags
        whose glyphs match a pair get directed semiotic signals.
        """
        from chicory.layer1.glyph_bridge import GlyphBridge

        if not isinstance(self._glyph_encoder, GlyphBridge):
            return 0

        bridge: GlyphBridge = self._glyph_encoder
        directed = bridge.get_directed_pairs()
        if not directed:
            return 0

        glyph_to_tids = self._build_glyph_to_tag_ids(bridge)
        if not glyph_to_tids:
            return 0

        resonant_set = self._load_resonant_tag_set()
        rows: list[tuple] = []
        for ga, gb, _ca, _cb in directed:
            tids_a = glyph_to_tids.get(ga, [])
            tids_b = glyph_to_tids.get(gb, [])
            for ta in tids_a:
                for tb in tids_b:
                    if ta == tb:
                        continue
                    a, b = min(ta, tb), max(ta, tb)
                    if (a, b) not in resonant_set:
                        continue
                    if ta == a:
                        rows.append((a, b, 0.7, 0.0, "[]"))
                    else:
                        rows.append((a, b, 0.0, 0.7, "[]"))

        if not rows:
            return 0

        self._db.executemany(
            """
            INSERT INTO tag_relational_tensor
                (tag_a_id, tag_b_id, semiotic_forward, semiotic_reverse, memory_ids)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(tag_a_id, tag_b_id) DO UPDATE SET
                semiotic_forward = MAX(
                    tag_relational_tensor.semiotic_forward,
                    excluded.semiotic_forward
                ),
                semiotic_reverse = MAX(
                    tag_relational_tensor.semiotic_reverse,
                    excluded.semiotic_reverse
                ),
                updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
            """,
            rows,
        )
        self._db.connection.commit()

        logger.info("Semiotic pairs seeded: %d directed edges", len(rows))
        return len(rows)

    def _build_glyph_to_tag_ids(self, bridge) -> dict[str, list[int]]:
        """Build glyph_symbol → [tag_id] mapping from bridge cache + tags table."""
        cache_rows = self._db.execute(
            "SELECT tag_name, glyph_symbol FROM glyph_bridge_cache"
        ).fetchall()
        if not cache_rows:
            return {}

        # tag_name -> glyph_symbol
        name_to_glyph: dict[str, str] = {
            r["tag_name"]: r["glyph_symbol"] for r in cache_rows
        }

        # tag_name -> tag_id for active tags
        name_to_id: dict[str, int] = {}
        tag_rows = self._db.execute(
            "SELECT id, name FROM tags WHERE is_active = 1"
        ).fetchall()
        for r in tag_rows:
            name_to_id[r["name"]] = r["id"]

        # glyph -> [tag_id]
        result: dict[str, list[int]] = {}
        for name, glyph in name_to_glyph.items():
            tid = name_to_id.get(name)
            if tid is not None:
                result.setdefault(glyph, []).append(tid)

        return result

    def get_glyph_association_scores(
        self, context_tag_ids: list[int],
    ) -> list[tuple[str, float]]:
        """Score memories by glyph network association (tensor edge traversal).

        Like GPT-GU's recall_by_association: given context tags, follow
        glyph tensor edges to find memories connected via glyph resonance,
        with lateral inhibition pruning antiparallel activations.

        Returns sorted (memory_id, score) pairs, normalized to [0, 1].
        """
        if not context_tag_ids:
            return []

        query_set = set(context_tag_ids)
        placeholders = ",".join("?" * len(context_tag_ids))
        w_inhib = self._config.tensor_inhibition_weight

        # Get all tensor edges touching context tags that have glyph signal
        rows = self._db.execute(
            f"""
            SELECT tag_a_id, tag_b_id,
                   glyph_strength, cooccurrence_strength,
                   inhibition_strength, parallelness
            FROM tag_relational_tensor
            WHERE (tag_a_id IN ({placeholders}) OR tag_b_id IN ({placeholders}))
              AND glyph_strength > 0
            """,
            (*context_tag_ids, *context_tag_ids),
        ).fetchall()

        if not rows:
            return []

        # Phase 1: score partner tags by glyph composite with inhibition
        partner_scores: dict[int, float] = {}
        for row in rows:
            a_in = row["tag_a_id"] in query_set
            b_in = row["tag_b_id"] in query_set
            if a_in and b_in:
                continue
            partner = row["tag_b_id"] if a_in else row["tag_a_id"]

            # Excitatory: glyph resonance + co-occurrence bonus
            score = row["glyph_strength"] + 0.3 * row["cooccurrence_strength"]

            # Inhibition: antiparallel suppression
            inhib = row["inhibition_strength"]
            par = row["parallelness"]
            if inhib > 0 and par < 0:
                score -= w_inhib * inhib * (-par)

            if score > 0:
                partner_scores[partner] = max(
                    partner_scores.get(partner, 0.0), score)

        if not partner_scores:
            return []

        # Phase 2: resolve partner tags → memories
        partner_ids = list(partner_scores.keys())
        p2 = ",".join("?" * len(partner_ids))
        mem_rows = self._db.execute(
            f"SELECT memory_id, tag_id FROM memory_tags WHERE tag_id IN ({p2})",
            tuple(partner_ids),
        ).fetchall()

        memory_scores: dict[str, float] = {}
        for r in mem_rows:
            mid = r["memory_id"]
            memory_scores[mid] = max(
                memory_scores.get(mid, 0.0),
                partner_scores[r["tag_id"]],
            )

        if not memory_scores:
            return []

        # Normalize to [0, 1]
        max_score = max(memory_scores.values())
        if max_score > 0:
            memory_scores = {
                mid: s / max_score for mid, s in memory_scores.items()
            }

        return sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)

    def get_inhibition_score(
        self, tag_ids_a: list[int], tag_ids_b: list[int],
    ) -> float:
        """Compute net inhibition between two sets of tags.

        Only considers glyph-resonant pairs; fetches all matching rows in
        a single batch query instead of one SELECT per pair.
        """
        if not tag_ids_a or not tag_ids_b:
            return 0.0

        resonant_set = self._load_resonant_tag_set()
        pairs: list[tuple[int, int]] = []
        for ta in tag_ids_a:
            for tb in tag_ids_b:
                if ta == tb:
                    continue
                key = (min(ta, tb), max(ta, tb))
                if key in resonant_set:
                    pairs.append(key)

        if not pairs:
            return 0.0

        total_suppression = 0.0
        for chunk_start in range(0, len(pairs), 450):
            chunk = pairs[chunk_start:chunk_start + 450]
            conditions = " OR ".join(
                "(tag_a_id = ? AND tag_b_id = ?)" for _ in chunk
            )
            params = [v for a, b in chunk for v in (a, b)]
            rows = self._db.execute(
                f"SELECT inhibition_strength, parallelness "
                f"FROM tag_relational_tensor "
                f"WHERE ({conditions}) AND inhibition_strength > 0",
                params,
            ).fetchall()
            for row in rows:
                par = row["parallelness"]
                total_suppression += row["inhibition_strength"] * max(0.0, -par)

        return total_suppression

    # ── Internal ─────────────────────────────────────────────────────

    def _compute_angle(self, event: SynchronicityEvent) -> Optional[float]:
        """Project an event's involved-tag embedding centroids to an angle."""
        result = self._compute_projection(event)
        if result is None:
            return None
        return result[0]

    def _compute_projection(
        self, event: SynchronicityEvent,
    ) -> Optional[tuple[float, np.ndarray]]:
        """Project an event to (angle, 2D point) via PCA.

        Returns (angle, projected_2d) or None if embeddings unavailable.
        The 2D point preserves magnitude — used by Poincaré exponential map.
        """
        tag_ids = json.loads(event.involved_tags)
        if not tag_ids:
            return None

        all_cached = self._embeddings.get_all_cached()
        if not all_cached:
            return None

        tag_centroids: list[np.ndarray] = []
        for tag_id in tag_ids:
            rows = self._db.query(
                "SELECT memory_id FROM memory_tags WHERE tag_id = ?",
                (tag_id,),
            )
            memory_ids = [r["memory_id"] for r in rows]

            vecs = [all_cached[mid] for mid in memory_ids if mid in all_cached]
            if not vecs:
                continue

            centroid = np.mean(vecs, axis=0)
            tag_centroids.append(centroid)

        if not tag_centroids:
            return None

        event_centroid = np.mean(tag_centroids, axis=0)

        self._ensure_pca_basis(all_cached)

        if self._pca_basis is not None:
            projected = self._pca_basis @ event_centroid  # shape (2,)
        else:
            if len(event_centroid) < 2:
                return None
            projected = event_centroid[:2]

        projected = np.asarray(projected, dtype=np.float32)
        angle = math.atan2(float(projected[1]), float(projected[0]))
        if angle < 0:
            angle += 2 * math.pi

        return angle, projected

    def _ensure_pca_basis(self, all_cached: dict[str, np.ndarray]) -> None:
        """Compute the 2D PCA projection basis from all memory embeddings.

        With fewer than 20 embeddings, generates a random orthonormal basis
        so the lattice has angular spread from the very first event.  Once
        enough real embeddings exist, SVD takes over and the random scaffold
        is replaced.  The basis is cached until invalidated.
        """
        if self._pca_basis is not None:
            return

        min_for_svd = 20

        if len(all_cached) >= min_for_svd:
            vecs = np.array(list(all_cached.values()))
            centered = vecs - vecs.mean(axis=0)
            try:
                _, _, Vt = np.linalg.svd(centered, full_matrices=False)
                self._pca_basis = Vt[:2]  # shape (2, d)
                return
            except np.linalg.LinAlgError:
                pass  # Fall through to random scaffold

        # Random orthonormal scaffold — gives the lattice immediate
        # angular diversity without requiring any external seed data.
        dim = self._config.embedding_dimension
        rng = np.random.default_rng()
        raw = rng.standard_normal((2, dim)).astype(np.float32)
        # Gram-Schmidt orthonormalization
        u0 = raw[0] / np.linalg.norm(raw[0])
        u1 = raw[1] - np.dot(raw[1], u0) * u0
        u1 = u1 / np.linalg.norm(u1)
        self._pca_basis = np.vstack([u0, u1])

    @staticmethod
    def _compute_proximity_threshold(
        primes: list[int], min_shared: int,
    ) -> float:
        """Compute the angular distance below which pairs share >= min_shared
        primes purely due to proximity (not structural resonance).

        Binary-searches for the distance d where the expected number of shared
        prime slots equals min_shared.  Pairs closer than d are filtered out.
        """
        two_pi = 2.0 * math.pi
        lo, hi = 0.0, math.pi
        for _ in range(60):
            mid = (lo + hi) / 2.0
            expected = sum(
                1.0 - min(mid * p / two_pi, 1.0) for p in primes
            )
            if expected > min_shared:
                lo = mid
            else:
                hi = mid
        return lo

    def _compute_prime_slots(
        self, angle: float, primes: list[int] | None = None,
    ) -> dict[int, int]:
        """Compute slot index at each prime scale for a given angle.

        For prime p the circle is divided into p equal sectors.
        """
        if primes is None:
            primes = self._config.lattice_primes
        two_pi = 2 * math.pi
        return {
            p: int(angle * p / two_pi) % p
            for p in primes
        }

    def _get_slot_populations(self) -> tuple[dict[int, dict[int, int]], int]:
        """Build prime slot population counts from all lattice positions.

        Returns (slot_populations, total_events) where slot_populations is
        {prime: {slot_index: count}}.
        """
        rows = self._db.query(
            "SELECT prime_slots FROM lattice_positions"
        )
        total = len(rows)
        pops: dict[int, dict[int, int]] = {}
        for r in rows:
            slots = json.loads(r["prime_slots"])
            for p_str, slot in slots.items():
                p = int(p_str)
                if p not in pops:
                    pops[p] = {}
                pops[p][slot] = pops[p].get(slot, 0) + 1
        return pops, total

    def _compute_poincare_coords(
        self,
        projected_2d: np.ndarray,
        prime_slots: dict[int, int],
        slot_populations: Optional[dict[int, dict[int, int]]] = None,
        total_events: Optional[int] = None,
    ) -> np.ndarray:
        """Compute Poincaré disk coordinates for a lattice position."""
        if (
            self._config.poincare_depth_enabled
            and slot_populations is not None
            and total_events is not None
            and total_events >= 2
        ):
            return self._poincare.project_with_ramsey_depth(
                projected_2d, prime_slots, slot_populations, total_events,
            )
        return self._poincare.exp_map_origin(projected_2d)

    def _load_position(self, sync_event_id: int) -> Optional[LatticePosition]:
        """Load a persisted lattice position."""
        row = self._db.execute(
            "SELECT * FROM lattice_positions WHERE sync_event_id = ?",
            (sync_event_id,),
        ).fetchone()
        if not row:
            return None
        return LatticePosition(
            id=row["id"],
            sync_event_id=row["sync_event_id"],
            angle=row["angle"],
            prime_slots=row["prime_slots"],
            poincare_x=row["poincare_x"],
            poincare_y=row["poincare_y"],
            placed_at=datetime.fromisoformat(row["placed_at"]),
        )
