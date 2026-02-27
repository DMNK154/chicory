"""Prime Ramsey lattice engine for synchronicity resonance detection."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine
from chicory.layer1.embedding_engine import EmbeddingEngine
from chicory.layer1.tag_manager import TagManager
from chicory.models.lattice import LatticePosition, Resonance, VoidProfile
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
    ) -> None:
        self._config = config
        self._db = db
        self._embeddings = embedding_engine
        self._tags = tag_manager
        self._pca_basis: Optional[np.ndarray] = None

    # ── Placement ────────────────────────────────────────────────────

    def place_event(self, event: SynchronicityEvent) -> Optional[LatticePosition]:
        """Place a synchronicity event on the prime Ramsey lattice.

        Computes angular position from tag embedding centroids, assigns slots
        at each prime scale, and persists the position.  Idempotent — placing
        the same event twice returns the existing position.

        Returns None if the event has no persisted ID or no computable angle.
        """
        if event.id is None:
            return None

        existing = self._db.execute(
            "SELECT id FROM lattice_positions WHERE sync_event_id = ?",
            (event.id,),
        ).fetchone()
        if existing:
            return self._load_position(event.id)

        angle = self._compute_angle(event)
        if angle is None:
            return None

        prime_slots = self._compute_prime_slots(angle)
        prime_slots_json = json.dumps(prime_slots)

        self._db.execute(
            """
            INSERT INTO lattice_positions (sync_event_id, angle, prime_slots)
            VALUES (?, ?, ?)
            """,
            (event.id, angle, prime_slots_json),
        )
        self._db.connection.commit()

        pos = LatticePosition(
            sync_event_id=event.id,
            angle=angle,
            prime_slots=prime_slots_json,
        )

        self._update_synchronicity_tensor(pos, event)

        return pos

    def place_events_batch(
        self, events: list[SynchronicityEvent]
    ) -> list[LatticePosition]:
        """Place multiple events on the lattice.  Returns successfully placed positions."""
        self._pca_basis = None  # Invalidate to pick up any new embeddings
        placed = []
        for event in events:
            pos = self.place_event(event)
            if pos is not None:
                placed.append(pos)
        return placed

    def invalidate_pca_cache(self) -> None:
        """Force recomputation of the PCA basis on next placement."""
        self._pca_basis = None

    def reseed(self) -> int:
        """Clear all lattice positions and re-place every synchronicity event.

        Use after changing lattice_primes to recompute all prime slots
        with the new configuration.  Returns the number of events placed.
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

    def get_lattice_state(self) -> dict:
        """Return the full lattice state for the tool handler."""
        rows = self._db.execute(
            """
            SELECT lp.sync_event_id, lp.angle, lp.prime_slots, lp.placed_at,
                   se.event_type, se.description AS event_description, se.strength
            FROM lattice_positions lp
            JOIN synchronicity_events se ON se.id = lp.sync_event_id
            ORDER BY lp.angle
            """
        ).fetchall()

        positions = []
        for r in rows:
            positions.append({
                "sync_event_id": r["sync_event_id"],
                "angle": round(r["angle"], 4),
                "prime_slots": json.loads(r["prime_slots"]),
                "placed_at": r["placed_at"],
                "event_type": r["event_type"],
                "event_description": r["event_description"],
                "event_strength": round(r["strength"], 3),
            })

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

        return {
            "positions": positions,
            "position_count": len(positions),
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
        }

    # ── Tag relational tensor ──────────────────────────────────────

    def _update_synchronicity_tensor(
        self, new_pos: LatticePosition, event: SynchronicityEvent,
    ) -> None:
        """Update the synchronicity network of the tag relational tensor
        and persist discovered resonances.

        For the newly placed position, compare against all existing positions
        (O(n) at write time).  For each resonance found:
        1. INSERT into the resonances table
        2. UPSERT tensor entries for every (tag_a, tag_b) cross-product
        """
        min_shared = self._config.lattice_min_resonance_primes
        primes = self._config.lattice_primes
        new_slots = {int(k): v for k, v in json.loads(new_pos.prime_slots).items()}
        new_tag_ids = set(json.loads(event.involved_tags))
        new_memory_ids = set(json.loads(event.involved_memories or "[]"))

        rows = self._db.execute(
            """
            SELECT lp.sync_event_id, lp.prime_slots,
                   se.involved_tags, se.involved_memories
            FROM lattice_positions lp
            JOIN synchronicity_events se ON se.id = lp.sync_event_id
            WHERE lp.sync_event_id != ?
            """,
            (event.id,),
        ).fetchall()

        for row in rows:
            other_slots = {int(k): v for k, v in json.loads(row["prime_slots"]).items()}
            shared = [p for p in primes if new_slots.get(p) == other_slots.get(p)]
            if len(shared) < min_shared:
                continue

            strength = sum(math.log(p) for p in shared)
            other_tag_ids = set(json.loads(row["involved_tags"]))
            other_memory_ids = set(json.loads(row["involved_memories"] or "[]"))
            combined_memories = sorted(new_memory_ids | other_memory_ids)

            # Persist resonance
            id_a = min(event.id, row["sync_event_id"])
            id_b = max(event.id, row["sync_event_id"])
            chance = math.exp(-strength)
            description = (
                f"Events {id_a} and {id_b} resonate across {len(shared)} "
                f"prime scales ({', '.join(str(p) for p in shared)}). "
                f"Surprise: {strength:.1f} nats. "
                f"Chance probability: {chance:.2e}"
            )
            self._db.execute(
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
                (id_a, id_b, json.dumps([id_a, id_b]),
                 json.dumps(shared), strength, description),
            )

            for ta in new_tag_ids:
                for tb in other_tag_ids:
                    key_a, key_b = min(ta, tb), max(ta, tb)
                    if key_a == key_b:
                        continue
                    self._db.execute(
                        """
                        INSERT INTO tag_relational_tensor
                            (tag_a_id, tag_b_id, synchronicity_strength, memory_ids)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(tag_a_id, tag_b_id) DO UPDATE SET
                            synchronicity_strength = MAX(
                                tag_relational_tensor.synchronicity_strength,
                                excluded.synchronicity_strength
                            ),
                            memory_ids = excluded.memory_ids,
                            updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                        """,
                        (key_a, key_b, strength, json.dumps(combined_memories)),
                    )

        self._db.connection.commit()

    def update_cooccurrence_tensor(self) -> int:
        """Populate the co-occurrence network from memory_tags PMI.

        PMI(a,b) = log(P(a,b) / (P(a) * P(b))) measures information-theoretic
        surprise of tag co-occurrence, paralleling how lattice resonance uses
        sum(log(p)) for structural surprise.

        Returns the number of tensor entries created/updated.
        """
        total_row = self._db.execute(
            "SELECT COUNT(DISTINCT memory_id) as cnt FROM memory_tags"
        ).fetchone()
        total_memories = total_row["cnt"] if total_row else 0
        if total_memories < 2:
            return 0

        # Per-tag memory counts
        tag_counts: dict[int, int] = {}
        rows = self._db.execute(
            "SELECT tag_id, COUNT(DISTINCT memory_id) as cnt FROM memory_tags GROUP BY tag_id"
        ).fetchall()
        for r in rows:
            tag_counts[r["tag_id"]] = r["cnt"]

        # Co-occurrence pairs
        cooccurrences = self._tags.get_all_co_occurrences(min_count=2)
        if not cooccurrences:
            return 0

        updated = 0
        for tag_a, tag_b, count in cooccurrences:
            p_a = tag_counts.get(tag_a, 0) / total_memories
            p_b = tag_counts.get(tag_b, 0) / total_memories
            p_ab = count / total_memories

            if p_a <= 0 or p_b <= 0 or p_ab <= 0:
                continue

            pmi = math.log(p_ab / (p_a * p_b))
            if pmi <= 0:
                continue  # Only store surprising (positive PMI) associations

            # Get memory IDs that have both tags
            memory_rows = self._db.execute(
                """
                SELECT a.memory_id FROM memory_tags a
                JOIN memory_tags b ON a.memory_id = b.memory_id
                WHERE a.tag_id = ? AND b.tag_id = ?
                """,
                (tag_a, tag_b),
            ).fetchall()
            memory_ids = sorted({r["memory_id"] for r in memory_rows})

            key_a, key_b = min(tag_a, tag_b), max(tag_a, tag_b)
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
                (key_a, key_b, pmi, json.dumps(memory_ids)),
            )
            updated += 1

        self._db.connection.commit()
        return updated

    def update_semantic_tensor(self) -> int:
        """Populate the semantic network from embedding cosine similarities.

        For each tag pair with co-occurrence >= 1, computes cosine similarity
        between tag centroids (average embedding of associated memories).

        Returns the number of tensor entries created/updated.
        """
        all_cached = self._embeddings.get_all_cached()
        if not all_cached:
            return 0

        # Compute tag centroids
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

        # Compute pairwise cosine similarities for co-occurring tags
        cooccurrences = self._tags.get_all_co_occurrences(min_count=1)
        updated = 0

        for tag_a, tag_b, _count in cooccurrences:
            if tag_a not in tag_centroids or tag_b not in tag_centroids:
                continue

            similarity = float(np.dot(tag_centroids[tag_a], tag_centroids[tag_b]))
            if similarity <= 0:
                continue

            key_a, key_b = min(tag_a, tag_b), max(tag_a, tag_b)
            combined_memories = sorted(
                tag_memory_sets.get(tag_a, set()) | tag_memory_sets.get(tag_b, set())
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
                (key_a, key_b, similarity, json.dumps(combined_memories)),
            )
            updated += 1

        self._db.connection.commit()
        return updated

    def update_semiotic_tensor(self) -> int:
        """Populate the semiotic network from conditional probability asymmetry.

        For each co-occurring tag pair (A, B) with A < B:
        - semiotic_forward  = P(B|A) = co_count / count(A)
        - semiotic_reverse  = P(A|B) = co_count / count(B)

        These are naturally asymmetric when tags have different frequencies.
        E.g. "chicory" on 10 memories, 8 also have "emergence"; "emergence" on 100.
        P(emergence|chicory) = 0.8, P(chicory|emergence) = 0.08.

        Returns the number of tensor entries created/updated.
        """
        # Per-tag memory counts
        tag_counts: dict[int, int] = {}
        rows = self._db.execute(
            "SELECT tag_id, COUNT(DISTINCT memory_id) as cnt FROM memory_tags GROUP BY tag_id"
        ).fetchall()
        for r in rows:
            tag_counts[r["tag_id"]] = r["cnt"]

        if not tag_counts:
            return 0

        # Co-occurrence pairs (tag_a < tag_b guaranteed)
        cooccurrences = self._tags.get_all_co_occurrences(min_count=1)
        if not cooccurrences:
            return 0

        updated = 0
        for tag_a, tag_b, co_count in cooccurrences:
            count_a = tag_counts.get(tag_a, 0)
            count_b = tag_counts.get(tag_b, 0)
            if count_a == 0 or count_b == 0:
                continue

            forward = co_count / count_a   # P(B|A)
            reverse = co_count / count_b   # P(A|B)

            # Get memory IDs that have both tags
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

        Populates R_cooccurrence, R_semantic, and R_semiotic independently.
        R_synchronicity is left at 0 (populated by lattice placement).
        """
        self.update_cooccurrence_tensor()
        self.update_semantic_tensor()
        self.update_semiotic_tensor()

    def rebuild_tensor(self) -> int:
        """Rebuild the entire tag relational tensor and resonances table from scratch.

        Clears all entries and repopulates all three networks:
        1. R_cooccurrence from memory_tags PMI
        2. R_semantic from embedding cosine similarities
        3. R_synchronicity from lattice resonances (inline pairwise, O(1) memory)

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

        # Populate synchronicity network — inline pairwise without list accumulation
        min_shared = self._config.lattice_min_resonance_primes
        primes = self._config.lattice_primes

        rows = self._db.execute(
            """
            SELECT lp.sync_event_id, lp.prime_slots,
                   se.involved_tags, se.involved_memories
            FROM lattice_positions lp
            JOIN synchronicity_events se ON se.id = lp.sync_event_id
            """
        ).fetchall()

        # Pre-parse all positions
        positions: list[tuple[int, dict[int, int], set[int], set[str]]] = []
        for r in rows:
            slots = {int(k): v for k, v in json.loads(r["prime_slots"]).items()}
            tag_ids = set(json.loads(r["involved_tags"]))
            memory_ids = set(json.loads(r["involved_memories"] or "[]"))
            positions.append((r["sync_event_id"], slots, tag_ids, memory_ids))

        for i in range(len(positions)):
            id_i, slots_i, tags_i, mems_i = positions[i]
            for j in range(i + 1, len(positions)):
                id_j, slots_j, tags_j, mems_j = positions[j]

                shared = [p for p in primes if slots_i.get(p) == slots_j.get(p)]
                if len(shared) < min_shared:
                    continue

                strength = sum(math.log(p) for p in shared)

                # Persist resonance
                id_a, id_b = min(id_i, id_j), max(id_i, id_j)
                chance = math.exp(-strength)
                description = (
                    f"Events {id_a} and {id_b} resonate across {len(shared)} "
                    f"prime scales ({', '.join(str(p) for p in shared)}). "
                    f"Surprise: {strength:.1f} nats. "
                    f"Chance probability: {chance:.2e}"
                )
                self._db.execute(
                    """
                    INSERT OR IGNORE INTO resonances
                        (event_a_id, event_b_id, event_ids,
                         shared_primes, resonance_strength, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (id_a, id_b, json.dumps([id_a, id_b]),
                     json.dumps(shared), strength, description),
                )

                # UPSERT tensor
                combined_memories = json.dumps(sorted(mems_i | mems_j))
                for ta in tags_i:
                    for tb in tags_j:
                        key_a, key_b = min(ta, tb), max(ta, tb)
                        if key_a == key_b:
                            continue
                        self._db.execute(
                            """
                            INSERT INTO tag_relational_tensor
                                (tag_a_id, tag_b_id, synchronicity_strength, memory_ids)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(tag_a_id, tag_b_id) DO UPDATE SET
                                synchronicity_strength = MAX(
                                    tag_relational_tensor.synchronicity_strength,
                                    excluded.synchronicity_strength
                                ),
                                memory_ids = CASE
                                    WHEN length(tag_relational_tensor.memory_ids) > length(excluded.memory_ids)
                                    THEN tag_relational_tensor.memory_ids
                                    ELSE excluded.memory_ids
                                END,
                                updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now')
                            """,
                            (key_a, key_b, strength, combined_memories),
                        )

        self._db.connection.commit()

        row = self._db.execute(
            "SELECT COUNT(*) as cnt FROM tag_relational_tensor"
        ).fetchone()
        return row["cnt"] if row else 0

    def get_resonant_memory_ids_fast(
        self,
        tag_ids: list[int],
    ) -> list[tuple[str, float]]:
        """O(k) tensor-based lookup for resonant memories.

        Given query-context tag_ids, look up the tag relational tensor and
        combine the four network strengths using configurable weights.
        The semiotic layer is direction-aware: when a query tag is tag_a_id,
        we use semiotic_forward (P(B|A)); when it's tag_b_id, we use
        semiotic_reverse (P(A|B)).
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
                   memory_ids
            FROM tag_relational_tensor
            WHERE (tag_a_id IN ({placeholders}) OR tag_b_id IN ({placeholders}))
              AND (cooccurrence_strength > 0
                   OR synchronicity_strength > 0
                   OR semantic_strength > 0
                   OR semiotic_forward > 0
                   OR semiotic_reverse > 0)
            """,
            (*tag_ids, *tag_ids),
        ).fetchall()

        if not rows:
            return []

        w_co = self._config.tensor_cooccurrence_weight
        w_sync = self._config.tensor_synchronicity_weight
        w_sem = self._config.tensor_semantic_weight
        w_semio = self._config.tensor_semiotic_weight

        memory_scores: dict[str, float] = {}
        max_combined = 0.0

        combined_values: list[tuple[float, list[str]]] = []
        for row in rows:
            # Direction-aware semiotic contribution
            a_in_query = row["tag_a_id"] in query_tag_set
            b_in_query = row["tag_b_id"] in query_tag_set
            if a_in_query and b_in_query:
                semiotic = max(row["semiotic_forward"], row["semiotic_reverse"])
            elif a_in_query:
                semiotic = row["semiotic_forward"]   # query is A, pulling B
            else:
                semiotic = row["semiotic_reverse"]   # query is B, pulling A

            combined = (
                w_co * row["cooccurrence_strength"]
                + w_sync * row["synchronicity_strength"]
                + w_sem * row["semantic_strength"]
                + w_semio * semiotic
            )
            if combined > max_combined:
                max_combined = combined
            combined_values.append((combined, json.loads(row["memory_ids"])))

        if max_combined <= 0:
            return []

        for combined, mids in combined_values:
            normalized = combined / max_combined
            for mid in mids:
                memory_scores[mid] = max(memory_scores.get(mid, 0.0), normalized)

        return sorted(memory_scores.items(), key=lambda x: x[1], reverse=True)

    # ── Internal ─────────────────────────────────────────────────────

    def _compute_angle(self, event: SynchronicityEvent) -> Optional[float]:
        """Project an event's involved-tag embedding centroids to an angle.

        1. Parse involved_tags → list of tag IDs
        2. For each tag, average the embeddings of all memories with that tag
        3. Average the tag centroids to get an event centroid
        4. PCA-project to 2D and convert to angle via atan2
        """
        tag_ids = json.loads(event.involved_tags)
        if not tag_ids:
            return None

        all_cached = self._embeddings.get_all_cached()
        if not all_cached:
            return None

        tag_centroids: list[np.ndarray] = []
        for tag_id in tag_ids:
            rows = self._db.execute(
                "SELECT memory_id FROM memory_tags WHERE tag_id = ?",
                (tag_id,),
            ).fetchall()
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
            # Fallback: use first two embedding dimensions directly
            if len(event_centroid) < 2:
                return None
            projected = event_centroid[:2]

        angle = math.atan2(float(projected[1]), float(projected[0]))
        if angle < 0:
            angle += 2 * math.pi

        return angle

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

    def _compute_prime_slots(self, angle: float) -> dict[int, int]:
        """Compute slot index at each prime scale for a given angle.

        For prime p the circle is divided into p equal sectors.
        """
        two_pi = 2 * math.pi
        return {
            p: int(angle * p / two_pi) % p
            for p in self._config.lattice_primes
        }

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
            placed_at=datetime.fromisoformat(row["placed_at"]),
        )
