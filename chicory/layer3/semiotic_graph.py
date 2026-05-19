"""Directed semiotic graph over the tag relational tensor.

Preserves the asymmetry between semiotic_forward and semiotic_reverse
as two distinct edge sets per tag: signifies (out) and signified_by (in).

Two traversal modes, used at different pipeline stages:
- IN→OUT (low-level): lateral discovery of sibling tags through shared signifiers
- OUT→IN (high-level): convergence validation of candidate memories
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field


@dataclass
class DiscoveredTag:
    tag_id: int
    max_strength: float
    paths: list[tuple[int, float]] = field(default_factory=list)

    @property
    def convergence(self) -> int:
        return len(self.paths)

    @property
    def score(self) -> float:
        return self.max_strength * (1.0 + math.log(max(self.convergence, 1)))


class SemioticDirectedGraph:
    """Directed view of the semiotic layer in tag_relational_tensor.

    Edge convention (enforced by CHECK tag_a_id < tag_b_id):
      semiotic_forward = strength of edge A → B  (A signifies B)
      semiotic_reverse = strength of edge B → A  (B signifies A)
    """

    def __init__(
        self,
        db: sqlite3.Connection,
        min_strength: float = 0.05,
        tensor_cache: object | None = None,
    ):
        self._db = db
        self._min_strength = min_strength
        self._tensor_cache = tensor_cache

    def signifies(self, tag_id: int) -> list[tuple[int, float]]:
        """Tags that tag_id points TO (its signifieds).

        When tag_id is in A position: outgoing = semiotic_forward (A→B)
        When tag_id is in B position: outgoing = semiotic_reverse (B→A)
        """
        if self._tensor_cache:
            result: list[tuple[int, float]] = []
            for r in self._tensor_cache.rows_for_a({tag_id}):
                if r["semiotic_forward"] > self._min_strength:
                    result.append((r["tag_b_id"], r["semiotic_forward"]))
            for r in self._tensor_cache.rows_for_b({tag_id}):
                if r["semiotic_reverse"] > self._min_strength:
                    result.append((r["tag_a_id"], r["semiotic_reverse"]))
            return result
        rows_a = self._db.execute(
            "SELECT tag_b_id AS partner, semiotic_forward AS strength "
            "FROM tag_relational_tensor WHERE tag_a_id = ? AND semiotic_forward > ?",
            (tag_id, self._min_strength),
        ).fetchall()
        rows_b = self._db.execute(
            "SELECT tag_a_id AS partner, semiotic_reverse AS strength "
            "FROM tag_relational_tensor WHERE tag_b_id = ? AND semiotic_reverse > ?",
            (tag_id, self._min_strength),
        ).fetchall()
        return [(r["partner"], r["strength"]) for r in rows_a] + [(r["partner"], r["strength"]) for r in rows_b]

    def signified_by(self, tag_id: int) -> list[tuple[int, float]]:
        """Tags that point TO tag_id (its signifiers).

        When tag_id is in B position: incoming = semiotic_forward (A→B, so A signifies B)
        When tag_id is in A position: incoming = semiotic_reverse (B→A, so B signifies A)
        """
        if self._tensor_cache:
            result: list[tuple[int, float]] = []
            for r in self._tensor_cache.rows_for_b({tag_id}):
                if r["semiotic_forward"] > self._min_strength:
                    result.append((r["tag_a_id"], r["semiotic_forward"]))
            for r in self._tensor_cache.rows_for_a({tag_id}):
                if r["semiotic_reverse"] > self._min_strength:
                    result.append((r["tag_b_id"], r["semiotic_reverse"]))
            return result
        rows_a = self._db.execute(
            "SELECT tag_a_id AS partner, semiotic_forward AS strength "
            "FROM tag_relational_tensor WHERE tag_b_id = ? AND semiotic_forward > ?",
            (tag_id, self._min_strength),
        ).fetchall()
        rows_b = self._db.execute(
            "SELECT tag_b_id AS partner, semiotic_reverse AS strength "
            "FROM tag_relational_tensor WHERE tag_a_id = ? AND semiotic_reverse > ?",
            (tag_id, self._min_strength),
        ).fetchall()
        return [(r["partner"], r["strength"]) for r in rows_a] + [(r["partner"], r["strength"]) for r in rows_b]

    def signifies_batch(self, tag_ids: list[int]) -> dict[int, list[tuple[int, float]]]:
        """Signifieds for multiple source tags in two queries instead of 2*N."""
        if not tag_ids:
            return {}
        result: dict[int, list[tuple[int, float]]] = {tid: [] for tid in tag_ids}
        if self._tensor_cache:
            tag_set = set(tag_ids)
            for r in self._tensor_cache.rows_for_a(tag_set):
                if r["semiotic_forward"] > self._min_strength:
                    result[r["tag_a_id"]].append((r["tag_b_id"], r["semiotic_forward"]))
            for r in self._tensor_cache.rows_for_b(tag_set):
                if r["semiotic_reverse"] > self._min_strength:
                    result[r["tag_b_id"]].append((r["tag_a_id"], r["semiotic_reverse"]))
            return result
        ph = ",".join("?" * len(tag_ids))
        params = tuple(tag_ids)
        rows_a = self._db.execute(
            f"SELECT tag_a_id AS src, tag_b_id AS partner, semiotic_forward AS strength "
            f"FROM tag_relational_tensor WHERE tag_a_id IN ({ph}) AND semiotic_forward > ?",
            (*params, self._min_strength),
        ).fetchall()
        rows_b = self._db.execute(
            f"SELECT tag_b_id AS src, tag_a_id AS partner, semiotic_reverse AS strength "
            f"FROM tag_relational_tensor WHERE tag_b_id IN ({ph}) AND semiotic_reverse > ?",
            (*params, self._min_strength),
        ).fetchall()
        for r in rows_a:
            result[r["src"]].append((r["partner"], r["strength"]))
        for r in rows_b:
            result[r["src"]].append((r["partner"], r["strength"]))
        return result

    def signified_by_batch(self, tag_ids: list[int]) -> dict[int, list[tuple[int, float]]]:
        """Signifiers for multiple target tags in two queries instead of 2*N."""
        if not tag_ids:
            return {}
        result: dict[int, list[tuple[int, float]]] = {tid: [] for tid in tag_ids}
        if self._tensor_cache:
            tag_set = set(tag_ids)
            for r in self._tensor_cache.rows_for_b(tag_set):
                if r["semiotic_forward"] > self._min_strength:
                    result[r["tag_b_id"]].append((r["tag_a_id"], r["semiotic_forward"]))
            for r in self._tensor_cache.rows_for_a(tag_set):
                if r["semiotic_reverse"] > self._min_strength:
                    result[r["tag_a_id"]].append((r["tag_b_id"], r["semiotic_reverse"]))
            return result
        ph = ",".join("?" * len(tag_ids))
        params = tuple(tag_ids)
        rows_a = self._db.execute(
            f"SELECT tag_b_id AS target, tag_a_id AS signifier, semiotic_forward AS strength "
            f"FROM tag_relational_tensor WHERE tag_b_id IN ({ph}) AND semiotic_forward > ?",
            (*params, self._min_strength),
        ).fetchall()
        rows_b = self._db.execute(
            f"SELECT tag_a_id AS target, tag_b_id AS signifier, semiotic_reverse AS strength "
            f"FROM tag_relational_tensor WHERE tag_a_id IN ({ph}) AND semiotic_reverse > ?",
            (*params, self._min_strength),
        ).fetchall()
        for r in rows_a:
            result[r["target"]].append((r["signifier"], r["strength"]))
        for r in rows_b:
            result[r["target"]].append((r["signifier"], r["strength"]))
        return result

    # ── Low-level: IN→OUT lateral discovery ──────────────────────────────

    def expand_in_through_out(
        self,
        query_tag_ids: list[int],
    ) -> dict[int, DiscoveredTag]:
        """Two-stage directed expansion for lateral sibling discovery.

        Stage 1 (IN): Find signifiers of query tags — the sources/frames
        Stage 2 (OUT): Find signifieds of those frames — sibling concepts

        Discovers tags reachable via the directed path:
            query ←IN← intermediate →OUT→ discovered

        No artificial top-K caps — min_strength is the only gate.
        """
        query_set = set(query_tag_ids)

        # Stage 1: gather intermediates (signifiers of query tags) — batch
        all_signifiers = self.signified_by_batch(query_tag_ids)
        intermediates: dict[int, float] = {}
        for tag_id in query_tag_ids:
            for signifier_id, strength in all_signifiers.get(tag_id, []):
                if signifier_id not in query_set:
                    intermediates[signifier_id] = max(
                        intermediates.get(signifier_id, 0.0), strength
                    )

        if not intermediates:
            return {}

        # Stage 2: expand OUT from intermediates — batch
        inter_ids = list(intermediates.keys())
        all_signifieds = self.signifies_batch(inter_ids)
        discovered: dict[int, DiscoveredTag] = {}
        for inter_id in inter_ids:
            in_strength = intermediates[inter_id]
            for target_id, out_strength in all_signifieds.get(inter_id, []):
                if target_id in query_set or target_id in intermediates:
                    continue

                path_strength = in_strength * out_strength
                if target_id in discovered:
                    entry = discovered[target_id]
                    entry.max_strength = max(entry.max_strength, path_strength)
                    entry.paths.append((inter_id, path_strength))
                else:
                    discovered[target_id] = DiscoveredTag(
                        tag_id=target_id,
                        max_strength=path_strength,
                        paths=[(inter_id, path_strength)],
                    )

        return discovered

    # ── High-level: OUT→IN convergence validation ────────────────────────

    def convergence_score(
        self,
        query_tag_ids: list[int],
        candidate_tag_ids: list[int],
    ) -> float:
        """How many of the query's signifieds independently point at candidate tags?

        Stage 1 (OUT): Decompose query tags into what they signify
        Stage 2 (IN): For each candidate tag, check which query-signifieds
                      also appear as signifiers of the candidate

        High score = multiple angles of the query's decomposition converge
        on this candidate independently.
        """
        query_set = set(query_tag_ids)
        candidate_set = set(candidate_tag_ids)

        # Stage 1: OUT from query — what do the query tags decompose into?
        query_signifieds: set[int] = set()
        for qt in query_tag_ids:
            for target_id, _ in self.signifies(qt):
                if target_id not in query_set and target_id not in candidate_set:
                    query_signifieds.add(target_id)

        if not query_signifieds:
            return 0.0

        # Stage 2: IN to candidate — which query-signifieds point at candidate?
        convergent_paths = 0
        for ct in candidate_tag_ids:
            signifiers = set(t for t, _ in self.signified_by(ct))
            convergent_paths += len(query_signifieds & signifiers)

        return float(convergent_paths)

    def convergence_scores_batch(
        self,
        query_tag_ids: list[int],
        candidates: dict[str, list[int]],
    ) -> dict[str, float]:
        """Batch convergence scoring for multiple candidate memories.

        Args:
            query_tag_ids: Tags from the query context
            candidates: {memory_id: [tag_ids]} for each candidate memory

        Returns:
            {memory_id: convergence_score}
        """
        query_set = set(query_tag_ids)

        all_signifieds = self.signifies_batch(query_tag_ids)
        query_signifieds: set[int] = set()
        for qt in query_tag_ids:
            for target_id, _ in all_signifieds.get(qt, []):
                if target_id not in query_set:
                    query_signifieds.add(target_id)

        if not query_signifieds:
            return {}

        all_candidate_tags = list({
            ct for tag_ids in candidates.values()
            for ct in tag_ids if ct not in query_set
        })
        raw_signifiers = self.signified_by_batch(all_candidate_tags)
        signifier_cache: dict[int, set[int]] = {
            ct: {t for t, _ in pairs} for ct, pairs in raw_signifiers.items()
        }

        scores: dict[str, float] = {}
        for mid, tag_ids in candidates.items():
            convergent = 0
            for ct in tag_ids:
                if ct in query_set:
                    continue
                convergent += len(query_signifieds & signifier_cache.get(ct, set()))
            if convergent > 0:
                scores[mid] = float(convergent)

        return scores
