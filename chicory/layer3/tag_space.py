"""Tag Space — independent tag-graph retrieval path.

Two entry points, used at different stages of retrieve_hybrid:
  1. Lexical: tokenize query text → match against tag names → fan-out
     through tensor edges → resolve to memory scores.  Runs BEFORE
     embedding (instant, no model inference).
  2. Centroid: cosine similarity against tag centroid matrix → fan-out
     → resolve.  Piggybacks on the embedding already computed for FAISS.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from chicory.config import ChicoryConfig
from chicory.db.engine import DatabaseEngine

if TYPE_CHECKING:
    from chicory.layer1.tag_manager import TagManager


class TagSpace:
    """Independent tag-graph retrieval alongside FAISS."""

    def __init__(
        self,
        config: ChicoryConfig,
        db: DatabaseEngine,
        tag_manager: TagManager,
        tensor_cache: object | None = None,
    ) -> None:
        self._config = config
        self._db = db
        self._tags = tag_manager
        self._tensor_cache = tensor_cache
        self._lexical_index: dict[str, list[int]] | None = None
        self._reverse_index: dict[int, set[str]] | None = None
        self._exact_name_ids: dict[str, set[int]] | None = None
        self._centroid_cache: tuple[list[int], np.ndarray] | None = None

    # ── Lexical index ───────────────────────────────────────────────

    _SKIP_PREFIXES = ("month-", "day-")
    _TOKENIZE_RE = re.compile(r"[a-z0-9]+")

    def _build_lexical_index(
        self,
    ) -> tuple[dict[str, list[int]], dict[int, set[str]], dict[str, set[int]]]:
        rows = self._db.execute(
            "SELECT id, name FROM tags WHERE is_active = 1"
        ).fetchall()
        forward: dict[str, list[int]] = {}
        reverse: dict[int, set[str]] = {}
        exact: dict[str, set[int]] = {}
        for r in rows:
            tag_id = r["id"]
            name: str = r["name"]
            if any(name.startswith(p) for p in self._SKIP_PREFIXES):
                continue
            if len(name) == 1 or name.isdigit():
                continue
            forward.setdefault(name, []).append(tag_id)
            reverse.setdefault(tag_id, set()).add(name)
            exact.setdefault(name, set()).add(tag_id)
            parts = re.split(r"[-_]", name)
            if len(parts) > 1:
                for part in parts:
                    if len(part) >= self._config.tag_space_lexical_min_tag_length:
                        forward.setdefault(part, []).append(tag_id)
                        reverse.setdefault(tag_id, set()).add(part)
        return forward, reverse, exact

    def _get_lexical_index(self) -> dict[str, list[int]]:
        if self._lexical_index is None:
            self._lexical_index, self._reverse_index, self._exact_name_ids = (
                self._build_lexical_index()
            )
        return self._lexical_index

    # ── Lexical match ───────────────────────────────────────────────

    def lexical_match(self, query: str) -> set[int]:
        """Match query words against the tag lexical index."""
        index = self._get_lexical_index()
        min_len = self._config.tag_space_lexical_min_tag_length
        tokens = self._TOKENIZE_RE.findall(query.lower())
        tokens = [t for t in tokens if len(t) >= min_len]

        matched: set[int] = set()
        for token in tokens:
            if token in index:
                matched.update(index[token])

        for i in range(len(tokens) - 1):
            compound = f"{tokens[i]}-{tokens[i+1]}"
            if compound in index:
                matched.update(index[compound])

        return matched

    # ── Tensor fan-out ──────────────────────────────────────────────

    def fan_out(
        self,
        seed_tag_ids: set[int],
        depth: int | None = None,
    ) -> dict[int, float]:
        """BFS through tensor edges from seed tags, scoring partners."""
        if not seed_tag_ids:
            return {}
        depth = depth if depth is not None else self._config.tag_space_fan_depth

        w_co = self._config.tensor_cooccurrence_weight
        w_sync = self._config.tensor_synchronicity_weight
        w_sem = self._config.tensor_semantic_weight
        w_glyph = self._config.tensor_glyph_weight
        w_meta = self._config.tensor_meta_resonance_weight
        w_inhib = self._config.tensor_inhibition_weight
        min_score = self._config.resonance_min_tensor_score

        visited = set(seed_tag_ids)
        frontier = set(seed_tag_ids)
        all_scores: dict[int, float] = {}
        decay = 1.0

        for _ in range(depth):
            if not frontier:
                break

            frontier_list = list(frontier)

            if self._tensor_cache:
                rows = self._tensor_cache.rows_touching(frontier_list)
            else:
                ph = ",".join("?" * len(frontier_list))
                params = tuple(frontier_list)
                rows_a = self._db.execute(
                    f"SELECT tag_a_id, tag_b_id, cooccurrence_strength,"
                    f" synchronicity_strength, semantic_strength,"
                    f" glyph_strength, inhibition_strength, parallelness"
                    f" FROM tag_relational_tensor WHERE tag_a_id IN ({ph})",
                    params,
                ).fetchall()
                rows_b = self._db.execute(
                    f"SELECT tag_a_id, tag_b_id, cooccurrence_strength,"
                    f" synchronicity_strength, semantic_strength,"
                    f" glyph_strength, inhibition_strength, parallelness"
                    f" FROM tag_relational_tensor WHERE tag_b_id IN ({ph})",
                    params,
                ).fetchall()
                seen_pairs = {(r["tag_a_id"], r["tag_b_id"]) for r in rows_a}
                rows = list(rows_a) + [
                    r for r in rows_b
                    if (r["tag_a_id"], r["tag_b_id"]) not in seen_pairs
                ]

            next_frontier: set[int] = set()
            for row in rows:
                a_in = row["tag_a_id"] in visited
                b_in = row["tag_b_id"] in visited
                if a_in and b_in:
                    continue
                partner = row["tag_b_id"] if a_in else row["tag_a_id"]

                sync_val = row["synchronicity_strength"]
                glyph_val = row["glyph_strength"]
                excitatory = (
                    w_co * row["cooccurrence_strength"]
                    + w_sync * sync_val
                    + w_sem * row["semantic_strength"]
                    + w_glyph * glyph_val
                )
                if sync_val > 0 and glyph_val > 0:
                    excitatory += w_meta * sync_val * glyph_val

                inhib = row["inhibition_strength"]
                par = row["parallelness"]
                suppression = (
                    w_inhib * inhib * max(0.0, -par)
                    if inhib > 0 and par < 0
                    else 0.0
                )

                combined = (excitatory - suppression) * decay
                if combined >= min_score:
                    all_scores[partner] = max(
                        all_scores.get(partner, 0.0), combined,
                    )
                    next_frontier.add(partner)

            visited.update(next_frontier)
            frontier = next_frontier
            decay *= 0.5

        if not all_scores:
            return all_scores

        # Path normalization: divide each partner's score by its total
        # tensor degree so hub tags with hundreds of edges don't dominate.
        partner_list = list(all_scores.keys())
        if self._tensor_cache:
            degree = self._tensor_cache.degree(partner_list)
        else:
            p_ph = ",".join("?" * len(partner_list))
            p_params = tuple(partner_list)
            degree = {tid: 0 for tid in partner_list}
            for col in ("tag_a_id", "tag_b_id"):
                rows = self._db.execute(
                    f"SELECT {col} AS tag, COUNT(*) AS cnt "
                    f"FROM tag_relational_tensor WHERE {col} IN ({p_ph}) "
                    f"GROUP BY {col}",
                    p_params,
                ).fetchall()
                for r in rows:
                    degree[r["tag"]] += r["cnt"]

        for tid in partner_list:
            d = degree[tid]
            if d > 1:
                all_scores[tid] /= d

        return all_scores

    # ── Centroid match ──────────────────────────────────────────────

    def centroid_match(
        self, query_vec: np.ndarray, threshold: float | None = None,
    ) -> set[int]:
        """Find tags whose centroids are similar to query_vec."""
        if self._centroid_cache is not None:
            tag_ids, centroid_matrix = self._centroid_cache
        else:
            rows = self._db.execute(
                "SELECT tag_id, centroid FROM tag_centroids"
            ).fetchall()
            if not rows:
                return set()
            tag_ids = [r["tag_id"] for r in rows]
            centroid_matrix = np.stack([
                np.frombuffer(r["centroid"], dtype=np.float32) for r in rows
            ])
            self._centroid_cache = (tag_ids, centroid_matrix)

        similarities = centroid_matrix @ query_vec
        thresh = threshold if threshold is not None else self._config.tag_space_centroid_similarity_threshold
        return {
            tag_ids[i]
            for i in range(len(tag_ids))
            if similarities[i] >= thresh
        }

    # ── Memory resolution ───────────────────────────────────────────

    def resolve_to_memories(
        self, tag_scores: dict[int, float],
    ) -> dict[str, float]:
        """Convert tag scores to memory scores via memory_tags."""
        if not tag_scores:
            return {}

        tag_list = list(tag_scores.keys())
        mem_scores: dict[str, float] = {}

        for i in range(0, len(tag_list), 500):
            chunk = tag_list[i:i + 500]
            ph = ",".join("?" * len(chunk))
            rows = self._db.execute(
                f"SELECT memory_id, tag_id FROM memory_tags WHERE tag_id IN ({ph})",
                tuple(chunk),
            ).fetchall()
            for r in rows:
                score = tag_scores[r["tag_id"]]
                mem_scores[r["memory_id"]] = max(
                    mem_scores.get(r["memory_id"], 0.0), score,
                )

        if not mem_scores:
            return {}

        max_score = max(mem_scores.values())
        if max_score > 0:
            return {mid: s / max_score for mid, s in mem_scores.items()}
        return mem_scores

    # ── Convenience: score_lexical / score_centroid ─────────────────

    def score_lexical(
        self, query: str,
    ) -> tuple[dict[str, float], set[int]]:
        """Sequential lexical expansion: each token fans out before later tokens match.

        Earlier tokens expand through the tensor, and their expanded tags
        become matchable seeds for later tokens.  "embedding engine" first
        matches "embedding", fans out to discover "embedding-engine", and
        then "engine" matches that expanded tag — promoting it to a seed
        for its own fan-out.
        """
        index = self._get_lexical_index()
        reverse = self._reverse_index or {}
        exact = self._exact_name_ids or {}
        min_len = self._config.tag_space_lexical_min_tag_length
        tokens = self._TOKENIZE_RE.findall(query.lower())
        tokens = [t for t in tokens if len(t) >= min_len]
        if not tokens:
            return {}, set()

        all_seeds: set[int] = set()
        all_tag_scores: dict[int, float] = {}
        expanded_ids: set[int] = set()

        for i, token in enumerate(tokens):
            direct: set[int] = set()
            if token in index:
                ids = set(index[token])
                exact_ids = exact.get(token, set())
                if exact_ids and len(ids) > 2 * len(exact_ids):
                    # Token is a common compound part — discard all matches
                    pass
                elif exact_ids:
                    # Token has an exact tag but few/no compound noise —
                    # keep the exact tag, discard compound decompositions
                    direct.update(exact_ids)
                else:
                    direct.update(ids)
            if i < len(tokens) - 1:
                compound = f"{tokens[i]}-{tokens[i + 1]}"
                if compound in index:
                    direct.update(index[compound])

            # Expanded tags from earlier fan-outs that match this token
            for tid in expanded_ids - all_seeds:
                if token in reverse.get(tid, set()):
                    direct.add(tid)

            truly_new = direct - all_seeds
            all_seeds.update(direct)

            if truly_new:
                fan_scores = self.fan_out(truly_new)
                for tid, score in fan_scores.items():
                    all_tag_scores[tid] = max(
                        all_tag_scores.get(tid, 0.0), score,
                    )
                expanded_ids.update(fan_scores.keys())
            expanded_ids.update(direct)

        for sid in all_seeds:
            all_tag_scores[sid] = max(all_tag_scores.get(sid, 0.0), 1.0)

        all_tag_scores = self._apply_inward_ratio(all_seeds, all_tag_scores)
        return self.resolve_to_memories(all_tag_scores), all_seeds

    # ── Inward ratio ───────────────────────────────────────────────

    def _apply_inward_ratio(
        self,
        seed_tags: set[int],
        tag_scores: dict[int, float],
    ) -> dict[int, float]:
        """Scale tag scores by seed convergence × inward fraction.

        Two signals multiplied together:
        1. Convergence: what fraction of seeds reach this tag (via BFS
           constrained to the search subgraph).
        2. Inward fraction: seed edges / total degree — how much of
           this tag's connectivity points toward the query vs outward.

        Combined ratio = convergence × inward_fraction.  Tags need both
        multi-seed agreement AND a non-trivial fraction of edges pointing
        toward the query.  Hub tags that connect to everything get crushed
        by the inward fraction even if convergence is high.
        """
        discovered = [tid for tid in tag_scores if tid not in seed_tags]
        if not discovered:
            return tag_scores

        depth = self._config.tag_space_fan_depth
        search_set = set(tag_scores.keys())
        search_list = list(search_set)
        sg_ph = ",".join("?" * len(search_list))
        sg_params = tuple(search_list)

        # ── Convergence: BFS with provenance tracking ──
        provenance: dict[int, set[int]] = {s: {s} for s in seed_tags}
        frontier_prov: dict[int, set[int]] = dict(provenance)
        visited: set[int] = set(seed_tags)

        for _ in range(depth):
            if not frontier_prov:
                break
            frontier_list = list(frontier_prov.keys())
            if self._tensor_cache:
                all_rows = self._tensor_cache.rows_touching(frontier_list)
                rows_a = [r for r in all_rows
                          if r["tag_a_id"] in frontier_prov and r["tag_b_id"] in search_set]
                rows_b = [r for r in all_rows
                          if r["tag_b_id"] in frontier_prov and r["tag_a_id"] in search_set]
            else:
                f_ph = ",".join("?" * len(frontier_list))
                f_params = tuple(frontier_list)
                rows_a = self._db.execute(
                    f"SELECT tag_a_id, tag_b_id "
                    f"FROM tag_relational_tensor "
                    f"WHERE tag_a_id IN ({f_ph}) AND tag_b_id IN ({sg_ph})",
                    (*f_params, *sg_params),
                ).fetchall()
                rows_b = self._db.execute(
                    f"SELECT tag_a_id, tag_b_id "
                    f"FROM tag_relational_tensor "
                    f"WHERE tag_b_id IN ({f_ph}) AND tag_a_id IN ({sg_ph})",
                    (*f_params, *sg_params),
                ).fetchall()
            next_prov: dict[int, set[int]] = {}
            for r in rows_a:
                src, partner = r["tag_a_id"], r["tag_b_id"]
                seeds_via = frontier_prov[src]
                provenance.setdefault(partner, set()).update(seeds_via)
                if partner not in visited:
                    next_prov.setdefault(partner, set()).update(seeds_via)
            for r in rows_b:
                src, partner = r["tag_b_id"], r["tag_a_id"]
                seeds_via = frontier_prov[src]
                provenance.setdefault(partner, set()).update(seeds_via)
                if partner not in visited:
                    next_prov.setdefault(partner, set()).update(seeds_via)
            visited.update(next_prov.keys())
            frontier_prov = next_prov

        # ── Inward fraction: seed edges / total degree ──
        if self._tensor_cache:
            total_degree = self._tensor_cache.degree(discovered)
            seed_set = set(seed_tags)
            disc_set = set(discovered)
            seed_degree: dict[int, int] = {tid: 0 for tid in discovered}
            for sid in seed_tags:
                for r in self._tensor_cache.rows_for_a({sid}):
                    if r["tag_b_id"] in disc_set:
                        seed_degree[r["tag_b_id"]] += 1
                for r in self._tensor_cache.rows_for_b({sid}):
                    if r["tag_a_id"] in disc_set:
                        seed_degree[r["tag_a_id"]] += 1
        else:
            d_ph = ",".join("?" * len(discovered))
            d_params = tuple(discovered)
            s_ph = ",".join("?" * len(seed_tags))
            s_params = tuple(seed_tags)

            total_degree = {tid: 0 for tid in discovered}
            for col in ("tag_a_id", "tag_b_id"):
                rows = self._db.execute(
                    f"SELECT {col} AS tag, COUNT(*) AS cnt "
                    f"FROM tag_relational_tensor WHERE {col} IN ({d_ph}) "
                    f"GROUP BY {col}",
                    d_params,
                ).fetchall()
                for r in rows:
                    total_degree[r["tag"]] += r["cnt"]

            seed_degree = {tid: 0 for tid in discovered}
            rows_ab = self._db.execute(
                f"SELECT tag_b_id AS tag, COUNT(*) AS cnt "
                f"FROM tag_relational_tensor "
                f"WHERE tag_a_id IN ({s_ph}) AND tag_b_id IN ({d_ph}) "
                f"GROUP BY tag_b_id",
                (*s_params, *d_params),
            ).fetchall()
            rows_ba = self._db.execute(
                f"SELECT tag_a_id AS tag, COUNT(*) AS cnt "
                f"FROM tag_relational_tensor "
                f"WHERE tag_b_id IN ({s_ph}) AND tag_a_id IN ({d_ph}) "
                f"GROUP BY tag_a_id",
                (*s_params, *d_params),
            ).fetchall()
            for r in rows_ab:
                seed_degree[r["tag"]] += r["cnt"]
            for r in rows_ba:
                seed_degree[r["tag"]] += r["cnt"]

        # ── Combine and filter ──
        import logging
        import math
        _log = logging.getLogger("chicory.tag_space")

        n_seeds = len(seed_tags)
        min_ratio = self._config.tag_space_min_inward_ratio * math.sqrt(
            n_seeds / 4
        )
        result = dict(tag_scores)
        for tid in discovered:
            n_converging = len(provenance.get(tid, set()))
            convergence = n_converging / n_seeds if n_seeds > 0 else 0.0
            total = total_degree[tid]
            inward_frac = seed_degree[tid] / total if total > 0 else 0.0
            combined = convergence * inward_frac
            if combined < min_ratio:
                del result[tid]
            else:
                result[tid] = tag_scores[tid] * combined
            if _log.isEnabledFor(logging.DEBUG):
                _log.debug(
                    "tag %d: conv=%d/%d=%.3f inward=%d/%d=%.4f "
                    "combined=%.5f %s",
                    tid, n_converging, n_seeds, convergence,
                    seed_degree[tid], total, inward_frac, combined,
                    f"score=%.4f→%.4f" % (tag_scores[tid], result.get(tid, 0.0))
                    if tid in result else "PRUNED",
                )

        return result

    # ── Context tag filtering ──────────────────────────────────────

    def filter_context_tags(
        self,
        seed_tags: set[int],
        context_tags: set[int],
    ) -> set[int]:
        """Prune hub tags by capping seed edge count.

        Hub tags connect to everything, including many seeds.  Focused
        tags (dead ends) connect to few seeds.  Tags whose seed edge
        count exceeds the configured maximum are pruned.
        """
        if not seed_tags or not context_tags:
            return context_tags

        max_seed = self._config.tag_space_max_context_seed_edges

        candidates = list(context_tags)
        cand_set = set(candidates)
        seed_set = set(seed_tags)

        seed_degree: dict[int, int] = {}
        if self._tensor_cache:
            for sid in seed_tags:
                for r in self._tensor_cache.rows_for_a({sid}):
                    if r["tag_b_id"] in cand_set:
                        tid = r["tag_b_id"]
                        seed_degree[tid] = seed_degree.get(tid, 0) + 1
                for r in self._tensor_cache.rows_for_b({sid}):
                    if r["tag_a_id"] in cand_set:
                        tid = r["tag_a_id"]
                        seed_degree[tid] = seed_degree.get(tid, 0) + 1
        else:
            s_list = list(seed_tags)
            s_ph = ",".join("?" * len(s_list))
            s_params = tuple(s_list)
            c_ph = ",".join("?" * len(candidates))
            c_params = tuple(candidates)
            rows_ab = self._db.execute(
                f"SELECT tag_b_id AS tag, COUNT(*) AS cnt "
                f"FROM tag_relational_tensor "
                f"WHERE tag_a_id IN ({s_ph}) AND tag_b_id IN ({c_ph}) "
                f"GROUP BY tag_b_id",
                (*s_params, *c_params),
            ).fetchall()
            rows_ba = self._db.execute(
                f"SELECT tag_a_id AS tag, COUNT(*) AS cnt "
                f"FROM tag_relational_tensor "
                f"WHERE tag_b_id IN ({s_ph}) AND tag_a_id IN ({c_ph}) "
                f"GROUP BY tag_a_id",
                (*s_params, *c_params),
            ).fetchall()
            for r in rows_ab:
                seed_degree[r["tag"]] = seed_degree.get(r["tag"], 0) + r["cnt"]
            for r in rows_ba:
                seed_degree[r["tag"]] = seed_degree.get(r["tag"], 0) + r["cnt"]

        import logging
        _log = logging.getLogger("chicory.tag_space")

        surviving: set[int] = set()
        for tid in candidates:
            sedges = seed_degree.get(tid, 0)
            if sedges <= max_seed:
                surviving.add(tid)
            elif _log.isEnabledFor(logging.DEBUG):
                _log.debug(
                    "context filter: tag %d seed_edges=%d > %d PRUNED",
                    tid, sedges, max_seed,
                )

        return surviving

    # ── Cache invalidation ──────────────────────────────────────────

    def invalidate_caches(self) -> None:
        self._lexical_index = None
        self._reverse_index = None
        self._exact_name_ids = None
        self._centroid_cache = None
