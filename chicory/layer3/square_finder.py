"""Square motif discovery over Chicory's tag relational tensor.

The square finder treats the tensor as a multiplex graph: every tag pair has a
small relation vector instead of a single edge weight.  A square motif is a
four-cycle whose sides are strong tensor edges, whose incident side colors are
optionally distinct, and whose two-hop paths imply one or both missing
diagonals.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


LAYER_ORDER = (
    "cooccurrence",
    "synchronicity",
    "semantic",
    "semiotic",
    "glyph",
    "inhibition",
)

DEFAULT_LAYER_WEIGHTS = {
    "cooccurrence": 0.5,
    "synchronicity": 0.3,
    "semantic": 0.2,
    "semiotic": 0.15,
    "glyph": 0.2,
    "inhibition": 0.2,
}

_LAYER_COLUMNS = {
    "cooccurrence": ("cooccurrence_strength",),
    "synchronicity": ("synchronicity_strength",),
    "semantic": ("semantic_strength",),
    "semiotic": ("semiotic_forward", "semiotic_reverse"),
    "glyph": ("glyph_strength",),
    "inhibition": ("inhibition_strength",),
}


@dataclass(frozen=True)
class TensorEdge:
    """One tag-pair tensor row reduced to relation-vector form."""

    tag_a_id: int
    tag_b_id: int
    tag_a: str
    tag_b: str
    strengths: dict[str, float]
    score: float
    dominant_layer: str
    memory_ids: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[int, int]:
        return edge_key(self.tag_a_id, self.tag_b_id)

    @property
    def vector(self) -> tuple[float, ...]:
        return tuple(self.strengths.get(layer, 0.0) for layer in LAYER_ORDER)

    def as_dict(self) -> dict[str, Any]:
        return {
            "tag_a_id": self.tag_a_id,
            "tag_b_id": self.tag_b_id,
            "tag_a": self.tag_a,
            "tag_b": self.tag_b,
            "strengths": self.strengths,
            "score": round(self.score, 6),
            "dominant_layer": self.dominant_layer,
            "memory_count": len(self.memory_ids),
            "memory_sample": list(self.memory_ids[:5]),
        }


@dataclass(frozen=True)
class DiagonalSignal:
    """Evidence for a square diagonal, drawn or implied."""

    endpoints: tuple[str, str]
    status: str
    existing_score: float
    existing_layer: str | None
    support: float
    gap: float
    path_consistency: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "endpoints": list(self.endpoints),
            "status": self.status,
            "existing_score": round(self.existing_score, 6),
            "existing_layer": self.existing_layer,
            "support": round(self.support, 6),
            "gap": round(self.gap, 6),
            "path_consistency": round(self.path_consistency, 6),
        }


@dataclass(frozen=True)
class SquareMotif:
    """A four-cycle plus diagonal and center-like consistency signals."""

    tag_ids: tuple[int, int, int, int]
    tags: tuple[str, str, str, str]
    side_edges: tuple[TensorEdge, TensorEdge, TensorEdge, TensorEdge]
    ac_diagonal: DiagonalSignal
    bd_diagonal: DiagonalSignal
    side_score: float
    color_diversity: int
    repeated_color_vertices: tuple[str, ...]
    center_score: float
    void_score: float
    shared_memory_count: int
    source_summaries: tuple[tuple[str, int], ...]
    interestingness: float

    @property
    def side_layers(self) -> tuple[str, str, str, str]:
        return tuple(edge.dominant_layer for edge in self.side_edges)  # type: ignore[return-value]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tags": list(self.tags),
            "tag_ids": list(self.tag_ids),
            "side_layers": list(self.side_layers),
            "side_score": round(self.side_score, 6),
            "color_diversity": self.color_diversity,
            "repeated_color_vertices": list(self.repeated_color_vertices),
            "center_score": round(self.center_score, 6),
            "void_score": round(self.void_score, 6),
            "shared_memory_count": self.shared_memory_count,
            "source_summaries": [
                {"summary": summary, "count": count}
                for summary, count in self.source_summaries
            ],
            "interestingness": round(self.interestingness, 6),
            "ac_diagonal": self.ac_diagonal.as_dict(),
            "bd_diagonal": self.bd_diagonal.as_dict(),
            "side_edges": [edge.as_dict() for edge in self.side_edges],
        }


def edge_key(a: int, b: int) -> tuple[int, int]:
    """Canonical undirected tag-pair key."""
    return (a, b) if a < b else (b, a)


def load_tensor_edges(
    db: Any,
    *,
    layer_weights: dict[str, float] | None = None,
) -> list[TensorEdge]:
    """Load tensor rows as scored multiplex edges."""
    weights = dict(DEFAULT_LAYER_WEIGHTS)
    if layer_weights:
        weights.update(layer_weights)

    tensor_cols = {
        row["name"]
        for row in db.execute("PRAGMA table_info(tag_relational_tensor)").fetchall()
    }

    select_cols = [
        "trt.tag_a_id",
        "trt.tag_b_id",
        "ta.name AS tag_a",
        "tb.name AS tag_b",
    ]
    if "memory_ids" in tensor_cols:
        select_cols.append("trt.memory_ids")
    for cols in _LAYER_COLUMNS.values():
        for col in cols:
            if col in tensor_cols:
                select_cols.append(f"trt.{col}")

    rows = db.execute(
        f"""
        SELECT {', '.join(select_cols)}
        FROM tag_relational_tensor trt
        JOIN tags ta ON ta.id = trt.tag_a_id
        JOIN tags tb ON tb.id = trt.tag_b_id
        """
    ).fetchall()

    edges: list[TensorEdge] = []
    for row in rows:
        row_dict = dict(row)
        strengths: dict[str, float] = {}
        for layer, cols in _LAYER_COLUMNS.items():
            values = [
                _as_float(row_dict.get(col))
                for col in cols
                if col in row_dict
            ]
            strengths[layer] = max(values) if values else 0.0

        weighted = {
            layer: strengths[layer] * weights.get(layer, 1.0)
            for layer in LAYER_ORDER
        }
        dominant_layer = max(weighted, key=weighted.get)
        score = sum(weighted.values())
        if score <= 0:
            continue

        edges.append(
            TensorEdge(
                tag_a_id=row["tag_a_id"],
                tag_b_id=row["tag_b_id"],
                tag_a=row["tag_a"],
                tag_b=row["tag_b"],
                strengths=strengths,
                score=score,
                dominant_layer=dominant_layer,
                memory_ids=_parse_memory_ids(row_dict.get("memory_ids")),
            )
        )

    edges.sort(key=lambda e: e.score, reverse=True)
    return edges


def find_square_motifs(
    db: Any,
    *,
    tag: str | None = None,
    tag_id: int | None = None,
    min_edge_score: float = 0.05,
    min_diagonal_score: float | None = None,
    min_path_consistency: float = 0.65,
    min_colors: int = 2,
    max_edges: int = 500,
    per_layer_edges: int = 100,
    min_tag_length: int = 2,
    require_void: bool = False,
    include_text: str | None = None,
    exclude_text: str | None = None,
    limit: int = 20,
    require_distinct_incident_layers: bool = True,
    layer_weights: dict[str, float] | None = None,
) -> list[SquareMotif]:
    """Find high-signal four-cycle motifs in the tensor graph.

    The candidate cycle uses nontrivial tags of at least ``min_tag_length``
    characters, taking the strongest ``max_edges`` above
    ``min_edge_score`` plus up to ``per_layer_edges`` from each dominant layer.
    This keeps quieter layers visible when synchronicity or glyph edges dominate
    the global ranking.  Diagonal lookup still uses every loaded tensor edge so
    weak-but-present diagonals are not mistaken for absent ones.
    """
    if min_diagonal_score is None:
        min_diagonal_score = min_edge_score

    all_edges = load_tensor_edges(db, layer_weights=layer_weights)
    edge_lookup = {edge.key: edge for edge in all_edges}

    selected_by_key: dict[tuple[int, int], TensorEdge] = {}
    scored_edges = [
        edge for edge in all_edges
        if edge.score >= min_edge_score
        and len(edge.tag_a) >= min_tag_length
        and len(edge.tag_b) >= min_tag_length
    ]
    for edge in scored_edges[:max_edges]:
        selected_by_key[edge.key] = edge
    if per_layer_edges > 0:
        for layer in LAYER_ORDER:
            layer_count = 0
            for edge in scored_edges:
                if edge.dominant_layer != layer:
                    continue
                selected_by_key[edge.key] = edge
                layer_count += 1
                if layer_count >= per_layer_edges:
                    break

    selected_edges = sorted(
        selected_by_key.values(), key=lambda edge: edge.score, reverse=True
    )
    if len(selected_edges) < 4:
        return []

    name_lookup = _load_tag_names(db)
    selected_tag_id = tag_id
    if tag and selected_tag_id is None:
        selected_tag_id = _find_tag_id(name_lookup, tag)
        if selected_tag_id is None:
            return []

    adjacency: dict[int, set[int]] = defaultdict(set)
    for edge in selected_edges:
        adjacency[edge.tag_a_id].add(edge.tag_b_id)
        adjacency[edge.tag_b_id].add(edge.tag_a_id)

    cycles: set[tuple[int, int, int, int]] = set()
    for a in sorted(adjacency):
        for b in sorted(adjacency[a]):
            for c in sorted(adjacency[b]):
                if c in (a, b):
                    continue
                for d in sorted(adjacency[a]):
                    if d in (a, b, c):
                        continue
                    if c not in adjacency[d]:
                        continue
                    cycle = (a, b, c, d)
                    if selected_tag_id is not None and selected_tag_id not in cycle:
                        continue
                    cycles.add(_canonical_cycle(cycle))

    motifs: list[SquareMotif] = []
    for cycle in cycles:
        sides = (
            edge_lookup[edge_key(cycle[0], cycle[1])],
            edge_lookup[edge_key(cycle[1], cycle[2])],
            edge_lookup[edge_key(cycle[2], cycle[3])],
            edge_lookup[edge_key(cycle[3], cycle[0])],
        )
        side_layers = tuple(edge.dominant_layer for edge in sides)
        color_diversity = len(set(side_layers))
        if color_diversity < min_colors:
            continue

        repeats = _repeated_color_vertices(cycle, side_layers, name_lookup)
        if require_distinct_incident_layers and repeats:
            continue

        tags = tuple(name_lookup.get(tid, str(tid)) for tid in cycle)
        ac_signal = _diagonal_signal(
            cycle[0],
            cycle[2],
            (sides[0], sides[1]),
            (sides[3], sides[2]),
            edge_lookup,
            name_lookup,
            min_diagonal_score=min_diagonal_score,
            min_path_consistency=min_path_consistency,
        )
        bd_signal = _diagonal_signal(
            cycle[1],
            cycle[3],
            (sides[0], sides[3]),
            (sides[1], sides[2]),
            edge_lookup,
            name_lookup,
            min_diagonal_score=min_diagonal_score,
            min_path_consistency=min_path_consistency,
        )

        side_score = sum(edge.score for edge in sides) / 4.0
        center_score = (
            ac_signal.path_consistency + bd_signal.path_consistency
        ) / 2.0
        void_score = sum(
            signal.gap
            for signal in (ac_signal, bd_signal)
            if signal.status == "void"
        )
        if require_void and void_score <= 0:
            continue

        shared_memory_count = _shared_memory_count(sides)
        source_summaries = _source_summaries_for_tags(db, cycle, name_lookup)
        diversity_bonus = color_diversity / len(LAYER_ORDER)
        drawn_penalty = 0.1 * sum(
            1 for signal in (ac_signal, bd_signal) if signal.status == "drawn"
        )
        interestingness = (
            side_score * (1.0 + diversity_bonus)
            + void_score
            + 0.5 * center_score
            + 0.05 * math.log1p(shared_memory_count)
            - drawn_penalty
        )

        motif = SquareMotif(
            tag_ids=cycle,
            tags=tags,  # type: ignore[arg-type]
            side_edges=sides,
            ac_diagonal=ac_signal,
            bd_diagonal=bd_signal,
            side_score=side_score,
            color_diversity=color_diversity,
            repeated_color_vertices=repeats,
            center_score=center_score,
            void_score=void_score,
            shared_memory_count=shared_memory_count,
            source_summaries=source_summaries,
            interestingness=interestingness,
        )
        if not _motif_matches_text_filters(motif, include_text, exclude_text):
            continue

        motifs.append(motif)

    motifs.sort(key=lambda motif: motif.interestingness, reverse=True)
    return motifs[:limit]


def _diagonal_signal(
    a: int,
    c: int,
    path_one: tuple[TensorEdge, TensorEdge],
    path_two: tuple[TensorEdge, TensorEdge],
    edge_lookup: dict[tuple[int, int], TensorEdge],
    name_lookup: dict[int, str],
    *,
    min_diagonal_score: float,
    min_path_consistency: float,
) -> DiagonalSignal:
    path_one_vec = _path_vector(path_one)
    path_two_vec = _path_vector(path_two)
    consistency = _cosine(path_one_vec, path_two_vec)
    path_one_strength = sum(edge.score for edge in path_one) / 2.0
    path_two_strength = sum(edge.score for edge in path_two) / 2.0
    support = min(path_one_strength, path_two_strength) * consistency

    existing = edge_lookup.get(edge_key(a, c))
    existing_score = existing.score if existing else 0.0
    existing_layer = existing.dominant_layer if existing else None
    gap = max(0.0, support - existing_score)

    if existing_score >= min_diagonal_score:
        status = "drawn"
    elif support >= min_diagonal_score and consistency >= min_path_consistency:
        status = "void"
    elif support > 0:
        status = "implied"
    else:
        status = "missing"

    return DiagonalSignal(
        endpoints=(name_lookup.get(a, str(a)), name_lookup.get(c, str(c))),
        status=status,
        existing_score=existing_score,
        existing_layer=existing_layer,
        support=support,
        gap=gap,
        path_consistency=consistency,
    )


def _path_vector(edges: tuple[TensorEdge, TensorEdge]) -> tuple[float, ...]:
    v1 = _unit(edges[0].vector)
    v2 = _unit(edges[1].vector)
    return tuple(a + b for a, b in zip(v1, v2))


def _unit(values: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(v * v for v in values))
    if norm <= 1e-12:
        return tuple(0.0 for _ in values)
    return tuple(v / norm for v in values)


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if denom <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, sum(x * y for x, y in zip(a, b)) / denom))


def _canonical_cycle(cycle: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    variants: list[tuple[int, int, int, int]] = []
    forward = cycle
    reverse = tuple(reversed(cycle))
    for seq in (forward, reverse):
        for i in range(4):
            variants.append(seq[i:] + seq[:i])  # type: ignore[operator]
    return min(variants)


def _repeated_color_vertices(
    cycle: tuple[int, int, int, int],
    layers: tuple[str, str, str, str],
    name_lookup: dict[int, str],
) -> tuple[str, ...]:
    incident = (
        (cycle[0], layers[3], layers[0]),
        (cycle[1], layers[0], layers[1]),
        (cycle[2], layers[1], layers[2]),
        (cycle[3], layers[2], layers[3]),
    )
    return tuple(
        name_lookup.get(tag_id, str(tag_id))
        for tag_id, left, right in incident
        if left == right
    )


def _shared_memory_count(edges: tuple[TensorEdge, ...]) -> int:
    memory_sets = [set(edge.memory_ids) for edge in edges if edge.memory_ids]
    if len(memory_sets) < 2:
        return 0
    shared = set(memory_sets[0])
    for memories in memory_sets[1:]:
        shared &= memories
    return len(shared)


def _source_summaries_for_tags(
    db: Any,
    tag_ids: tuple[int, int, int, int],
    name_lookup: dict[int, str],
    *,
    limit: int = 5,
) -> tuple[tuple[str, int], ...]:
    """Return dominant memory summaries for non-letter motif tags.

    Single-letter tags are intentionally ignored here: they touch a huge share of
    the corpus and swamp source attribution.  The query prefers memories tagged
    by at least two non-letter motif tags, then falls back to one.
    """
    non_letter_ids = [
        tag_id for tag_id in tag_ids
        if len(name_lookup.get(tag_id, "")) > 1
    ]
    if not non_letter_ids:
        return ()

    for threshold in (min(2, len(non_letter_ids)), 1):
        summaries = _query_source_summaries(db, non_letter_ids, threshold, limit)
        if summaries:
            return summaries
    return ()


def _query_source_summaries(
    db: Any,
    tag_ids: list[int],
    threshold: int,
    limit: int,
) -> tuple[tuple[str, int], ...]:
    placeholders = ",".join("?" for _ in tag_ids)
    rows = db.execute(
        f"""
        SELECT COALESCE(NULLIF(m.summary, ''), '(no summary)') AS summary,
               COUNT(*) AS cnt
        FROM (
            SELECT memory_id
            FROM memory_tags
            WHERE tag_id IN ({placeholders})
            GROUP BY memory_id
            HAVING COUNT(DISTINCT tag_id) >= ?
        ) matched
        JOIN memories m ON m.id = matched.memory_id
        GROUP BY summary
        ORDER BY cnt DESC, summary ASC
        LIMIT ?
        """,
        tuple(tag_ids) + (threshold, limit),
    ).fetchall()
    return tuple((row["summary"], row["cnt"]) for row in rows)


def _motif_matches_text_filters(
    motif: SquareMotif,
    include_text: str | None,
    exclude_text: str | None,
) -> bool:
    haystack = " ".join(
        [
            *motif.tags,
            *(summary for summary, _count in motif.source_summaries),
        ]
    ).lower()
    include_terms = _text_terms(include_text)
    exclude_terms = _text_terms(exclude_text)
    if include_terms and not any(term in haystack for term in include_terms):
        return False
    if exclude_terms and any(term in haystack for term in exclude_terms):
        return False
    return True


def _text_terms(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    terms = [part.strip().lower() for part in value.split(",")]
    return tuple(term for term in terms if term)


def _parse_memory_ids(raw: Any) -> tuple[str, ...]:
    if not raw:
        return ()
    try:
        values = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return ()
    if not isinstance(values, list):
        return ()
    return tuple(str(value) for value in values)


def _load_tag_names(db: Any) -> dict[int, str]:
    rows = db.execute("SELECT id, name FROM tags").fetchall()
    return {row["id"]: row["name"] for row in rows}


def _find_tag_id(name_lookup: dict[int, str], tag: str) -> int | None:
    normalized = tag.strip().lower().replace(" ", "-")
    for tag_id, name in name_lookup.items():
        if name == normalized:
            return tag_id
    return None


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)
