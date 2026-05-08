"""Cross-project tensor alignment over shared tag anchors."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from chicory.layer3.square_finder import (
    DEFAULT_LAYER_WEIGHTS,
    LAYER_ORDER,
    TensorEdge,
    load_tensor_edges,
)


@dataclass(frozen=True)
class AlignmentCell:
    """One cross-project middle cell induced by a shared anchor."""

    cell_type: str
    anchor: str
    layer_a: str
    layer_b: str
    score: float
    score_a: float
    score_b: float
    detail_a: tuple[str, ...]
    detail_b: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "cell_type": self.cell_type,
            "anchor": self.anchor,
            "layer_a": self.layer_a,
            "layer_b": self.layer_b,
            "score": round(self.score, 6),
            "score_a": round(self.score_a, 6),
            "score_b": round(self.score_b, 6),
            "detail_a": list(self.detail_a),
            "detail_b": list(self.detail_b),
        }


@dataclass(frozen=True)
class CrossProjectAlignmentReport:
    """A 6x6 relation-layer alignment between two Chicory-style DBs."""

    project_a: str
    project_b: str
    edge_count_a: int
    edge_count_b: int
    shared_tag_count: int
    exact_pair_count: int
    neighborhood_matrix: tuple[tuple[str, str, float], ...]
    exact_pair_matrix: tuple[tuple[str, str, float], ...]
    top_cells: tuple[AlignmentCell, ...]
    top_exact_cells: tuple[AlignmentCell, ...]
    parameters: dict[str, Any]

    @property
    def strongest_neighborhood_pair(self) -> tuple[str, str, float] | None:
        if not self.neighborhood_matrix:
            return None
        return max(self.neighborhood_matrix, key=lambda item: item[2])

    @property
    def strongest_exact_pair(self) -> tuple[str, str, float] | None:
        if not self.exact_pair_matrix:
            return None
        return max(self.exact_pair_matrix, key=lambda item: item[2])

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_a": self.project_a,
            "project_b": self.project_b,
            "edge_count_a": self.edge_count_a,
            "edge_count_b": self.edge_count_b,
            "shared_tag_count": self.shared_tag_count,
            "exact_pair_count": self.exact_pair_count,
            "neighborhood_matrix": [
                {"layer_a": a, "layer_b": b, "score": round(score, 6)}
                for a, b, score in self.neighborhood_matrix
            ],
            "exact_pair_matrix": [
                {"layer_a": a, "layer_b": b, "score": round(score, 6)}
                for a, b, score in self.exact_pair_matrix
            ],
            "strongest_neighborhood_pair": self.strongest_neighborhood_pair,
            "strongest_exact_pair": self.strongest_exact_pair,
            "top_cells": [cell.as_dict() for cell in self.top_cells],
            "top_exact_cells": [cell.as_dict() for cell in self.top_exact_cells],
            "parameters": self.parameters,
        }


@dataclass(frozen=True)
class _ProjectGraph:
    edges: list[TensorEdge]
    tag_names: dict[int, str]
    canonical_to_ids: dict[str, tuple[int, ...]]
    incident_profiles: dict[str, dict[str, float]]
    incident_neighbors: dict[tuple[str, str], list[tuple[str, float]]]
    exact_pairs: dict[tuple[str, str], TensorEdge]


def analyze_cross_project_alignment(
    db_a: Any,
    db_b: Any,
    *,
    project_a: str = "project-a",
    project_b: str = "project-b",
    edge_limit: int = 1500,
    per_layer_edges: int = 200,
    min_edge_score: float = 0.05,
    min_tag_length: int = 1,
    anchor_text: str | None = None,
    top_cells: int = 24,
    neighbor_limit: int = 5,
    layer_weights: dict[str, float] | None = None,
) -> CrossProjectAlignmentReport:
    """Align two tensor graphs through shared tag anchors.

    The neighborhood matrix asks: around the same shared tag, which relation
    layers in project A co-occur with which relation layers in project B?  The
    exact-pair matrix is stricter: it only counts identical shared tag pairs.
    """
    graph_a = _load_project_graph(
        db_a,
        edge_limit=edge_limit,
        per_layer_edges=per_layer_edges,
        min_edge_score=min_edge_score,
        min_tag_length=min_tag_length,
        layer_weights=layer_weights,
    )
    graph_b = _load_project_graph(
        db_b,
        edge_limit=edge_limit,
        per_layer_edges=per_layer_edges,
        min_edge_score=min_edge_score,
        min_tag_length=min_tag_length,
        layer_weights=layer_weights,
    )

    shared_tags = sorted(
        set(graph_a.incident_profiles) & set(graph_b.incident_profiles)
    )
    terms = _text_terms(anchor_text)
    if terms:
        shared_tags = [
            tag for tag in shared_tags
            if any(term in tag for term in terms)
        ]

    neighborhood_matrix: dict[tuple[str, str], float] = defaultdict(float)
    cells: list[AlignmentCell] = []
    for anchor in shared_tags:
        profile_a = _normalize_profile(graph_a.incident_profiles[anchor])
        profile_b = _normalize_profile(graph_b.incident_profiles[anchor])
        for layer_a, score_a in profile_a.items():
            for layer_b, score_b in profile_b.items():
                score = score_a * score_b
                neighborhood_matrix[(layer_a, layer_b)] += score
                cells.append(
                    AlignmentCell(
                        cell_type="tag-neighborhood",
                        anchor=anchor,
                        layer_a=layer_a,
                        layer_b=layer_b,
                        score=score,
                        score_a=score_a,
                        score_b=score_b,
                        detail_a=_top_neighbors(
                            graph_a,
                            anchor,
                            layer_a,
                            limit=neighbor_limit,
                        ),
                        detail_b=_top_neighbors(
                            graph_b,
                            anchor,
                            layer_b,
                            limit=neighbor_limit,
                        ),
                    )
                )

    shared_pairs = sorted(set(graph_a.exact_pairs) & set(graph_b.exact_pairs))
    if terms:
        shared_pairs = [
            pair for pair in shared_pairs
            if any(term in pair[0] or term in pair[1] for term in terms)
        ]

    exact_matrix: dict[tuple[str, str], float] = defaultdict(float)
    exact_cells: list[AlignmentCell] = []
    for tag_a, tag_b in shared_pairs:
        edge_a = graph_a.exact_pairs[(tag_a, tag_b)]
        edge_b = graph_b.exact_pairs[(tag_a, tag_b)]
        score = min(edge_a.score, edge_b.score)
        exact_matrix[(edge_a.dominant_layer, edge_b.dominant_layer)] += 1.0
        exact_cells.append(
            AlignmentCell(
                cell_type="exact-edge-pair",
                anchor=f"{tag_a} / {tag_b}",
                layer_a=edge_a.dominant_layer,
                layer_b=edge_b.dominant_layer,
                score=score,
                score_a=edge_a.score,
                score_b=edge_b.score,
                detail_a=(edge_a.tag_a, edge_a.tag_b),
                detail_b=(edge_b.tag_a, edge_b.tag_b),
            )
        )

    cells.sort(key=lambda cell: cell.score, reverse=True)
    exact_cells.sort(key=lambda cell: cell.score, reverse=True)
    return CrossProjectAlignmentReport(
        project_a=project_a,
        project_b=project_b,
        edge_count_a=len(graph_a.edges),
        edge_count_b=len(graph_b.edges),
        shared_tag_count=len(shared_tags),
        exact_pair_count=len(shared_pairs),
        neighborhood_matrix=_matrix_tuple(neighborhood_matrix),
        exact_pair_matrix=_matrix_tuple(exact_matrix),
        top_cells=tuple(cells[:top_cells]),
        top_exact_cells=tuple(exact_cells[:top_cells]),
        parameters={
            "edge_limit": edge_limit,
            "per_layer_edges": per_layer_edges,
            "min_edge_score": min_edge_score,
            "min_tag_length": min_tag_length,
            "anchor_text": anchor_text,
            "top_cells": top_cells,
            "neighbor_limit": neighbor_limit,
        },
    )


def _load_project_graph(
    db: Any,
    *,
    edge_limit: int,
    per_layer_edges: int,
    min_edge_score: float,
    min_tag_length: int,
    layer_weights: dict[str, float] | None,
) -> _ProjectGraph:
    loaded_edges, tag_names = _load_project_edges(db, layer_weights=layer_weights)
    edges = _select_edges(
        loaded_edges,
        edge_limit=edge_limit,
        per_layer_edges=per_layer_edges,
        min_edge_score=min_edge_score,
        min_tag_length=min_tag_length,
    )
    canonical_to_ids: dict[str, list[int]] = defaultdict(list)
    for tag_id, tag_name in tag_names.items():
        canonical = canonical_tag(tag_name)
        if canonical:
            canonical_to_ids[canonical].append(tag_id)

    incident_profiles: dict[str, dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    incident_neighbors: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    exact_pairs: dict[tuple[str, str], TensorEdge] = {}

    for edge in edges:
        tag_a = canonical_tag(edge.tag_a)
        tag_b = canonical_tag(edge.tag_b)
        if not tag_a or not tag_b:
            continue
        incident_profiles[tag_a][edge.dominant_layer] += edge.score
        incident_profiles[tag_b][edge.dominant_layer] += edge.score
        incident_neighbors[(tag_a, edge.dominant_layer)].append((tag_b, edge.score))
        incident_neighbors[(tag_b, edge.dominant_layer)].append((tag_a, edge.score))
        exact_pairs[_pair_key(tag_a, tag_b)] = edge

    for neighbors in incident_neighbors.values():
        neighbors.sort(key=lambda item: item[1], reverse=True)

    return _ProjectGraph(
        edges=edges,
        tag_names=tag_names,
        canonical_to_ids={
            tag: tuple(ids) for tag, ids in canonical_to_ids.items()
        },
        incident_profiles={
            tag: dict(profile) for tag, profile in incident_profiles.items()
        },
        incident_neighbors=dict(incident_neighbors),
        exact_pairs=exact_pairs,
    )


def _load_project_edges(
    db: Any,
    *,
    layer_weights: dict[str, float] | None,
) -> tuple[list[TensorEdge], dict[int, str]]:
    if _has_table(db, "tag_relational_tensor") and _has_table(db, "tags"):
        edges = load_tensor_edges(db, layer_weights=layer_weights)
        tag_names = _load_tag_names(db)
        if edges:
            return edges, tag_names
    if _has_table(db, "glyph_edges"):
        return _load_gu_glyph_edges(db, layer_weights=layer_weights)
    return [], {}


def _load_gu_glyph_edges(
    db: Any,
    *,
    layer_weights: dict[str, float] | None,
) -> tuple[list[TensorEdge], dict[int, str]]:
    weights = dict(DEFAULT_LAYER_WEIGHTS)
    if layer_weights:
        weights.update(layer_weights)
    glyph_names = _gu_glyph_names()
    rows = db.execute(
        """
        SELECT glyph_a, glyph_b, cooccurrence_strength, synchronicity_strength,
               semantic_strength, semiotic_forward, semiotic_reverse,
               ramsey_strength, inhibition_strength, parallelness
        FROM glyph_edges
        """
    ).fetchall()

    name_to_id: dict[str, int] = {}
    tag_names: dict[int, str] = {}

    def tag_id_for(glyph: str) -> tuple[int, str]:
        tag_name = glyph_names.get(glyph, glyph).lower()
        if tag_name not in name_to_id:
            tag_id = len(name_to_id) + 1
            name_to_id[tag_name] = tag_id
            tag_names[tag_id] = tag_name
        return name_to_id[tag_name], tag_name

    edges: list[TensorEdge] = []
    for row in rows:
        row_dict = dict(row)
        tag_a_id, tag_a = tag_id_for(row_dict["glyph_a"])
        tag_b_id, tag_b = tag_id_for(row_dict["glyph_b"])
        strengths = {
            "cooccurrence": _as_float(row_dict.get("cooccurrence_strength")),
            "synchronicity": _as_float(row_dict.get("synchronicity_strength")),
            "semantic": _as_float(row_dict.get("semantic_strength")),
            "semiotic": max(
                _as_float(row_dict.get("semiotic_forward")),
                _as_float(row_dict.get("semiotic_reverse")),
            ),
            "glyph": max(
                _as_float(row_dict.get("ramsey_strength")),
                _as_float(row_dict.get("parallelness")),
            ),
            "inhibition": _as_float(row_dict.get("inhibition_strength")),
        }
        weighted = {
            layer: strengths[layer] * weights.get(layer, 1.0)
            for layer in LAYER_ORDER
        }
        dominant_layer = max(weighted, key=weighted.get)
        score = sum(weighted.values())
        if score <= 0:
            continue
        left_id, right_id = sorted((tag_a_id, tag_b_id))
        left_name = tag_names[left_id]
        right_name = tag_names[right_id]
        edges.append(
            TensorEdge(
                tag_a_id=left_id,
                tag_b_id=right_id,
                tag_a=left_name,
                tag_b=right_name,
                strengths=strengths,
                score=score,
                dominant_layer=dominant_layer,
                memory_ids=(),
            )
        )
    edges.sort(key=lambda edge: edge.score, reverse=True)
    return edges, tag_names


def _select_edges(
    edges: list[TensorEdge],
    *,
    edge_limit: int,
    per_layer_edges: int,
    min_edge_score: float,
    min_tag_length: int,
) -> list[TensorEdge]:
    selected: dict[tuple[int, int], TensorEdge] = {}
    eligible = [
        edge for edge in edges
        if edge.score >= min_edge_score
        and len(canonical_tag(edge.tag_a)) >= min_tag_length
        and len(canonical_tag(edge.tag_b)) >= min_tag_length
    ]
    for edge in eligible[:edge_limit]:
        selected[edge.key] = edge
    if per_layer_edges > 0:
        for layer in LAYER_ORDER:
            count = 0
            for edge in eligible:
                if edge.dominant_layer != layer:
                    continue
                selected[edge.key] = edge
                count += 1
                if count >= per_layer_edges:
                    break
    return sorted(selected.values(), key=lambda edge: edge.score, reverse=True)


def canonical_tag(tag: str) -> str:
    """Normalize project-prefixed tag names to a shared anchor key."""
    value = tag.strip().lower()
    if ":" in value:
        _project, _sep, value = value.partition(":")
    value = value.replace("_", "-")
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"[^a-z0-9\-]", "", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value


def _top_neighbors(
    graph: _ProjectGraph,
    anchor: str,
    layer: str,
    *,
    limit: int = 5,
) -> tuple[str, ...]:
    return tuple(
        f"{neighbor} ({score:.2f})"
        for neighbor, score in graph.incident_neighbors.get((anchor, layer), [])[:limit]
    )


def _matrix_tuple(
    matrix: dict[tuple[str, str], float],
) -> tuple[tuple[str, str, float], ...]:
    values: list[tuple[str, str, float]] = []
    for layer_a in LAYER_ORDER:
        for layer_b in LAYER_ORDER:
            score = matrix.get((layer_a, layer_b), 0.0)
            if score > 0:
                values.append((layer_a, layer_b, score))
    values.sort(key=lambda item: item[2], reverse=True)
    return tuple(values)


def _normalize_profile(profile: dict[str, float]) -> dict[str, float]:
    total = sum(profile.values())
    if total <= 1e-12:
        return {}
    return {
        layer: score / total
        for layer, score in profile.items()
        if score > 0
    }


def _pair_key(tag_a: str, tag_b: str) -> tuple[str, str]:
    return (tag_a, tag_b) if tag_a <= tag_b else (tag_b, tag_a)


def _load_tag_names(db: Any) -> dict[int, str]:
    rows = db.execute("SELECT id, name FROM tags").fetchall()
    return {row["id"]: row["name"] for row in rows}


def _has_table(db: Any, table: str) -> bool:
    row = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _gu_glyph_names() -> dict[str, str]:
    try:
        from chicory.layer1.glyph_lexicon import GLYPH2DICT
    except Exception:
        return {}
    return {glyph: concept for glyph, concept in GLYPH2DICT.items()}


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _text_terms(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    terms = [canonical_tag(part) for part in value.split(",")]
    return tuple(term for term in terms if term)
