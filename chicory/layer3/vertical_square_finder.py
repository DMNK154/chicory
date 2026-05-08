"""Source-by-tag vertical square discovery.

Horizontal square motifs live entirely inside the tag tensor.  Vertical squares
look across provenance instead: two source files form rows, two tags form
columns, and all four source/tag incidences are present.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from chicory.layer3.square_finder import TensorEdge, edge_key, load_tensor_edges


@dataclass(frozen=True)
class VerticalSquareMotif:
    """A 2x2 source/tag rectangle, optionally backed by a tensor edge."""

    sources: tuple[str, str]
    tags: tuple[str, str]
    tag_ids: tuple[int, int]
    counts: tuple[int, int, int, int]
    relation_layer: str | None
    relation_score: float
    relation_status: str
    source_balance: float
    repetition_score: float
    interestingness: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "sources": list(self.sources),
            "tags": list(self.tags),
            "tag_ids": list(self.tag_ids),
            "counts": {
                f"{self.sources[0]}::{self.tags[0]}": self.counts[0],
                f"{self.sources[0]}::{self.tags[1]}": self.counts[1],
                f"{self.sources[1]}::{self.tags[0]}": self.counts[2],
                f"{self.sources[1]}::{self.tags[1]}": self.counts[3],
            },
            "relation_layer": self.relation_layer,
            "relation_score": round(self.relation_score, 6),
            "relation_status": self.relation_status,
            "source_balance": round(self.source_balance, 6),
            "repetition_score": round(self.repetition_score, 6),
            "interestingness": round(self.interestingness, 6),
        }


@dataclass(frozen=True)
class OrientationSignature:
    """Color pattern analysis under the eight symmetries of a square."""

    orientation_class: str
    diagonal_bias: str
    color_signature: str
    symmetry_variants: int


@dataclass(frozen=True)
class TensorVerticalSquareMotif:
    """A 2x2 tensor rectangle with source/file tags as rows."""

    sources: tuple[str, str]
    row_tags: tuple[str, str]
    column_tags: tuple[str, str]
    row_tag_ids: tuple[int, int]
    column_tag_ids: tuple[int, int]
    cell_layers: tuple[str, str, str, str]
    cell_scores: tuple[float, float, float, float]
    orientation_class: str
    diagonal_bias: str
    color_signature: str
    symmetry_variants: int
    column_relation_layer: str | None
    column_relation_score: float
    column_relation_status: str
    row_relation_layer: str | None
    row_relation_score: float
    interestingness: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "sources": list(self.sources),
            "row_tags": list(self.row_tags),
            "column_tags": list(self.column_tags),
            "row_tag_ids": list(self.row_tag_ids),
            "column_tag_ids": list(self.column_tag_ids),
            "cell_layers": list(self.cell_layers),
            "cell_scores": [round(score, 6) for score in self.cell_scores],
            "orientation_class": self.orientation_class,
            "diagonal_bias": self.diagonal_bias,
            "color_signature": self.color_signature,
            "symmetry_variants": self.symmetry_variants,
            "column_relation_layer": self.column_relation_layer,
            "column_relation_score": round(self.column_relation_score, 6),
            "column_relation_status": self.column_relation_status,
            "row_relation_layer": self.row_relation_layer,
            "row_relation_score": round(self.row_relation_score, 6),
            "interestingness": round(self.interestingness, 6),
        }


def find_tensor_vertical_square_motifs(
    db: Any,
    *,
    limit: int = 20,
    min_edge_score: float = 0.0,
    require_letter: bool = True,
    tag_text: str | None = "sequence",
    exact_tag_text: bool = True,
    include_source: str | None = None,
    exclude_source: str | None = None,
    include_text: str | None = None,
    exclude_text: str | None = None,
    layer_weights: dict[str, float] | None = None,
) -> list[TensorVerticalSquareMotif]:
    """Find tensor rectangles using source-derived file tags as rows."""
    tag_names = _load_tag_names(db)
    source_rows = _load_source_file_tag_rows(
        db,
        tag_names,
        include_source=include_source,
        exclude_source=exclude_source,
    )
    if len(source_rows) < 2:
        return []

    edges = load_tensor_edges(db, layer_weights=layer_weights)
    edge_lookup = {edge.key: edge for edge in edges}

    column_ids = [
        tag_id for tag_id, tag in tag_names.items()
        if _tag_allowed(tag, 1, require_letter, tag_text, exact_tag_text)
    ]
    column_ids.sort(key=lambda tag_id: tag_names[tag_id])

    column_pairs: list[tuple[int, int]] = []
    for i in range(len(column_ids)):
        for j in range(i + 1, len(column_ids)):
            tag_a = tag_names[column_ids[i]]
            tag_b = tag_names[column_ids[j]]
            if _pair_allowed(tag_a, tag_b, require_letter, tag_text, exact_tag_text):
                column_pairs.append((column_ids[i], column_ids[j]))

    motifs: list[TensorVerticalSquareMotif] = []
    for left_idx in range(len(source_rows) - 1):
        source_a, row_a = source_rows[left_idx]
        for right_idx in range(left_idx + 1, len(source_rows)):
            source_b, row_b = source_rows[right_idx]
            for col_a, col_b in column_pairs:
                cells = (
                    edge_lookup.get(edge_key(row_a, col_a)),
                    edge_lookup.get(edge_key(row_a, col_b)),
                    edge_lookup.get(edge_key(row_b, col_a)),
                    edge_lookup.get(edge_key(row_b, col_b)),
                )
                if any(cell is None for cell in cells):
                    continue
                typed_cells = tuple(cell for cell in cells if cell is not None)
                if any(cell.score < min_edge_score for cell in typed_cells):
                    continue

                column_edge = edge_lookup.get(edge_key(col_a, col_b))
                row_edge = edge_lookup.get(edge_key(row_a, row_b))
                cell_scores = (
                    typed_cells[0].score,
                    typed_cells[1].score,
                    typed_cells[2].score,
                    typed_cells[3].score,
                )
                cell_layers = (
                    typed_cells[0].dominant_layer,
                    typed_cells[1].dominant_layer,
                    typed_cells[2].dominant_layer,
                    typed_cells[3].dominant_layer,
                )
                orientation = analyze_cell_layer_orientation(cell_layers)
                column_relation_score = column_edge.score if column_edge else 0.0
                row_relation_score = row_edge.score if row_edge else 0.0
                column_relation_status = "drawn" if column_edge else "source-repeated"
                layer_diversity = len(set(cell_layers)) / 4.0
                balance = min(cell_scores) / max(cell_scores)
                letter_sequence_bonus = (
                    1.0
                    if _is_letter_sequence_pair(tag_names[col_a], tag_names[col_b])
                    else 0.0
                )
                interestingness = (
                    sum(cell_scores) / 4.0
                    + column_relation_score
                    + 0.25 * row_relation_score
                    + balance
                    + layer_diversity
                    + letter_sequence_bonus
                )

                motif = TensorVerticalSquareMotif(
                    sources=(source_a, source_b),
                    row_tags=(tag_names[row_a], tag_names[row_b]),
                    column_tags=(tag_names[col_a], tag_names[col_b]),
                    row_tag_ids=(row_a, row_b),
                    column_tag_ids=(col_a, col_b),
                    cell_layers=cell_layers,
                    cell_scores=cell_scores,
                    orientation_class=orientation.orientation_class,
                    diagonal_bias=orientation.diagonal_bias,
                    color_signature=orientation.color_signature,
                    symmetry_variants=orientation.symmetry_variants,
                    column_relation_layer=(
                        column_edge.dominant_layer if column_edge else None
                    ),
                    column_relation_score=column_relation_score,
                    column_relation_status=column_relation_status,
                    row_relation_layer=row_edge.dominant_layer if row_edge else None,
                    row_relation_score=row_relation_score,
                    interestingness=interestingness,
                )
                if not _tensor_motif_matches_text_filters(
                    motif, include_text, exclude_text
                ):
                    continue
                motifs.append(motif)

    motifs.sort(key=lambda motif: motif.interestingness, reverse=True)
    return motifs[:limit]


def find_vertical_square_motifs(
    db: Any,
    *,
    limit: int = 20,
    min_count: int = 1,
    min_tag_length: int = 1,
    require_letter: bool = True,
    tag_text: str | None = "sequence",
    exact_tag_text: bool = True,
    include_source: str | None = None,
    exclude_source: str | None = None,
    include_text: str | None = None,
    exclude_text: str | None = None,
    min_relation_score: float = 0.0,
    max_tags_per_source: int = 80,
    layer_weights: dict[str, float] | None = None,
) -> list[VerticalSquareMotif]:
    """Find source-by-tag rectangles.

    The default is tuned for the glyph/letter discovery path: one column must be
    a single-letter tag and the other must include "sequence".  Set
    ``require_letter=False`` or ``tag_text=None`` to broaden the scan.
    """
    source_tag_counts = _load_source_tag_counts(
        db,
        min_count=min_count,
        min_tag_length=min_tag_length,
        include_source=include_source,
        exclude_source=exclude_source,
    )
    if not source_tag_counts:
        return []

    tag_names = _load_tag_names(db)
    edge_lookup = {
        edge.key: edge
        for edge in load_tensor_edges(db, layer_weights=layer_weights)
    }

    source_tags: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for (source, tag_id), count in source_tag_counts.items():
        tag = tag_names.get(tag_id, "")
        if not _tag_allowed(
            tag, min_tag_length, require_letter, tag_text, exact_tag_text
        ):
            continue
        source_tags[source].append((tag_id, count))

    for source, tags in list(source_tags.items()):
        tags.sort(key=lambda item: item[1], reverse=True)
        source_tags[source] = tags[:max_tags_per_source]
        if len(source_tags[source]) < 2:
            del source_tags[source]

    pair_sources: dict[
        tuple[int, int], list[tuple[str, int, int]]
    ] = defaultdict(list)

    for source, tags in source_tags.items():
        for i in range(len(tags)):
            tag_a, count_a = tags[i]
            for j in range(i + 1, len(tags)):
                tag_b, count_b = tags[j]
                name_a = tag_names.get(tag_a, "")
                name_b = tag_names.get(tag_b, "")
                if not _pair_allowed(
                    name_a, name_b, require_letter, tag_text, exact_tag_text
                ):
                    continue
                key = edge_key(tag_a, tag_b)
                pair_sources[key].append((source, count_a, count_b))

    motifs: list[VerticalSquareMotif] = []
    for (tag_a, tag_b), sources in pair_sources.items():
        if len(sources) < 2:
            continue

        sources.sort(key=lambda item: min(item[1], item[2]), reverse=True)
        for left_idx in range(len(sources) - 1):
            for right_idx in range(left_idx + 1, len(sources)):
                source_a, a_tag_a, a_tag_b = sources[left_idx]
                source_b, b_tag_a, b_tag_b = sources[right_idx]
                counts = (a_tag_a, a_tag_b, b_tag_a, b_tag_b)
                edge = edge_lookup.get((tag_a, tag_b))
                relation_score = edge.score if edge else 0.0
                if relation_score < min_relation_score:
                    continue
                relation_layer = edge.dominant_layer if edge else None
                relation_status = "drawn" if edge else "source-repeated"
                repetition_score = math.log1p(sum(counts))
                source_balance = min(counts) / max(counts)
                letter_sequence_bonus = (
                    1.0
                    if _is_letter_sequence_pair(
                        tag_names.get(tag_a, ""), tag_names.get(tag_b, "")
                    )
                    else 0.0
                )
                interestingness = (
                    repetition_score
                    + source_balance
                    + relation_score
                    + letter_sequence_bonus
                )

                motif = VerticalSquareMotif(
                    sources=(source_a, source_b),
                    tags=(tag_names.get(tag_a, str(tag_a)), tag_names.get(tag_b, str(tag_b))),
                    tag_ids=(tag_a, tag_b),
                    counts=counts,
                    relation_layer=relation_layer,
                    relation_score=relation_score,
                    relation_status=relation_status,
                    source_balance=source_balance,
                    repetition_score=repetition_score,
                    interestingness=interestingness,
                )
                if not _motif_matches_text_filters(motif, include_text, exclude_text):
                    continue
                motifs.append(motif)

    motifs.sort(key=lambda motif: motif.interestingness, reverse=True)
    return motifs[:limit]


def normalize_source_summary(summary: str | None) -> str:
    """Collapse chunk summaries back to a stable source/file key."""
    if not summary:
        return "(unknown source)"
    source = summary.strip()
    source = re.sub(r"\s+\[\d+/\d+\]\s*$", "", source)
    source = re.sub(r"\s+\(part\s+\d+/\d+\)\s*$", "", source, flags=re.I)
    source = re.sub(r"\s+", " ", source)
    return source or "(unknown source)"


def source_summary_to_tag_name(summary: str | None) -> str:
    """Normalize a source summary the same way file-derived tags are named."""
    source = normalize_source_summary(summary)
    source = re.sub(r"\.[A-Za-z0-9]+$", "", source)
    source = source.strip().lower()
    source = re.sub(r"\s+", "-", source)
    source = re.sub(r"_+", "-", source)
    source = re.sub(r"[^a-z0-9\-:]", "", source)
    source = re.sub(r"-+", "-", source).strip("-")
    return source


def analyze_cell_layer_orientation(
    cell_layers: tuple[str, str, str, str],
) -> OrientationSignature:
    """Classify a 2x2 layer coloring under D4 square symmetries.

    Cells are ordered row-major: top-left, top-right, bottom-left, bottom-right.
    The canonical signature is the lexicographically smallest transformed grid
    over rotations and reflections, so equivalent colorings get the same key.
    """
    a, b, c, d = cell_layers
    transformed = tuple(
        _format_layer_grid(_transform_layers(cell_layers, order))
        for order in _D4_ORDERS
    )
    unique_forms = set(transformed)
    main_same = a == d
    anti_same = b == c

    if main_same and anti_same and a != b:
        orientation_class = "checkerboard-diagonal"
    elif len({a, b, c, d}) == 1:
        orientation_class = "uniform"
    elif a == b and c == d and a != c:
        orientation_class = "row-banded"
    elif a == c and b == d and a != b:
        orientation_class = "column-banded"
    elif main_same and not anti_same:
        orientation_class = "main-diagonal-biased"
    elif anti_same and not main_same:
        orientation_class = "anti-diagonal-biased"
    else:
        orientation_class = "mixed"

    if main_same and anti_same and len({a, b, c, d}) > 1:
        diagonal_bias = "both"
    elif main_same and not anti_same:
        diagonal_bias = "main"
    elif anti_same and not main_same:
        diagonal_bias = "anti"
    else:
        diagonal_bias = "none"

    return OrientationSignature(
        orientation_class=orientation_class,
        diagonal_bias=diagonal_bias,
        color_signature=min(unique_forms),
        symmetry_variants=len(unique_forms),
    )


_D4_ORDERS: tuple[tuple[int, int, int, int], ...] = (
    (0, 1, 2, 3),  # identity
    (2, 0, 3, 1),  # rotate 90
    (3, 2, 1, 0),  # rotate 180
    (1, 3, 0, 2),  # rotate 270
    (2, 3, 0, 1),  # reflect horizontal
    (1, 0, 3, 2),  # reflect vertical
    (0, 2, 1, 3),  # reflect main diagonal
    (3, 1, 2, 0),  # reflect anti-diagonal
)


def _transform_layers(
    cell_layers: tuple[str, str, str, str],
    order: tuple[int, int, int, int],
) -> tuple[str, str, str, str]:
    return (
        cell_layers[order[0]],
        cell_layers[order[1]],
        cell_layers[order[2]],
        cell_layers[order[3]],
    )


def _format_layer_grid(cell_layers: tuple[str, str, str, str]) -> str:
    return f"{cell_layers[0]}|{cell_layers[1]}/{cell_layers[2]}|{cell_layers[3]}"


def _load_source_tag_counts(
    db: Any,
    *,
    min_count: int,
    min_tag_length: int,
    include_source: str | None,
    exclude_source: str | None,
) -> dict[tuple[str, int], int]:
    rows = db.execute(
        """
        SELECT m.summary AS summary, mt.tag_id AS tag_id, t.name AS tag_name,
               COUNT(DISTINCT m.id) AS cnt
        FROM memories m
        JOIN memory_tags mt ON mt.memory_id = m.id
        JOIN tags t ON t.id = mt.tag_id
        WHERE m.is_archived = 0
          AND m.summary IS NOT NULL
        GROUP BY m.summary, mt.tag_id, t.name
        """
    ).fetchall()

    counts: dict[tuple[str, int], int] = defaultdict(int)
    include_terms = _text_terms(include_source)
    exclude_terms = _text_terms(exclude_source)
    for row in rows:
        tag_name = row["tag_name"]
        if len(tag_name) < min_tag_length:
            continue
        source = normalize_source_summary(row["summary"])
        source_l = source.lower()
        if include_terms and not any(term in source_l for term in include_terms):
            continue
        if exclude_terms and any(term in source_l for term in exclude_terms):
            continue
        counts[(source, row["tag_id"])] += row["cnt"]

    return {
        key: count
        for key, count in counts.items()
        if count >= min_count
    }


def _load_source_file_tag_rows(
    db: Any,
    tag_names: dict[int, str],
    *,
    include_source: str | None,
    exclude_source: str | None,
) -> list[tuple[str, int]]:
    name_to_id = {name: tag_id for tag_id, name in tag_names.items()}
    include_terms = _text_terms(include_source)
    exclude_terms = _text_terms(exclude_source)
    rows = db.execute(
        """
        SELECT DISTINCT summary
        FROM memories
        WHERE is_archived = 0
          AND summary IS NOT NULL
        """
    ).fetchall()

    source_rows: dict[str, int] = {}
    for row in rows:
        source = normalize_source_summary(row["summary"])
        source_l = source.lower()
        if include_terms and not any(term in source_l for term in include_terms):
            continue
        if exclude_terms and any(term in source_l for term in exclude_terms):
            continue
        tag_name = source_summary_to_tag_name(source)
        tag_id = name_to_id.get(tag_name)
        if tag_id is None:
            continue
        source_rows[source] = tag_id

    return sorted(source_rows.items())


def _load_tag_names(db: Any) -> dict[int, str]:
    rows = db.execute("SELECT id, name FROM tags").fetchall()
    return {row["id"]: row["name"] for row in rows}


def _tag_allowed(
    tag: str,
    min_tag_length: int,
    require_letter: bool,
    tag_text: str | None,
    exact_tag_text: bool,
) -> bool:
    if len(tag) < min_tag_length:
        return False
    if require_letter and len(tag) == 1:
        return True
    if tag_text and _tag_text_matches(tag, tag_text, exact_tag_text):
        return True
    return not require_letter and not tag_text


def _pair_allowed(
    tag_a: str,
    tag_b: str,
    require_letter: bool,
    tag_text: str | None,
    exact_tag_text: bool,
) -> bool:
    if require_letter and not (len(tag_a) == 1 or len(tag_b) == 1):
        return False
    if tag_text:
        if not (
            _tag_text_matches(tag_a, tag_text, exact_tag_text)
            or _tag_text_matches(tag_b, tag_text, exact_tag_text)
        ):
            return False
    return True


def _tag_text_matches(tag: str, tag_text: str, exact: bool) -> bool:
    tag_l = tag.lower()
    text_l = tag_text.lower()
    if exact:
        return tag_l == text_l
    return text_l in tag_l


def _is_letter_sequence_pair(tag_a: str, tag_b: str) -> bool:
    tags = (tag_a.lower(), tag_b.lower())
    return (
        (len(tags[0]) == 1 and "sequence" in tags[1])
        or (len(tags[1]) == 1 and "sequence" in tags[0])
    )


def _motif_matches_text_filters(
    motif: VerticalSquareMotif,
    include_text: str | None,
    exclude_text: str | None,
) -> bool:
    haystack = " ".join([*motif.sources, *motif.tags]).lower()
    include_terms = _text_terms(include_text)
    exclude_terms = _text_terms(exclude_text)
    if include_terms and not any(term in haystack for term in include_terms):
        return False
    if exclude_terms and any(term in haystack for term in exclude_terms):
        return False
    return True


def _tensor_motif_matches_text_filters(
    motif: TensorVerticalSquareMotif,
    include_text: str | None,
    exclude_text: str | None,
) -> bool:
    haystack = " ".join(
        [*motif.sources, *motif.row_tags, *motif.column_tags]
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
