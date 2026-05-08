"""Materialized cross-project middle layer queries."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chicory.layer3.cross_project_alignment import (
    CrossProjectAlignmentReport,
    canonical_tag,
)

MIDDLE_LAYER_SCHEMA = "chicory-cross-middle-v1"


def build_middle_layer_document(
    report: CrossProjectAlignmentReport,
    *,
    source_db_a: str | Path | None = None,
    source_db_b: str | Path | None = None,
) -> dict[str, Any]:
    """Build a standalone in-between layer from a cross-project report."""
    report_data = report.as_dict()
    cells = [
        _materialize_cell(cell, scope="neighborhood")
        for cell in report_data["top_cells"]
    ]
    cells.extend(
        _materialize_cell(cell, scope="exact")
        for cell in report_data["top_exact_cells"]
    )
    cells.sort(key=lambda cell: cell["score"], reverse=True)

    return {
        "schema": MIDDLE_LAYER_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_a": report.project_a,
        "project_b": report.project_b,
        "source_db_a": str(source_db_a) if source_db_a is not None else None,
        "source_db_b": str(source_db_b) if source_db_b is not None else None,
        "summary": {
            "edge_count_a": report.edge_count_a,
            "edge_count_b": report.edge_count_b,
            "shared_tag_count": report.shared_tag_count,
            "exact_pair_count": report.exact_pair_count,
            "strongest_neighborhood_pair": report_data[
                "strongest_neighborhood_pair"
            ],
            "strongest_exact_pair": report_data["strongest_exact_pair"],
        },
        "matrices": {
            "neighborhood": report_data["neighborhood_matrix"],
            "exact_pairs": report_data["exact_pair_matrix"],
        },
        "parameters": report_data["parameters"],
        "cells": cells,
    }


def write_middle_layer(
    report: CrossProjectAlignmentReport,
    path: str | Path,
    *,
    source_db_a: str | Path | None = None,
    source_db_b: str | Path | None = None,
) -> dict[str, Any]:
    """Write a middle-layer JSON document and return the document."""
    output_path = Path(path)
    document = build_middle_layer_document(
        report,
        source_db_a=source_db_a,
        source_db_b=source_db_b,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return document


def load_middle_layer(path: str | Path) -> dict[str, Any]:
    """Load a materialized middle-layer document."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("schema") != MIDDLE_LAYER_SCHEMA:
        raise ValueError(
            f"Unsupported middle-layer schema: {document.get('schema')!r}"
        )
    return document


def query_middle_layer(
    document: dict[str, Any],
    query: str | None = None,
    *,
    limit: int = 10,
    layer_a: str | None = None,
    layer_b: str | None = None,
    cell_type: str | None = None,
) -> list[dict[str, Any]]:
    """Search the materialized in-between cells without touching source DBs."""
    terms = _query_terms(query)
    layer_a_key = canonical_tag(layer_a or "") or None
    layer_b_key = canonical_tag(layer_b or "") or None
    cell_type_key = canonical_tag(cell_type or "") or None

    ranked: list[tuple[float, float, dict[str, Any]]] = []
    for cell in document.get("cells", []):
        if layer_a_key and canonical_tag(cell.get("layer_a", "")) != layer_a_key:
            continue
        if layer_b_key and canonical_tag(cell.get("layer_b", "")) != layer_b_key:
            continue
        if (
            cell_type_key
            and canonical_tag(cell.get("cell_type", "")) != cell_type_key
            and canonical_tag(cell.get("scope", "")) != cell_type_key
        ):
            continue

        text = cell.get("search_text") or _cell_search_text(cell)
        lexical_score = _lexical_score(terms, text, cell)
        if terms and lexical_score <= 0:
            continue
        score = float(cell.get("score", 0.0))
        result = dict(cell)
        result["query_score"] = round(lexical_score + score, 6)
        ranked.append((lexical_score, score, result))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in ranked[:limit]]


def _materialize_cell(cell: dict[str, Any], *, scope: str) -> dict[str, Any]:
    materialized = dict(cell)
    materialized["scope"] = scope
    materialized["search_text"] = _cell_search_text(materialized)
    return materialized


def _cell_search_text(cell: dict[str, Any]) -> str:
    fields: list[str] = [
        str(cell.get("scope", "")),
        str(cell.get("cell_type", "")),
        str(cell.get("anchor", "")),
        str(cell.get("layer_a", "")),
        str(cell.get("layer_b", "")),
    ]
    for key in ("detail_a", "detail_b"):
        values = cell.get(key, [])
        if isinstance(values, list):
            fields.extend(str(value) for value in values)
        else:
            fields.append(str(values))
    return " ".join(fields).lower()


def _query_terms(query: str | None) -> tuple[str, ...]:
    if not query:
        return ()
    terms = []
    for part in re.split(r"[^A-Za-z0-9:_\-.]+", query):
        term = canonical_tag(part)
        if term:
            terms.append(term)
    return tuple(terms)


def _lexical_score(
    terms: tuple[str, ...],
    text: str,
    cell: dict[str, Any],
) -> float:
    if not terms:
        return 0.0
    compact_text = canonical_tag(text)
    word_text = compact_text.replace("-", " ")
    anchor = canonical_tag(str(cell.get("anchor", "")))
    layers = {
        canonical_tag(str(cell.get("layer_a", ""))),
        canonical_tag(str(cell.get("layer_b", ""))),
    }
    score = 0.0
    for term in terms:
        if term == anchor:
            score += 5.0
        elif term in layers:
            score += 3.0
        elif term in compact_text or term.replace("-", " ") in word_text:
            score += 1.0
    return score
