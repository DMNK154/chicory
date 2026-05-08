"""Detect raw-only cross-project bridge cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chicory.layer3.cross_project_alignment import canonical_tag
from chicory.layer3.cross_project_middle import query_middle_layer

HIDDEN_BRIDGE_SYSTEM_PROMPT = """You are analyzing Chicory hidden bridge evidence.

Hidden bridges are raw-derived cross-project graph cells that are absent from a
stricter visible alignment. Treat them as candidate structural signals, not as
proven meanings. Ground your answer in the provided cells, separate observations
from hypotheses, and name practical follow-up tests when useful.

You may use Chicory read-only retrieval tools when they would clarify a term,
anchor, source, or memory pattern. Do not store memories or ingest documents.
"""


@dataclass(frozen=True)
class HiddenBridgeReport:
    """Raw bridge cells absent from the visible cross-project view."""

    project_a: str
    project_b: str
    visible_summary: dict[str, Any]
    raw_summary: dict[str, Any]
    hidden_cells: tuple[dict[str, Any], ...]
    parameters: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_a": self.project_a,
            "project_b": self.project_b,
            "visible_summary": self.visible_summary,
            "raw_summary": self.raw_summary,
            "hidden_count": len(self.hidden_cells),
            "hidden_cells": list(self.hidden_cells),
            "parameters": self.parameters,
        }


def find_hidden_bridges(
    visible_document: dict[str, Any],
    raw_document: dict[str, Any],
    *,
    query: str | None = None,
    limit: int = 20,
    layer_a: str | None = None,
    layer_b: str | None = None,
    cell_type: str | None = None,
) -> HiddenBridgeReport:
    """Compare visible and raw bridge documents and return raw-only cells."""
    visible_keys = {_cell_key(cell) for cell in visible_document.get("cells", [])}
    visible_neighborhood_anchors = {
        canonical_tag(str(cell.get("anchor", "")))
        for cell in visible_document.get("cells", [])
        if _cell_scope(cell) == "neighborhood"
    }
    visible_exact_anchors = {
        canonical_tag(str(cell.get("anchor", "")))
        for cell in visible_document.get("cells", [])
        if _cell_scope(cell) == "exact"
    }

    hidden_candidates = []
    for cell in raw_document.get("cells", []):
        if _cell_key(cell) in visible_keys:
            continue
        reason = _hidden_reason(
            cell,
            visible_neighborhood_anchors=visible_neighborhood_anchors,
            visible_exact_anchors=visible_exact_anchors,
        )
        hidden = dict(cell)
        hidden["hidden_reason"] = reason
        hidden["hidden_score"] = round(
            float(hidden.get("score", 0.0)) + _reason_bonus(reason),
            6,
        )
        hidden_candidates.append(hidden)

    filtered = query_middle_layer(
        {"cells": hidden_candidates},
        query,
        limit=max(limit, len(hidden_candidates)),
        layer_a=layer_a,
        layer_b=layer_b,
        cell_type=cell_type,
    )
    filtered.sort(
        key=lambda cell: (
            float(cell.get("query_score", cell.get("score", 0.0))),
            float(cell.get("hidden_score", 0.0)),
            float(cell.get("score", 0.0)),
        ),
        reverse=True,
    )

    return HiddenBridgeReport(
        project_a=str(raw_document.get("project_a", "")),
        project_b=str(raw_document.get("project_b", "")),
        visible_summary=dict(visible_document.get("summary", {})),
        raw_summary=dict(raw_document.get("summary", {})),
        hidden_cells=tuple(filtered[:limit]),
        parameters={
            "query": query,
            "limit": limit,
            "layer_a": layer_a,
            "layer_b": layer_b,
            "cell_type": cell_type,
            "raw_cell_count": len(raw_document.get("cells", [])),
            "visible_cell_count": len(visible_document.get("cells", [])),
            "hidden_candidate_count": len(hidden_candidates),
        },
    )


def format_hidden_bridge_context(
    report: HiddenBridgeReport,
    *,
    max_cells: int = 24,
    detail_limit: int = 4,
) -> str:
    """Format hidden bridge cells as compact LLM context."""
    visible = report.visible_summary
    raw = report.raw_summary
    lines = [
        "Hidden bridge scan",
        f"Projects: {report.project_a} <-> {report.project_b}",
        (
            "Visible vs raw shared anchors: "
            f"{visible.get('shared_tag_count', 0)} -> "
            f"{raw.get('shared_tag_count', 0)}"
        ),
        (
            "Visible vs raw exact edge pairs: "
            f"{visible.get('exact_pair_count', 0)} -> "
            f"{raw.get('exact_pair_count', 0)}"
        ),
        (
            "Visible vs raw cells: "
            f"{report.parameters.get('visible_cell_count', 0)} -> "
            f"{report.parameters.get('raw_cell_count', 0)}"
        ),
        f"Hidden candidate count: {report.parameters.get('hidden_candidate_count', 0)}",
        "",
        "Top hidden bridge cells:",
    ]
    for index, cell in enumerate(report.hidden_cells[:max_cells], 1):
        detail_a = _detail_text(cell.get("detail_a"), detail_limit=detail_limit)
        detail_b = _detail_text(cell.get("detail_b"), detail_limit=detail_limit)
        lines.extend(
            [
                (
                    f"{index}. anchor={cell.get('anchor', '')}; "
                    f"layers={cell.get('layer_a', '')}->{cell.get('layer_b', '')}; "
                    f"type={cell.get('scope', cell.get('cell_type', ''))}; "
                    f"reason={cell.get('hidden_reason', '')}; "
                    f"score={float(cell.get('score', 0.0)):.3f}; "
                    f"hidden_score={float(cell.get('hidden_score', 0.0)):.3f}"
                ),
                f"   {report.project_a}: {detail_a}",
                f"   {report.project_b}: {detail_b}",
            ]
        )
    return "\n".join(lines)


def build_hidden_bridge_prompt(
    report: HiddenBridgeReport,
    question: str,
    *,
    max_cells: int = 24,
    detail_limit: int = 4,
) -> str:
    """Build the user message for an LLM hidden bridge analysis."""
    context = format_hidden_bridge_context(
        report,
        max_cells=max_cells,
        detail_limit=detail_limit,
    )
    return (
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Please answer from the hidden bridge evidence. Call out the most useful "
        "anchors/layer lanes, likely interpretation, and the next concrete test."
    )


def _cell_key(cell: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _cell_scope(cell),
        canonical_tag(str(cell.get("anchor", ""))),
        canonical_tag(str(cell.get("layer_a", ""))),
        canonical_tag(str(cell.get("layer_b", ""))),
    )


def _cell_scope(cell: dict[str, Any]) -> str:
    scope = canonical_tag(str(cell.get("scope", "")))
    if scope in {"exact", "neighborhood"}:
        return scope
    cell_type = canonical_tag(str(cell.get("cell_type", "")))
    if cell_type == "exact-edge-pair":
        return "exact"
    return "neighborhood"


def _hidden_reason(
    cell: dict[str, Any],
    *,
    visible_neighborhood_anchors: set[str],
    visible_exact_anchors: set[str],
) -> str:
    anchor = canonical_tag(str(cell.get("anchor", "")))
    if _cell_scope(cell) == "exact":
        if anchor not in visible_exact_anchors:
            return "raw-only-exact-pair"
        return "raw-only-exact-layer"
    if anchor not in visible_neighborhood_anchors:
        return "raw-only-anchor"
    return "raw-only-layer-pair"


def _reason_bonus(reason: str) -> float:
    if reason == "raw-only-anchor":
        return 0.35
    if reason == "raw-only-exact-pair":
        return 0.30
    if reason == "raw-only-layer-pair":
        return 0.15
    if reason == "raw-only-exact-layer":
        return 0.10
    return 0.0


def _detail_text(values: object, *, detail_limit: int) -> str:
    if isinstance(values, list):
        return "; ".join(str(value) for value in values[:detail_limit]) or "none"
    if isinstance(values, tuple):
        return "; ".join(str(value) for value in values[:detail_limit]) or "none"
    return str(values or "none")
