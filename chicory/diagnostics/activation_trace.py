"""Activation trace — captures the full scoring breakdown for a retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ScoreBreakdown:
    """Per-memory score components from hybrid retrieval."""

    memory_id: str
    semantic: float = 0.0
    tag: float = 0.0
    lattice: float = 0.0
    glyph: float = 0.0
    convergence: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "semantic": round(self.semantic, 4),
            "tag": round(self.tag, 4),
            "lattice": round(self.lattice, 4),
            "glyph": round(self.glyph, 4),
            "convergence": round(self.convergence, 4),
            "total": round(self.total, 4),
        }


@dataclass
class ActivationTrace:
    """Full activation trace for a single retrieval event."""

    query: str
    method: str
    tags_requested: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Top-N tag activations from tensor/centroid path: {tag_name: strength}
    tag_activations: dict[str, float] = field(default_factory=dict)

    # Per-memory score breakdowns
    score_breakdowns: list[ScoreBreakdown] = field(default_factory=list)

    # Timing from hybrid retrieval
    timing: dict = field(default_factory=dict)

    # Context sent to the LLM (only populated when full logging is on)
    context_entries: list[dict] | None = None

    def to_dict(self, include_context: bool = False) -> dict:
        d: dict = {
            "query": self.query,
            "method": self.method,
            "tags_requested": self.tags_requested,
            "timestamp": self.timestamp,
            "tag_activations": {
                k: round(v, 4) for k, v in
                sorted(self.tag_activations.items(), key=lambda x: x[1], reverse=True)
            },
            "score_breakdowns": [s.to_dict() for s in self.score_breakdowns],
            "timing": self.timing,
        }
        if include_context and self.context_entries is not None:
            d["context"] = self.context_entries
        return d

    def _score_map(self) -> dict[str, ScoreBreakdown]:
        return {s.memory_id: s for s in self.score_breakdowns}


def compare_traces(a: ActivationTrace, b: ActivationTrace) -> dict:
    """Structured diff of two activation traces.

    Returns tag activation deltas, memory-level diffs (appeared/disappeared/
    shared with component-level deltas), and summary stats.
    """
    # ── Tag activation diff ──
    all_tags = set(a.tag_activations) | set(b.tag_activations)
    tag_diffs = []
    for tag in all_tags:
        va = a.tag_activations.get(tag, 0.0)
        vb = b.tag_activations.get(tag, 0.0)
        tag_diffs.append({
            "tag": tag,
            "a": round(va, 4),
            "b": round(vb, 4),
            "delta": round(vb - va, 4),
        })
    tag_diffs.sort(key=lambda x: abs(x["delta"]), reverse=True)

    # ── Memory diff ──
    scores_a = a._score_map()
    scores_b = b._score_map()
    ids_a = set(scores_a)
    ids_b = set(scores_b)

    appeared = []
    for mid in sorted(ids_b - ids_a):
        s = scores_b[mid]
        appeared.append(s.to_dict())

    disappeared = []
    for mid in sorted(ids_a - ids_b):
        s = scores_a[mid]
        disappeared.append(s.to_dict())

    shared = []
    for mid in sorted(ids_a & ids_b):
        sa = scores_a[mid]
        sb = scores_b[mid]
        shared.append({
            "memory_id": mid,
            "a_total": round(sa.total, 4),
            "b_total": round(sb.total, 4),
            "delta_total": round(sb.total - sa.total, 4),
            "delta_semantic": round(sb.semantic - sa.semantic, 4),
            "delta_tag": round(sb.tag - sa.tag, 4),
            "delta_lattice": round(sb.lattice - sa.lattice, 4),
            "delta_glyph": round(sb.glyph - sa.glyph, 4),
        })
    shared.sort(key=lambda x: abs(x["delta_total"]), reverse=True)

    # ── Rank changes ──
    rank_a = {s.memory_id: i for i, s in enumerate(a.score_breakdowns)}
    rank_b = {s.memory_id: i for i, s in enumerate(b.score_breakdowns)}
    rank_changes = []
    for mid in ids_a & ids_b:
        ra = rank_a[mid]
        rb = rank_b[mid]
        if ra != rb:
            rank_changes.append({
                "memory_id": mid,
                "rank_a": ra + 1,
                "rank_b": rb + 1,
                "shift": ra - rb,
            })
    rank_changes.sort(key=lambda x: abs(x["shift"]), reverse=True)

    # ── Summary ──
    total_a = sum(s.total for s in a.score_breakdowns)
    total_b = sum(s.total for s in b.score_breakdowns)

    return {
        "trace_a": {"query": a.query, "method": a.method, "timestamp": a.timestamp},
        "trace_b": {"query": b.query, "method": b.method, "timestamp": b.timestamp},
        "tag_activation_diffs": tag_diffs,
        "memories_appeared": appeared,
        "memories_disappeared": disappeared,
        "memories_shared": shared,
        "rank_changes": rank_changes,
        "summary": {
            "total_score_a": round(total_a, 4),
            "total_score_b": round(total_b, 4),
            "total_score_delta": round(total_b - total_a, 4),
            "memories_appeared": len(appeared),
            "memories_disappeared": len(disappeared),
            "memories_shared": len(shared),
            "rank_changes": len(rank_changes),
            "tag_activations_changed": sum(1 for d in tag_diffs if d["delta"] != 0),
        },
    }
