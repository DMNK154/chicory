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
    total: float = 0.0

    def to_dict(self) -> dict:
        return {
            "memory_id": self.memory_id,
            "semantic": round(self.semantic, 4),
            "tag": round(self.tag, 4),
            "lattice": round(self.lattice, 4),
            "glyph": round(self.glyph, 4),
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
