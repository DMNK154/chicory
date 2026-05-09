"""Episodic relational tensor models — memory-to-memory edge cache and temporal episodes."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TemporalEpisode(BaseModel):
    """A contiguous period of coherent tag-space activity."""

    id: Optional[int] = None
    tag_ids: list[int] = []
    status: str = "active"
    visit_count: int = 1
    operation_count: int = 0
    created_at: Optional[datetime] = None
    last_active_at: Optional[datetime] = None
    snapshot_at: Optional[datetime] = None


class EpisodeTransition(BaseModel):
    """An edge in the episode graph."""

    from_episode_id: int
    to_episode_id: int
    transition_type: str
    transition_at: Optional[datetime] = None
    metadata: Optional[dict] = None


class EpisodicEdge(BaseModel):
    """A single edge in the episodic relational tensor R(mem_a, mem_b).

    Captures memory-native relationships that the tag tensor cannot:
    narrative continuity, supersession, retrieval reinforcement, and
    cross-cluster bridge utility.
    """

    memory_a_id: str
    memory_b_id: str

    # Cheap / eager fields (computed at creation)
    semantic_strength: float = 0.0
    tag_projected_strength: float = 0.0
    co_retrieval_strength: float = 0.0
    temporal_proximity: float = 0.0
    source_proximity: float = 0.0

    # Projected tag tensor channels (multiplex)
    tag_semantic_projected: float = 0.0
    tag_sync_projected: float = 0.0
    tag_cooccurrence_projected: float = 0.0
    tag_inhibition_projected: float = 0.0
    tag_glyph_projected: float = 0.0

    # Expensive / lazy fields (computed on demand)
    retrieval_reinforcement: float = 0.0
    narrative_continuity: float = 0.0
    supersession_strength: float = 0.0
    supersession_direction: int = 0  # +1 = a supersedes b, -1 = b supersedes a, 0 = unknown
    contradiction_strength: float = 0.0
    bridge_strength: float = 0.0

    # Lifecycle
    activation_count: int = 0
    edge_status: str = "candidate"  # candidate, warm, mature, decaying, archived
    created_at: Optional[datetime] = None
    last_activated_at: Optional[datetime] = None
