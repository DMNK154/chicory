"""Forest layer models: co-occurrence edges, bridge edges, blocks, memberships, adjacency, snapshots."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class CooccurrenceEdge(BaseModel):
    left_type: str
    left_id: str
    right_type: str
    right_id: str
    scope_type: str
    raw_count: float = 0.0
    expected_count: float = 0.0
    lift: float = 0.0
    pmi: float = 0.0
    co_strength: float = 0.0
    evidence_count: int = 0
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None


class ForestBlock(BaseModel):
    id: Optional[int] = None
    block_key: str
    block_type: str
    forest_type: str  # "cooccurrence" or "bridge"
    internal_density: float = 0.0
    external_bridge_strength: float = 0.0
    evidence_count: int = 0
    created_at: Optional[datetime] = None
    last_observed_at: Optional[datetime] = None


class BlockMembership(BaseModel):
    block_id: int
    target_type: str
    target_id: str
    membership_strength: float = 0.0
    evidence_count: int = 0
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None


class BridgeEdge(BaseModel):
    left_block_id: int
    right_block_id: int
    connection_strength: float = 0.0
    cluster_distance: float = 0.0
    rarity_bonus: float = 0.0
    bridge_strength: float = 0.0
    evidence_count: int = 0
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None


class BlockAdjacency(BaseModel):
    left_block_id: int
    right_block_id: int
    adjacency_type: str
    cooccurrence_weight: float = 0.0
    bridge_weight: float = 0.0
    evidence_count: int = 0
    last_observed_at: Optional[datetime] = None


class ForestSnapshot(BaseModel):
    id: Optional[int] = None
    snapshot_at: Optional[datetime] = None
    trigger_type: str
    trigger_id: Optional[str] = None
    touched_memory_ids: str = "[]"
    touched_tag_ids: str = "[]"
    touched_block_ids: str = "[]"
    co_edge_count: int = 0
    bridge_edge_count: int = 0
    block_count: int = 0
    notes: Optional[str] = None
